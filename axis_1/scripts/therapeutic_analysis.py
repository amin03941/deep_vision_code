import os
import json
import re
import time
from datetime import datetime
from google import genai


class TherapeuticAnalyzer:
    """
    - Gère 429 (rate limit / quota) avec retry/backoff
    - Si quota journalier (RPD) épuisé -> bascule vers un autre modèle
    - Si impossible -> génère un rapport fallback (sans LLM)
    """

    def __init__(
        self,
        api_key: str,
        primary_model: str = "gemini-3-flash",
        fallback_models=None,
        max_retries: int = 4,
    ):
        if not api_key or len(api_key) < 20:
            raise ValueError("Clé Gemini invalide. Utilise GEMINI_API_KEY (env) ou passe api_key.")

        self.client = genai.Client(api_key=api_key)

        self.primary_model = primary_model
        self.fallback_models = fallback_models or [
            "gemini-3-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ]
        # on évite le doublon
        self.fallback_models = list(dict.fromkeys([self.primary_model] + self.fallback_models))

        self.max_retries = max_retries

        self.microbiome_knowledge = """
CONNAISSANCES SCIENTIFIQUES (16S rRNA & Microbiome):

⚠️ LIMITATIONS:
- 16S rRNA -> taxonomie (quelles bactéries), pas une fonction métabolique directe
- Pas d’abondances absolues fiables
- Associations ≠ causalité

ASSOCIATIONS (littérature):
- FPG élevé / insulino-résistance : souvent associés à baisse de diversité, inflammation, moins de butyrate (associations)
- BMI élevé : associations avec dysbiose, parfois baisse d’Akkermansia (associations)

INTERVENTIONS (générales):
- + fibres / prébiotiques, régime méditerranéen, polyphénols
- activité physique, sommeil, stress
- probiotiques: effets variables selon études
"""

    # ---------- helpers erreurs ----------
    def _is_429(self, msg: str) -> bool:
        msg_low = msg.lower()
        return ("429" in msg_low) or ("resource_exhausted" in msg_low) or ("quota" in msg_low)

    def _extract_retry_seconds(self, msg: str) -> int:
        # Ex: "Please retry in 22.6s"
        m = re.search(r"retry in\s+([\d\.]+)s", msg.lower())
        if m:
            return max(1, int(float(m.group(1))))
        # Ex: "retry_delay { seconds: 22 }"
        m = re.search(r"retry_delay.*seconds\D+(\d+)", msg.lower(), re.DOTALL)
        if m:
            return max(1, int(m.group(1)))
        return 3  # default

    def _looks_like_daily_quota(self, msg: str) -> bool:
        msg_low = msg.lower()
        # indices fréquents dans les erreurs quotas
        return ("perday" in msg_low) or ("requestsperday" in msg_low) or ("rpd" in msg_low)

    # ---------- prompt ----------
    def _format_neighbors(self, neighbors):
        out = []
        for i, n in enumerate(neighbors[:5], 1):
            out.append(
                f"{i}. {n['subject_id']} (sim={n['similarity']:.3f}) | "
                f"class={n['class']} | FPG={n['fpg_mean']:.2f} | BMI={n['bmi']:.2f} | age={n['age']:.0f}"
            )
        return "\n".join(out)

    def _build_prompt(self, analysis_data):
        patient = analysis_data["patient"]
        neighbors = analysis_data["neighbors"]
        risk_score = analysis_data.get("risk_score", 0)

        return f"""
Tu es un expert en bioinformatique et santé computationnelle.

⚠️ Contraintes:
- Pas d’abondances bactériennes réelles
- Pas de fonctions métaboliques
- Hypothèses uniquement basées sur associations (littérature)
- Recommandations générales, non médicales

PATIENT:
- ID: {patient['subject_id']}
- Classe: {patient['class']}
- FPG Mean: {patient['clinical_data'].get('FPG_Mean', 'N/A')}
- BMI: {patient['clinical_data'].get('BMI', 'N/A')}
- Âge: {patient['clinical_data'].get('Adj.age', 'N/A')}
- OGTT: {patient['clinical_data'].get('OGTT', 'N/A')}

Top voisins:
{self._format_neighbors(neighbors)}

Risque voisinage:
- {risk_score*100:.1f}% des voisins sont Prediabetic/Diabetic
- Distribution: {json.dumps(analysis_data['class_distribution'], ensure_ascii=False)}

{self.microbiome_knowledge}

Rends un rapport structuré:
1) Analyse de similarité (ce que signifie ce % de voisins malades)
2) Hypothèses microbiome possibles (prudentes)
3) 3-5 recommandations générales (format: Reco → justification → association)
4) Disclaimers obligatoires (pas diagnostic, limites 16S, consulter pro)
Style: scientifique, honnête, non sensationnaliste.
"""

    # ---------- fallback local ----------
    def _fallback_report(self, analysis_data) -> str:
        p = analysis_data["patient"]
        risk = analysis_data.get("risk_score", 0)
        dist = analysis_data.get("class_distribution", {})
        neigh = analysis_data.get("neighbors", [])[:5]

        lines = []
        lines.append("📄 RAPPORT (Fallback local — Gemini indisponible)")
        lines.append("=" * 70)
        lines.append(f"Patient: {p['subject_id']} | Classe: {p['class']}")
        lines.append(f"FPG: {p['clinical_data'].get('FPG_Mean', 'N/A')} | BMI: {p['clinical_data'].get('BMI', 'N/A')} | Age: {p['clinical_data'].get('Adj.age', 'N/A')}")
        lines.append("")
        lines.append(f"Risque (voisins malades): {risk*100:.1f}%")
        lines.append(f"Distribution classes voisins: {json.dumps(dist, ensure_ascii=False)}")
        lines.append("")
        lines.append("Top 5 voisins:")
        for i, n in enumerate(neigh, 1):
            lines.append(f"- {i}. {n['subject_id']} | sim={n['similarity']:.3f} | {n['class']} | FPG={n['fpg_mean']:.2f} | BMI={n['bmi']:.2f}")
        lines.append("")
        lines.append("Recommandations générales (non médicales):")
        lines.append("- Augmenter fibres/prébiotiques → support diversité (association) → utile si paramètres glycémiques/IMC élevés")
        lines.append("- Régime méditerranéen → association amélioration métabolique")
        lines.append("- Activité physique régulière → association avec microbiome plus favorable")
        lines.append("- Sommeil/stress → association avec inflammation et stabilité microbienne")
        lines.append("")
        lines.append("DISCLAIMERS:")
        lines.append("- Analyse computationnelle, pas un diagnostic.")
        lines.append("- 16S rRNA = taxonomie, pas fonction métabolique.")
        lines.append("- Associations ≠ causalité. Consulter un professionnel de santé.")
        return "\n".join(lines)

    def _save_report(self, subject_id, report, analysis_data):
        os.makedirs("reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/report_{subject_id}_{ts}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("DONNÉES D'ANALYSE\n")
            f.write("=" * 70 + "\n")
            f.write(f"Risk Score: {analysis_data.get('risk_score', 0)*100:.1f}%\n")
            f.write(f"Distribution: {json.dumps(analysis_data.get('class_distribution', {}), ensure_ascii=False, indent=2)}\n")
        print(f"💾 Rapport sauvegardé: {filename}")

    # ---------- génération robuste ----------
    def generate_therapeutic_report(self, analysis_data):
        prompt = self._build_prompt(analysis_data)
        patient_id = analysis_data["patient"]["subject_id"]

        last_error = None

        for model in self.fallback_models:
            print(f"\n🤖 Tentative Gemini avec modèle: {model}")

            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    report = resp.text if hasattr(resp, "text") else str(resp)

                    self._save_report(patient_id, report, analysis_data)
                    return report

                except Exception as e:
                    msg = str(e)
                    last_error = msg

                    if not self._is_429(msg):
                        print(f"❌ Erreur Gemini (non-429): {msg}")
                        break

                    # 429: quota/rate limit
                    if self._looks_like_daily_quota(msg):
                        print(f"⚠️ Quota journalier (RPD) atteint pour {model}. On bascule de modèle.")
                        break  # change model

                    wait_s = self._extract_retry_seconds(msg)
                    wait_s = min(60, wait_s * attempt)  # petit backoff
                    print(f"⚠️ 429 rate limit. Retry {attempt}/{self.max_retries} dans {wait_s}s ...")
                    time.sleep(wait_s)

        # Si aucun modèle n'a marché
        print("\n⚠️ Gemini indisponible (quota/rate limit). Génération d'un rapport fallback local.")
        fallback = self._fallback_report(analysis_data)
        self._save_report(patient_id, fallback, analysis_data)
        print(f"🧾 Détail dernière erreur Gemini: {last_error}")
        return fallback


if __name__ == "__main__":
    # Exemple d’utilisation
    from scripts.search_neighbors import NeighborAnalyzer

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY manquante. Dans CMD: set GEMINI_API_KEY=YOUR_KEY puis relance.")

    analyzer = NeighborAnalyzer()
    analysis = analyzer.analyze_patient("Subject_CBVHYJ", top_k=10)

    if analysis:
        ta = TherapeuticAnalyzer(api_key=api_key, primary_model="gemini-3-flash")
        print(ta.generate_therapeutic_report(analysis))

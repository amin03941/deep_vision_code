from qdrant_client import QdrantClient
import pickle
import pandas as pd


class NeighborAnalyzer:
    def __init__(self, qdrant_path="./qdrant_data", collection_name="microbiome_vectors"):
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

        with open("vectors.pkl", "rb") as f:
            data = pickle.load(f)
            self.vectors_data = data["vectors"]
            self.vectorizer_config = data.get("vectorizer_config", {})

    def get_subject_vector(self, subject_id):
        for v in self.vectors_data:
            if v["subject_id"] == subject_id:
                return v
        return None

    # ✅ CORRECTION: wrapper compatible toutes versions qdrant-client
    def _qdrant_search(self, query_vector, limit):
        """
        Retourne une liste de résultats (ScoredPoint) avec payload.
        Compatible avec:
        - Anciennes versions: client.search(...)
        - Nouvelles versions: client.query_points(...)
        """
        # Ancienne API
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )

        # Nouvelle API
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            # query_points renvoie un objet avec .points
            return res.points

        raise AttributeError(
            "QdrantClient: ni 'search' ni 'query_points' n'existe. "
            "Mettez à jour qdrant-client: pip install -U qdrant-client"
        )

    def find_neighbors(self, subject_id, top_k=10):
        print(f"\n🔍 Recherche des voisins pour: {subject_id}")

        subject_vector = self.get_subject_vector(subject_id)
        if subject_vector is None:
            print(f"❌ Sujet {subject_id} non trouvé!")
            return None

        # ✅ Utilise le vecteur fusionné
        results = self._qdrant_search(
            query_vector=subject_vector["fused_vector"].tolist(),
            limit=top_k + 1,  # +1 car le sujet lui-même peut remonter
        )

        neighbors = []
        for r in results:
            # Selon version, payload peut être None si pas demandé
            payload = getattr(r, "payload", None) or {}

            # skip si c'est le même sujet
            if payload.get("subject_id") == subject_id:
                continue

            neighbors.append(
                {
                    "subject_id": payload.get("subject_id", "UNKNOWN"),
                    "similarity": float(getattr(r, "score", 0.0)),
                    "class": payload.get("class", "Unknown"),
                    "fpg_mean": float(payload.get("fpg_mean", 0.0) or 0.0),
                    "bmi": float(payload.get("bmi", 0.0) or 0.0),
                    "gender": payload.get("gender", "Unknown"),
                    "age": float(payload.get("age", 0.0) or 0.0),
                    "ogtt": float(payload.get("ogtt", 0.0) or 0.0),
                }
            )

        return neighbors[:top_k]

    def analyze_patient(self, subject_id, top_k=10):
        print(f"\n{'='*60}")
        print(f"📊 ANALYSE DU PATIENT: {subject_id}")
        print(f"{'='*60}")

        patient_data = self.get_subject_vector(subject_id)
        if not patient_data:
            print(f"❌ Sujet {subject_id} non trouvé dans vectors.pkl")
            return None

        print(f"\n👤 Profil du patient:")
        print(f"  • Classe: {patient_data.get('class', 'Unknown')}")
        print(f"  • FPG Mean: {patient_data['clinical_data'].get('FPG_Mean', 'N/A')}")
        print(f"  • BMI: {patient_data['clinical_data'].get('BMI', 'N/A')}")
        print(f"  • Âge: {patient_data['clinical_data'].get('Adj.age', 'N/A')}")
        print(f"  • Séquences: {patient_data.get('num_sequences', 'N/A')}")

        neighbors = self.find_neighbors(subject_id, top_k)
        if not neighbors:
            print("❌ Aucun voisin trouvé")
            return None

        print(f"\n🎯 Top {len(neighbors)} voisins similaires (génétiquement + cliniquement):")
        print("-" * 60)

        neighbors_df = pd.DataFrame(neighbors)

        for i, neighbor in enumerate(neighbors[:5], 1):
            print(f"{i}. {neighbor['subject_id']}")
            print(f"   Similarité: {neighbor['similarity']:.3f}")
            print(f"   Classe: {neighbor['class']}")
            print(f"   FPG: {neighbor['fpg_mean']:.2f}, BMI: {neighbor['bmi']:.2f}")
            print()

        class_distribution = neighbors_df["class"].value_counts()
        print(f"\n📈 Distribution des classes parmi les voisins:")
        for cls, count in class_distribution.items():
            print(f"  • {cls}: {count} ({count/len(neighbors)*100:.1f}%)")

        sick_neighbors = neighbors_df[neighbors_df["class"].isin(["Prediabetic", "Diabetic"])]
        risk_score = len(sick_neighbors) / len(neighbors)

        print(f"\n🎯 Évaluation du risque:")
        print(f"  • Score: {risk_score*100:.1f}% de voisins malades")
        if risk_score > 0.7:
            print("  • Niveau: ⚠️ RISQUE ÉLEVÉ")
        elif risk_score > 0.4:
            print("  • Niveau: ⚡ RISQUE MODÉRÉ")
        else:
            print("  • Niveau: ✅ RISQUE FAIBLE")

        healthy = neighbors_df[neighbors_df["class"] == "Control"]
        sick = neighbors_df[neighbors_df["class"].isin(["Prediabetic", "Diabetic"])]

        analysis = {
            "patient": patient_data,
            "neighbors": neighbors,
            "neighbors_df": neighbors_df,
            "healthy_neighbors": healthy.to_dict("records") if len(healthy) > 0 else [],
            "sick_neighbors": sick.to_dict("records") if len(sick) > 0 else [],
            "class_distribution": class_distribution.to_dict(),
            "risk_score": risk_score,
        }

        if len(healthy) > 0 and len(sick) > 0:
            print(f"\n⚖️ Comparaison Sains vs Malades (moyennes):")
            print(f"  FPG:  Sains={healthy['fpg_mean'].mean():.2f}  vs  Malades={sick['fpg_mean'].mean():.2f}")
            print(f"  BMI:  Sains={healthy['bmi'].mean():.2f}  vs  Malades={sick['bmi'].mean():.2f}")
            print(f"  Âge:  Sains={healthy['age'].mean():.2f}  vs  Malades={sick['age'].mean():.2f}")

            analysis["comparison"] = {
                "fpg_healthy": float(healthy["fpg_mean"].mean()),
                "fpg_sick": float(sick["fpg_mean"].mean()),
                "bmi_healthy": float(healthy["bmi"].mean()),
                "bmi_sick": float(sick["bmi"].mean()),
                "age_healthy": float(healthy["age"].mean()),
                "age_sick": float(sick["age"].mean()),
            }

        return analysis


if __name__ == "__main__":
    analyzer = NeighborAnalyzer()

    # Mets ici les IDs EXACTS présents dans vectors.pkl
    result = analyzer.analyze_patient("Subject_UDAXIH", top_k=10)
    result2 = analyzer.analyze_patient("Subject_CBVHYJ", top_k=10)

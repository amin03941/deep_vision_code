"""
🎯 ÉVALUATION DU SYSTÈME DE RECHERCHE DE VOISINS
Compatible avec toutes versions de qdrant-client
"""

from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest, Filter, FieldCondition, MatchValue
import pickle
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report

class SystemEvaluator:
    def __init__(self, qdrant_path="./qdrant_data", collection_name="microbiome_vectors"):
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        
        with open("vectors.pkl", 'rb') as f:
            data = pickle.load(f)
            self.vectors_data = data['vectors']
    
    def predict_class_by_neighbors(self, subject_id, top_k=5):
        """
        Prédit la classe d'un sujet basée sur ses voisins
        """
        # Trouve le vecteur du sujet
        subject_vector = None
        true_class = None
        
        for v in self.vectors_data:
            if v['subject_id'] == subject_id:
                subject_vector = v['fused_vector']
                true_class = v['class']
                break
        
        if subject_vector is None:
            return None, None
        
        # ✅ CORRECTION: Utilise query() au lieu de search()
        try:
            # Nouvelle API (v1.7+)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=subject_vector.tolist(),
                limit=top_k + 1
            ).points
        except AttributeError:
            try:
                # API moyenne (v1.1 - v1.6)
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=subject_vector.tolist(),
                    limit=top_k + 1
                )
            except AttributeError:
                # Ancienne API (v0.x)
                from qdrant_client.models import SearchRequest
                results = self.client.search(
                    collection_name=self.collection_name,
                    search_params=SearchRequest(
                        vector=subject_vector.tolist(),
                        limit=top_k + 1
                    )
                )
        
        # Collecte les classes des voisins (exclut le sujet lui-même)
        neighbor_classes = []
        for result in results:
            payload = result.payload if hasattr(result, 'payload') else result['payload']
            result_id = payload['subject_id']
            
            if result_id != subject_id:
                neighbor_classes.append(payload['class'])
        
        # Vote majoritaire
        if len(neighbor_classes) == 0:
            return None, true_class
        
        class_counts = Counter(neighbor_classes[:top_k])
        predicted_class = class_counts.most_common(1)[0][0]
        
        return predicted_class, true_class
    
    def evaluate_leave_one_out(self, top_k=7):
        """
        ✅ ÉVALUATION LEAVE-ONE-OUT
        """
        print(f"\n{'='*70}")
        print(f"📊 ÉVALUATION LEAVE-ONE-OUT (k={top_k})")
        print(f"{'='*70}")
        
        y_true = []
        y_pred = []
        
        print(f"\n🔄 Évaluation de {len(self.vectors_data)} sujets...")
        
        for i, v in enumerate(self.vectors_data):
            print(f"\r  Progression: {i+1}/{len(self.vectors_data)}", end="")
            
            try:
                predicted, true = self.predict_class_by_neighbors(
                    v['subject_id'],
                    top_k=top_k
                )
                
                if predicted is not None and true is not None:
                    y_true.append(true)
                    y_pred.append(predicted)
            except Exception as e:
                print(f"\n⚠️ Erreur pour {v['subject_id']}: {e}")
                continue
        
        print("\n")
        
        if len(y_true) == 0:
            print("❌ Aucune prédiction réussie!")
            return None
        
        # Calcul des métriques
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n📈 RÉSULTATS:")
        print(f"  • Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  • F1-Score (weighted): {f1_weighted:.3f}")
        
        # Rapport détaillé
        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        # Analyse par classe
        print(f"\n🔍 Analyse par classe:")
        classes = sorted(set(y_true))
        
        for cls in classes:
            true_count = y_true.count(cls)
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            print(f"  • {cls}: {correct}/{true_count} corrects ({correct/true_count*100:.1f}%)")
        
        # Verdict
        print(f"\n{'='*70}")
        if accuracy > 0.7:
            print("✅ SYSTÈME VALIDÉ - Le retrieval fonctionne bien!")
        elif accuracy > 0.5:
            print("⚠️ SYSTÈME MOYEN - Améliorer la vectorisation ou augmenter k")
        else:
            print("❌ SYSTÈME NON VALIDÉ - Voisins non pertinents, revoir l'approche")
        print(f"{'='*70}\n")
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def test_different_k_values(self, k_values=[3, 5, 7, 10, 15]):
        """
        Teste différentes valeurs de k
        """
        print(f"\n{'='*70}")
        print("🔬 TEST DE DIFFÉRENTES VALEURS DE K")
        print(f"{'='*70}\n")
        
        results = []
        
        for k in k_values:
            print(f"Testing k={k}...")
            eval_result = self.evaluate_leave_one_out(top_k=k)
            if eval_result:
                results.append({
                    'k': k,
                    'accuracy': eval_result['accuracy'],
                    'f1': eval_result['f1_weighted']
                })
        
        if len(results) > 0:
            print(f"\n📊 COMPARAISON DES K:")
            print(f"{'k':<5} {'Accuracy':<12} {'F1-Score':<12}")
            print("-" * 30)
            for r in results:
                print(f"{r['k']:<5} {r['accuracy']:<12.3f} {r['f1']:<12.3f}")
            
            # Meilleur k
            best = max(results, key=lambda x: x['f1'])
            print(f"\n✨ Meilleur k: {best['k']} (F1={best['f1']:.3f})")
        
        return results

if __name__ == "__main__":
    evaluator = SystemEvaluator()
    
    # Évaluation avec k=5
    evaluator.evaluate_leave_one_out(top_k=7)
    
    # Décommenter pour tester différentes valeurs de k
    # evaluator.test_different_k_values([3, 5, 7, 10, 15])
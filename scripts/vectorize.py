import numpy as np
import pandas as pd
import json
import hashlib
from sklearn.preprocessing import StandardScaler
import pickle

class MultimodalVectorizer:
    def __init__(self, k=6, genetic_dim=4096):
        """
        k: taille des k-mers
        genetic_dim: dimension du vecteur génétique (puissance de 2 recommandée: 2048, 4096, 8192)
        """
        self.k = k
        self.genetic_dim = genetic_dim
        self.scaler = StandardScaler()
        
        # Features cliniques
        self.clinical_features = [
            'FPG_Mean', 'IRIS', 'SSPG', 'FPG', 'BMI', 'OGTT', 'Adj.age'
        ]
        
    def kmer_hash_vector(self, sequences, k=6, dim=4096):
        """
        ✅ CORRECTION MAJEURE : Feature Hashing pour k-mers
        Tous les sujets ont maintenant un vecteur comparable
        """
        vector = np.zeros(dim, dtype=np.float32)
        total_kmers = 0
        
        for seq_data in sequences:
            sequence = seq_data['sequence']
            
            # Skip si séquence trop courte
            if len(sequence) < k:
                continue
            
            # Génère tous les k-mers
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i+k]
                
                # Hash stable vers un index
                h = int(hashlib.md5(kmer.encode()).hexdigest(), 16)
                idx = h % dim
                vector[idx] += 1.0
                total_kmers += 1
        
        # Normalisation par fréquence
        if total_kmers > 0:
            vector /= total_kmers
        
        # L2 normalization (pour cosine similarity)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        
        return vector
    
    def encode_clinical_data(self, clinical_dict):
        """
        Encode les données cliniques en vecteur numérique
        """
        vector = []
        
        for feature in self.clinical_features:
            value = clinical_dict.get(feature, np.nan)
            
            # Gestion valeurs manquantes
            if pd.isna(value) or value == '':
                vector.append(0.0)
            else:
                try:
                    vector.append(float(value))
                except:
                    vector.append(0.0)
        
        # Encodage catégoriel
        class_value = clinical_dict.get('Class', 'Control')
        class_encoding = {'Control': 0, 'Prediabetic': 1, 'Diabetic': 2}
        vector.append(class_encoding.get(class_value, 0))
        
        gender = clinical_dict.get('Gender', 'M')
        vector.append(1 if gender == 'M' else 0)
        
        return np.array(vector, dtype=np.float32)
    
    def create_multimodal_vector(self, subject_data, alpha=0.7):
        """
        ✅ CORRECTION : Fusion pondérée génétique + clinique
        alpha: poids de la partie génétique (0.5 = équilibre, 0.7 = génétique domine)
        """
        sequences = subject_data['sequences']
        
        # 1. Vecteur génétique (déjà L2-normalisé)
        genetic_vector = self.kmer_hash_vector(
            sequences, 
            k=self.k, 
            dim=self.genetic_dim
        )
        
        # 2. Vecteur clinique (sera standardisé plus tard)
        clinical_data = subject_data['clinical_data']
        clinical_vector = self.encode_clinical_data(clinical_data)
        
        return {
            'subject_id': subject_data['subject_id'],
            'genetic_vector': genetic_vector,
            'clinical_vector': clinical_vector,
            'clinical_data': clinical_data,
            'class': clinical_data.get('Class', 'Unknown'),
            'num_sequences': len(sequences)
        }
    
    def vectorize_all_subjects(self, subjects_data, alpha=0.7, output_file="vectors.pkl"):
        """
        Vectorise tous les sujets avec fusion pondérée
        """
        print(f"🧬 Vectorisation avec k={self.k}, dim={self.genetic_dim}, alpha={alpha}")
        
        all_vectors = []
        
        for i, subject in enumerate(subjects_data):
            print(f"\r  Vectorisation: {i+1}/{len(subjects_data)}", end="")
            
            if len(subject['sequences']) == 0:
                print(f"\n⚠️ Pas de séquences pour {subject['subject_id']}, skip")
                continue
            
            vector_data = self.create_multimodal_vector(subject, alpha)
            all_vectors.append(vector_data)
        
        print(f"\n✅ {len(all_vectors)} vecteurs créés")
        
        # Standardisation de la partie CLINIQUE uniquement
        clinical_matrix = np.array([v['clinical_vector'] for v in all_vectors])
        clinical_normalized = self.scaler.fit_transform(clinical_matrix)
        
        # Fusion pondérée : alpha * genetic + (1-alpha) * clinical
        for i, vector_data in enumerate(all_vectors):
            gen_vec = vector_data['genetic_vector']
            clin_vec = clinical_normalized[i]
            
            # Normalisation L2 de la partie clinique
            clin_norm = np.linalg.norm(clin_vec)
            if clin_norm > 0:
                clin_vec = clin_vec / clin_norm
            
            # Fusion pondérée
            fused_vector = np.concatenate([
                alpha * gen_vec,
                (1 - alpha) * clin_vec
            ])
            
            vector_data['fused_vector'] = fused_vector
            vector_data['clinical_normalized'] = clin_vec
        
        # Sauvegarde
        with open(output_file, 'wb') as f:
            pickle.dump({
                'vectors': all_vectors,
                'scaler': self.scaler,
                'vectorizer_config': {
                    'k': self.k,
                    'genetic_dim': self.genetic_dim,
                    'alpha': alpha,
                    'clinical_features': self.clinical_features
                }
            }, f)
        
        print(f"💾 Vecteurs sauvegardés dans {output_file}")
        
        # Statistiques
        total_dim = len(all_vectors[0]['fused_vector'])
        print(f"\n📊 Dimensions:")
        print(f"  • Génétique: {self.genetic_dim} (poids: {alpha})")
        print(f"  • Clinique: {len(all_vectors[0]['clinical_vector'])} (poids: {1-alpha})")
        print(f"  • Total fusionné: {total_dim}")
        
        return all_vectors

# Utilisation
if __name__ == "__main__":
    with open("processed_subjects.json", 'r') as f:
        subjects_data = json.load(f)
    
    # ✅ Paramètres recommandés pour compétition
    vectorizer = MultimodalVectorizer(
        k=6,              # k-mer size
        genetic_dim=4096  # 4096 ou 8192 pour meilleure résolution
    )
    
    vectors = vectorizer.vectorize_all_subjects(
        subjects_data,
        alpha=0.7,  # 70% génétique, 30% clinique
        output_file="vectors.pkl"
    )
    
    print("\n✨ Vectorisation corrigée complète!")
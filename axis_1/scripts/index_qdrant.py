from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pickle
import numpy as np

class QdrantIndexer:
    def __init__(self, collection_name="microbiome_vectors", path="./qdrant_data"):
        self.collection_name = collection_name
        self.client = QdrantClient(path=path)
        print(f"✅ Client Qdrant initialisé (stockage: {path})")
    
    def create_collection(self, vector_dim):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"🗑️ Collection existante supprimée")
        except:
            pass
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE
            )
        )
        print(f"📦 Collection '{self.collection_name}' créée (dimension: {vector_dim})")
    
    def index_vectors(self, vectors_data):
        """
        ✅ CORRECTION : Utilise fused_vector au lieu de normalized_vector
        """
        print(f"\n🔄 Indexation de {len(vectors_data)} vecteurs...")
        
        # ✅ Utilise le vecteur fusionné
        vector_dim = len(vectors_data[0]['fused_vector'])
        
        self.create_collection(vector_dim)
        
        points = []
        
        for i, vector_data in enumerate(vectors_data):
            payload = {
                'subject_id': vector_data['subject_id'],
                'class': vector_data['class'],
                'fpg_mean': float(vector_data['clinical_data'].get('FPG_Mean', 0)),
                'bmi': float(vector_data['clinical_data'].get('BMI', 0)),
                'gender': vector_data['clinical_data'].get('Gender', 'Unknown'),
                'ethnicity': vector_data['clinical_data'].get('Ethnicity', 'Unknown'),
                'age': float(vector_data['clinical_data'].get('Adj.age', 0)),
                'ogtt': float(vector_data['clinical_data'].get('OGTT', 0)),
                'num_sequences': int(vector_data.get('num_sequences', 0))
            }
            
            point = PointStruct(
                id=i,
                vector=vector_data['fused_vector'].tolist(),  # ✅ Correction
                payload=payload
            )
            
            points.append(point)
        
        # Upload par batch
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"  ✅ Batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} indexé")
        
        print(f"\n🎉 Indexation terminée!")
        
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        print(f"📊 Vecteurs dans la collection: {collection_info.points_count}")
    
    def search_neighbors(self, query_vector, top_k=5, filter_class=None):
        query_filter = None
        if filter_class:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="class",
                        match=MatchValue(value=filter_class)
                    )
                ]
            )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter
        )
        
        return results
    
    def get_collection_stats(self):
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        
        print(f"\n📊 Statistiques de la collection '{self.collection_name}':")
        print(f"  • Nombre de vecteurs: {collection_info.points_count}")
        print(f"  • Dimension: {collection_info.config.params.vectors.size}")
        print(f"  • Métrique: {collection_info.config.params.vectors.distance}")

if __name__ == "__main__":
    with open("vectors.pkl", 'rb') as f:
        data = pickle.load(f)
        vectors_data = data['vectors']
    
    indexer = QdrantIndexer()
    indexer.index_vectors(vectors_data)
    indexer.get_collection_stats()
    
    print("\n✨ Indexation Qdrant corrigée complète!")
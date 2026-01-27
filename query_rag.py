"""
Query your RAG system - Cross-modal search across text, sequence, and image collections
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import torch

class MicrobiomeRAG:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        text_collection: str = "microbiome_text",
        sequence_collection: str = "microbiome_sequence",
        image_collection: str = "microbiome_image",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG system for cross-modal search.
        
        Args:
            qdrant_url: Local Qdrant URL
            text_collection: Text collection name
            sequence_collection: Sequence collection name
            image_collection: Image collection name
            embedding_model: Embedding model name
        """
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.text_collection = text_collection
        self.sequence_collection = sequence_collection
        self.image_collection = image_collection
        
        # Load embedding model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        print(f"✅ RAG system initialized")
        print(f"   Model: {embedding_model} ({self.vector_size}-dim) on {device}")
        print(f"   Collections: {text_collection}, {sequence_collection}, {image_collection}")
    
    def search_text(self, query: str, limit: int = 5) -> List[Dict]:
        """Search in text collection"""
        query_vector = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        results = self.qdrant_client.search(
            collection_name=self.text_collection,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                'score': r.score,
                'payload': r.payload,
                'id': r.id,
                'type': 'text'
            }
            for r in results
        ]
    
    def search_sequence(self, query: str, limit: int = 5) -> List[Dict]:
        """Search in sequence collection (using text query converted to k-mer representation)"""
        # Convert query to k-mer like representation for sequence search
        query_vector = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        results = self.qdrant_client.search(
            collection_name=self.sequence_collection,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                'score': r.score,
                'payload': r.payload,
                'id': r.id,
                'type': 'sequence'
            }
            for r in results
        ]
    
    def search_image(self, query: str, limit: int = 5) -> List[Dict]:
        """Search in image collection"""
        query_vector = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        results = self.qdrant_client.search(
            collection_name=self.image_collection,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                'score': r.score,
                'payload': r.payload,
                'id': r.id,
                'type': 'image'
            }
            for r in results
        ]
    
    def cross_modal_search(
        self,
        query: str,
        limit_per_modality: int = 3,
        fusion_method: str = "late"
    ) -> List[Dict]:
        """
        Cross-modal search across all collections.
        
        Args:
            query: Search query
            limit_per_modality: Results per modality
            fusion_method: "late" (combine after search) or "early" (combine vectors)
        """
        # Search all modalities
        text_results = self.search_text(query, limit=limit_per_modality)
        sequence_results = self.search_sequence(query, limit=limit_per_modality)
        image_results = self.search_image(query, limit=limit_per_modality)
        
        # Late fusion: combine and sort by score
        all_results = text_results + sequence_results + image_results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        return all_results[:limit_per_modality * 3]  # Return top results
    
    def get_sample_info(self, sample_id: str) -> Dict:
        """Get full information about a sample"""
        # Search in text collection for sample
        results = self.qdrant_client.scroll(
            collection_name=self.text_collection,
            scroll_filter={
                "must": [{"key": "sample_id", "match": {"value": sample_id}}]
            },
            limit=1
        )
        
        if results[0]:
            return results[0][0].payload
        return {}


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Query Microbiome RAG')
    parser.add_argument('--query', type=str, required=True,
                       help='Search query')
    parser.add_argument('--modality', type=str, default='all',
                       choices=['text', 'sequence', 'image', 'all'],
                       help='Search modality')
    parser.add_argument('--limit', type=int, default=5,
                       help='Number of results')
    parser.add_argument('--qdrant-url', type=str, default='http://localhost:6333',
                       help='Qdrant URL')
    
    args = parser.parse_args()
    
    # Initialize RAG
    rag = MicrobiomeRAG(qdrant_url=args.qdrant_url)
    
    # Search
    print(f"\n🔍 Query: '{args.query}'")
    print("="*60)
    
    if args.modality == 'all':
        results = rag.cross_modal_search(args.query, limit_per_modality=args.limit)
        print(f"\n📊 Cross-Modal Results ({len(results)} total):")
    elif args.modality == 'text':
        results = rag.search_text(args.query, limit=args.limit)
        print(f"\n📊 Text Results ({len(results)}):")
    elif args.modality == 'sequence':
        results = rag.search_sequence(args.query, limit=args.limit)
        print(f"\n📊 Sequence Results ({len(results)}):")
    elif args.modality == 'image':
        results = rag.search_image(args.query, limit=args.limit)
        print(f"\n📊 Image Results ({len(results)}):")
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['type'].upper()}] Score: {result['score']:.4f}")
        payload = result['payload']
        print(f"   Sample: {payload.get('sample_id', 'N/A')}")
        print(f"   Type: {payload.get('sample_type', 'N/A')}")
        print(f"   Subject: {payload.get('subject_id', 'N/A')}")
        if 'text_description' in payload:
            print(f"   Description: {payload['text_description'][:100]}...")
        if 'image_type' in payload:
            print(f"   Image Type: {payload['image_type']}")


if __name__ == "__main__":
    # Example interactive usage
    rag = MicrobiomeRAG()
    
    # Example queries
    queries = [
        "diabetic patient with high BMI",
        "stool sample from prediabetic",
        "insulin resistant patient"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = rag.cross_modal_search(query, limit_per_modality=3)
        
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. [{result['type']}] Score: {result['score']:.4f}")
            payload = result['payload']
            print(f"   {payload.get('text_description', 'N/A')[:80]}...")

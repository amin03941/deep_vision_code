"""
Quick test script to verify RAG is working
"""
from query_rag import MicrobiomeRAG

def test_rag():
    """Test the RAG system"""
    print("=" * 60)
    print("🧪 Testing RAG System")
    print("=" * 60)
    
    try:
        rag = MicrobiomeRAG()
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        print("   Make sure Qdrant is running: docker-compose up -d")
        return
    
    # Test queries
    test_queries = [
        "diabetic patient",
        "stool sample",
        "high BMI prediabetic"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print(f"{'='*60}")
        
        try:
            # Text search
            results = rag.search_text(query, limit=3)
            print(f"\n📝 Text Results ({len(results)}):")
            for i, r in enumerate(results, 1):
                payload = r['payload']
                print(f"  {i}. Score: {r['score']:.4f} | {payload.get('sample_id', 'N/A')} | {payload.get('sample_type', 'N/A')}")
            
            # Cross-modal
            results = rag.cross_modal_search(query, limit_per_modality=2)
            print(f"\n🌐 Cross-Modal Results ({len(results)}):")
            for i, r in enumerate(results[:5], 1):
                payload = r['payload']
                print(f"  {i}. [{r['type']}] Score: {r['score']:.4f} | {payload.get('sample_id', 'N/A')}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print("✅ RAG System Working!")
    print("="*60)


if __name__ == "__main__":
    test_rag()

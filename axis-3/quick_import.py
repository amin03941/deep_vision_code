"""
Quick import script - imports all collections without prompts
"""
from import_all_collections_to_qdrant import import_collection
from qdrant_client import QdrantClient
import os

def quick_import_all(qdrant_url: str = "http://localhost:6333", batch_size: int = 100):
    """Import all collections without prompts"""
    print("=" * 60)
    print("Quick Import - All Collections")
    print("=" * 60)
    
    collections = {
        'text': {
            'file': 'qdrant_text_vectors.pkl',
            'name': 'microbiome_text'
        },
        'sequence': {
            'file': 'qdrant_sequence_vectors.pkl',
            'name': 'microbiome_sequence'
        },
        'image': {
            'file': 'qdrant_image_vectors.pkl',
            'name': 'microbiome_image'
        }
    }
    
    # Check Qdrant is running
    try:
        client = QdrantClient(url=qdrant_url)
        client.get_collections()
        print("[OK] Qdrant is running")
    except Exception as e:
        print(f"[ERROR] Cannot connect to Qdrant: {e}")
        print("   Start Qdrant first: docker-compose up -d")
        return
    
    # Import each collection
    for name, config in collections.items():
        if os.path.exists(config['file']):
            print(f"\n[IMPORT] Importing {name} collection...")
            try:
                # Delete existing if exists
                try:
                    client.delete_collection(config['name'])
                    print(f"   Deleted existing collection")
                except:
                    pass
                
                # Import
                import_collection(
                    vectors_file=config['file'],
                    collection_name=config['name'],
                    qdrant_url=qdrant_url,
                    batch_size=batch_size
                )
            except Exception as e:
                print(f"[ERROR] Error importing {name}: {e}")
        else:
            print(f"[ERROR] File not found: {config['file']}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Import Complete!")
    print(f"{'='*60}")
    
    client = QdrantClient(url=qdrant_url)
    for name, config in collections.items():
        try:
            info = client.get_collection(config['name'])
            print(f"[OK] {config['name']}: {info.points_count:,} points")
        except:
            print(f"[FAIL] {config['name']}: Failed")
    
    print(f"\n[INFO] Qdrant: {qdrant_url}")
    print(f"[INFO] Ready to query!")
    print("="*60)


if __name__ == "__main__":
    quick_import_all()

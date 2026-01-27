"""
Import all collections (text, sequence, image) from Kaggle to local Qdrant
"""
import pickle
import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

def import_collection(
    vectors_file: str,
    collection_name: str,
    qdrant_url: str = "http://localhost:6333",
    batch_size: int = 100
):
    """
    Import a single collection to Qdrant.
    
    Args:
        vectors_file: Path to .pkl file
        collection_name: Qdrant collection name
        qdrant_url: Local Qdrant URL
        batch_size: Vectors per batch insert
    """
    print(f"\n{'='*60}")
    print(f"[IMPORT] Importing: {collection_name}")
    print(f"{'='*60}")
    
    # Load vectors
    print(f"[LOAD] Loading: {vectors_file}")
    with open(vectors_file, 'rb') as f:
        vectors_data = pickle.load(f)
    
    print(f"[OK] Loaded {len(vectors_data):,} vectors")
    
    # Get vector size
    vector_size = len(vectors_data[0]['vector'])
    print(f"[INFO] Vector size: {vector_size}-dim")
    
    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url)
    
    # Create collection if needed (delete existing)
    try:
        collection_info = client.get_collection(collection_name)
        print(f"[WARN] Collection '{collection_name}' exists ({collection_info.points_count:,} points)")
        print(f"   Deleting existing collection...")
        client.delete_collection(collection_name)
        print(f"   [OK] Deleted existing collection")
    except:
        pass
    
    print(f"[CREATE] Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    print(f"[OK] Collection created")
    
    # Import vectors in batches
    print(f"[IMPORT] Importing vectors...")
    total = len(vectors_data)
    
    for i in tqdm(range(0, total, batch_size), desc=f"Importing {collection_name}"):
        batch = vectors_data[i:i+batch_size]
        
        points = []
        for vec_data in batch:
            # Handle different payload structures
            if 'payload' in vec_data:
                payload = vec_data['payload']
            else:
                # Fallback for old format
                payload = {
                    'text': vec_data.get('text', ''),
                    'source': vec_data.get('source', ''),
                    'row_id': vec_data.get('row_id', 0),
                    **vec_data.get('metadata', {})
                }
            
            points.append(
                PointStruct(
                    id=vec_data['id'],
                    vector=vec_data['vector'],
                    payload=payload
                )
            )
        
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    # Verify
    collection_info = client.get_collection(collection_name)
    print(f"[OK] Imported {collection_info.points_count:,} points to '{collection_name}'")


def import_all_collections(
    base_dir: str = ".",
    qdrant_url: str = "http://localhost:6333",
    batch_size: int = 100
):
    """
    Import all collections (text, sequence, image) to Qdrant.
    
    Args:
        base_dir: Directory containing the .pkl files
        qdrant_url: Local Qdrant URL
        batch_size: Vectors per batch insert
    """
    print("=" * 60)
    print("Import All Collections → Local Qdrant")
    print("=" * 60)
    
    # Define collections
    collections = {
        'text': {
            'file': os.path.join(base_dir, 'qdrant_text_vectors.pkl'),
            'name': 'microbiome_text'
        },
        'sequence': {
            'file': os.path.join(base_dir, 'qdrant_sequence_vectors.pkl'),
            'name': 'microbiome_sequence'
        },
        'image': {
            'file': os.path.join(base_dir, 'qdrant_image_vectors.pkl'),
            'name': 'microbiome_image'
        }
    }
    
    # Check files exist
    print("\n[CHECK] Checking files...")
    for name, config in collections.items():
        if os.path.exists(config['file']):
            size_mb = os.path.getsize(config['file']) / (1024 * 1024)
            print(f"[OK] {name}: {config['file']} ({size_mb:.1f} MB)")
        else:
            print(f"[ERROR] {name}: {config['file']} NOT FOUND")
            return
    
    # Import each collection
    for name, config in collections.items():
        import_collection(
            vectors_file=config['file'],
            collection_name=config['name'],
            qdrant_url=qdrant_url,
            batch_size=batch_size
        )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"[SUCCESS] All Collections Imported!")
    print(f"{'='*60}")
    
    client = QdrantClient(url=qdrant_url)
    for name, config in collections.items():
        try:
            info = client.get_collection(config['name'])
            print(f"[OK] {config['name']}: {info.points_count:,} points")
        except:
            print(f"[FAIL] {config['name']}: Failed")
    
    print(f"\n[INFO] Qdrant URL: {qdrant_url}")
    print(f"[INFO] Collections ready for cross-modal search!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Import all Kaggle collections to local Qdrant')
    parser.add_argument('--base-dir', type=str, default='.',
                       help='Directory containing .pkl files')
    parser.add_argument('--qdrant-url', type=str, default='http://localhost:6333',
                       help='Local Qdrant URL')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for import')
    parser.add_argument('--collection', type=str, default=None,
                       choices=['text', 'sequence', 'image'],
                       help='Import only specific collection (optional)')
    
    args = parser.parse_args()
    
    if args.collection:
        # Import single collection
        collections = {
            'text': {'file': 'qdrant_text_vectors.pkl', 'name': 'microbiome_text'},
            'sequence': {'file': 'qdrant_sequence_vectors.pkl', 'name': 'microbiome_sequence'},
            'image': {'file': 'qdrant_image_vectors.pkl', 'name': 'microbiome_image'}
        }
        config = collections[args.collection]
        import_collection(
            vectors_file=os.path.join(args.base_dir, config['file']),
            collection_name=config['name'],
            qdrant_url=args.qdrant_url,
            batch_size=args.batch_size
        )
    else:
        # Import all collections
        import_all_collections(
            base_dir=args.base_dir,
            qdrant_url=args.qdrant_url,
            batch_size=args.batch_size
        )

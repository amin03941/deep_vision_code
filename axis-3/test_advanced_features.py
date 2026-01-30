"""
Test script for advanced features
"""
from advanced_features import AdvancedMicrobiomeRAG

def test_variant_prioritizer():
    """Test variant prioritizer"""
    print("="*60)
    print("Testing Variant Prioritizer")
    print("="*60)
    
    rag = AdvancedMicrobiomeRAG()
    
    # Test 1: Prioritize by sequence similarity
    print("\n[Test 1] Prioritize by sequence similarity")
    results = rag.variant_prioritizer(
        reference_query="diabetic patient with high diversity",
        prioritize_by="sequence",
        limit=5
    )
    
    for i, result in enumerate(results, 1):
        payload = result['payload']
        print(f"{i}. {payload.get('sample_id', 'N/A')}")
        print(f"   Composite Score: {result['composite_score']:.3f}")
        print(f"   Similarity: {result['similarity_score']:.3f}, Clinical: {result['clinical_score']:.3f}")
        print(f"   Class: {payload.get('class', 'N/A')}, BMI: {payload.get('bmi', 'N/A')}")
    
    # Test 2: With clinical filters
    print("\n[Test 2] With clinical filters")
    results = rag.variant_prioritizer(
        reference_query="prediabetic insulin sensitive",
        clinical_filters={'class': 'Prediabetic', 'iris': 'IS'},
        limit=5
    )
    
    for i, result in enumerate(results, 1):
        payload = result['payload']
        print(f"{i}. {payload.get('sample_id', 'N/A')} - {payload.get('class', 'N/A')}, IRIS: {payload.get('iris', 'N/A')}")


def test_advanced_filter():
    """Test advanced filtering"""
    print("\n" + "="*60)
    print("Testing Advanced Filtering")
    print("="*60)
    
    rag = AdvancedMicrobiomeRAG()
    
    # Test 1: Filter by clinical parameters
    print("\n[Test 1] Filter by BMI and class")
    results = rag.advanced_filter(
        filters={
            'class': 'Diabetic',
            'bmi_min': 25,
            'bmi_max': 40,
            'iris': 'IR'
        },
        limit=10
    )
    
    print(f"Found {len(results)} results")
    for i, result in enumerate(results[:5], 1):
        payload = result['payload']
        print(f"{i}. {payload.get('sample_id', 'N/A')} - BMI: {payload.get('bmi', 'N/A')}, IRIS: {payload.get('iris', 'N/A')}")
    
    # Test 2: Search with filters
    print("\n[Test 2] Search with filters")
    results = rag.advanced_filter(
        query="high BMI prediabetic",
        filters={'bmi_min': 30},
        modality="text",
        limit=5
    )
    
    for i, result in enumerate(results, 1):
        payload = result['payload']
        print(f"{i}. {payload.get('sample_id', 'N/A')} - Score: {result['score']:.3f}, BMI: {payload.get('bmi', 'N/A')}")


def test_therapeutic_analysis():
    """Test therapeutic analysis"""
    print("\n" + "="*60)
    print("Testing Therapeutic Analysis")
    print("="*60)
    
    rag = AdvancedMicrobiomeRAG()
    
    analysis = rag.therapeutic_analysis(
        treatment_profile="insulin sensitive diabetic patient with successful intervention",
        outcome_filter={'iris': 'IS'},
        limit=10
    )
    
    print(f"Total matches: {analysis['total_matches']}")
    print(f"\nStatistics:")
    stats = analysis['statistics']
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")


def test_cluster_explorer():
    """Test cluster explorer"""
    print("\n" + "="*60)
    print("Testing Cluster Explorer")
    print("="*60)
    
    rag = AdvancedMicrobiomeRAG()
    
    clusters = rag.cluster_explorer(
        query="diabetic samples",
        n_clusters=3,
        modality="text",
        min_cluster_size=5
    )
    
    print(f"Found {clusters['n_clusters']} clusters")
    for cluster_id, cluster_info in clusters['clusters'].items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {cluster_info['size']} samples")
        chars = cluster_info['characteristics']
        print(f"  Dominant Class: {chars.get('dominant_class', 'N/A')}")
        print(f"  Dominant IRIS: {chars.get('dominant_iris', 'N/A')}")
        print(f"  Avg BMI: {chars.get('avg_bmi', 'N/A'):.2f}" if chars.get('avg_bmi') else "  Avg BMI: N/A")
        print(f"  Avg Age: {chars.get('avg_age', 'N/A'):.2f}" if chars.get('avg_age') else "  Avg Age: N/A")


if __name__ == "__main__":
    try:
        test_variant_prioritizer()
        test_advanced_filter()
        test_therapeutic_analysis()
        test_cluster_explorer()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

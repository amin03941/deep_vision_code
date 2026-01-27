"""
Advanced Features for Microbiome RAG System
- Variant Prioritizer
- Advanced Filtering
- Therapeutic Analysis
- Cluster Explorer
"""
from query_rag import MicrobiomeRAG
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict


class AdvancedMicrobiomeRAG(MicrobiomeRAG):
    """Extended RAG system with advanced features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def variant_prioritizer(
        self,
        reference_sample_id: Optional[str] = None,
        reference_query: Optional[str] = None,
        clinical_filters: Optional[Dict] = None,
        limit: int = 10,
        prioritize_by: str = "sequence"
    ) -> List[Dict]:
        """
        Prioritize variants/samples based on genomic and clinical similarity.
        
        Args:
            reference_sample_id: Sample ID to use as reference
            reference_query: Text query describing desired profile
            clinical_filters: Dict with filters like {'class': 'Diabetic', 'bmi_min': 25}
            limit: Number of results
            prioritize_by: 'sequence', 'text', or 'combined'
        
        Returns:
            List of prioritized samples with similarity scores
        """
        if reference_sample_id:
            # Get reference sample vector
            ref_info = self.get_sample_info(reference_sample_id)
            if not ref_info:
                return []
            
            # Build query from reference
            reference_query = ref_info.get('text_description', '')
        
        if not reference_query:
            return []
        
        # Search based on priority modality
        if prioritize_by == "sequence":
            results = self.search_sequence(reference_query, limit=limit * 2)
        elif prioritize_by == "text":
            results = self.search_text(reference_query, limit=limit * 2)
        else:  # combined
            results = self.cross_modal_search(reference_query, limit_per_modality=limit)
        
        # Apply clinical filters if provided
        if clinical_filters:
            results = self._apply_clinical_filters(results, clinical_filters)
        
        # Calculate composite scores (sequence similarity + clinical relevance)
        prioritized = []
        for result in results[:limit * 2]:
            payload = result['payload']
            
            # Base similarity score
            similarity_score = result['score']
            
            # Clinical relevance bonus
            clinical_score = 0.0
            if clinical_filters:
                clinical_score = self._calculate_clinical_relevance(payload, clinical_filters)
            
            # Composite score (weighted average)
            composite_score = 0.7 * similarity_score + 0.3 * clinical_score
            
            prioritized.append({
                **result,
                'composite_score': composite_score,
                'similarity_score': similarity_score,
                'clinical_score': clinical_score
            })
        
        # Sort by composite score
        prioritized.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return prioritized[:limit]
    
    def advanced_filter(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict] = None,
        modality: str = "text",
        limit: int = 10
    ) -> List[Dict]:
        """
        Advanced filtering with clinical parameters.
        
        Args:
            query: Optional text query
            filters: Dict with filters:
                - class: 'Diabetic', 'Prediabetic', 'Normal'
                - iris: 'IS', 'IR', 'Unknown'
                - fpg_class: 'Diabetes', 'Prediabetes', 'Normal'
                - bmi_min, bmi_max: BMI range
                - age_min, age_max: Age range
                - gender: 'M', 'F'
                - sample_type: 'Stool', etc.
            modality: 'text', 'sequence', 'image', or 'all'
            limit: Number of results
        
        Returns:
            Filtered results
        """
        # Build Qdrant filter
        qdrant_filter = self._build_qdrant_filter(filters)
        
        if query:
            # Vector search with filter
            query_vector = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
            
            if modality == "all":
                # Search all collections
                all_results = []
                for coll_name in [self.text_collection, self.sequence_collection, self.image_collection]:
                    results = self.qdrant_client.search(
                        collection_name=coll_name,
                        query_vector=query_vector,
                        query_filter=qdrant_filter,
                        limit=limit
                    )
                    all_results.extend([
                        {
                            'score': r.score,
                            'payload': r.payload,
                            'id': r.id,
                            'type': coll_name.split('_')[-1]
                        }
                        for r in results
                    ])
                all_results.sort(key=lambda x: x['score'], reverse=True)
                return all_results[:limit]
            else:
                collection = {
                    'text': self.text_collection,
                    'sequence': self.sequence_collection,
                    'image': self.image_collection
                }.get(modality, self.text_collection)
                
                results = self.qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    limit=limit
                )
                
                return [
                    {
                        'score': r.score,
                        'payload': r.payload,
                        'id': r.id,
                        'type': modality
                    }
                    for r in results
                ]
        else:
            # Just filter without search
            collection = self.text_collection
            results = self.qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=qdrant_filter,
                limit=limit
            )
            
            return [
                {
                    'score': 1.0,  # No similarity score for filtered-only
                    'payload': r.payload,
                    'id': r.id,
                    'type': 'text'
                }
                for r in results[0]
            ]
    
    def therapeutic_analysis(
        self,
        treatment_profile: str,
        outcome_filter: Optional[Dict] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze therapeutic profiles and find similar treatment cases.
        
        Args:
            treatment_profile: Description of treatment/intervention
            outcome_filter: Filter by outcomes (e.g., {'iris': 'IS'} for insulin sensitive)
            limit: Number of results
        
        Returns:
            Analysis results with treatment patterns
        """
        # Search for similar treatment profiles
        results = self.cross_modal_search(treatment_profile, limit_per_modality=5)
        
        if outcome_filter:
            results = self._apply_clinical_filters(results, outcome_filter)
        
        # Analyze patterns
        analysis = {
            'total_matches': len(results),
            'samples': results[:limit],
            'statistics': self._calculate_therapeutic_stats(results),
            'recommendations': []
        }
        
        # Generate recommendations based on patterns
        if analysis['statistics']:
            stats = analysis['statistics']
            if stats.get('iris_is_rate', 0) > 0.5:
                analysis['recommendations'].append(
                    "High proportion of insulin-sensitive cases found. Consider similar intervention strategies."
                )
            if stats.get('avg_bmi', 0) > 30:
                analysis['recommendations'].append(
                    "High BMI cases prevalent. Weight management interventions may be beneficial."
                )
        
        return analysis
    
    def cluster_explorer(
        self,
        query: Optional[str] = None,
        n_clusters: int = 5,
        modality: str = "text",
        min_cluster_size: int = 3
    ) -> Dict[str, Any]:
        """
        Explore clusters of similar samples.
        
        Args:
            query: Optional query to focus clustering
            n_clusters: Number of clusters
            modality: 'text', 'sequence', or 'image'
            min_cluster_size: Minimum samples per cluster
        
        Returns:
            Cluster information
        """
        from sklearn.cluster import KMeans
        
        # Get sample vectors
        collection = {
            'text': self.text_collection,
            'sequence': self.sequence_collection,
            'image': self.image_collection
        }.get(modality, self.text_collection)
        
        # Retrieve all vectors (or subset if query provided)
        if query:
            query_vector = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
            results = self.qdrant_client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=100  # Get top 100 for clustering
            )
            vectors = [r.vector for r in results]
            payloads = [r.payload for r in results]
        else:
            # Get all vectors (may be slow for large collections)
            scroll_results = self.qdrant_client.scroll(
                collection_name=collection,
                limit=500  # Limit for performance
            )
            vectors = [r.vector for r in scroll_results[0]]
            payloads = [r.payload for r in scroll_results[0]]
        
        if len(vectors) < n_clusters:
            return {'error': 'Not enough samples for clustering'}
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)
        
        # Organize clusters
        clusters = defaultdict(list)
        for i, (label, payload) in enumerate(zip(cluster_labels, payloads)):
            clusters[label].append({
                'sample_id': payload.get('sample_id', f'unknown_{i}'),
                'payload': payload
            })
        
        # Filter small clusters
        clusters = {
            k: v for k, v in clusters.items()
            if len(v) >= min_cluster_size
        }
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id, samples in clusters.items():
            cluster_analysis[cluster_id] = {
                'size': len(samples),
                'samples': samples[:10],  # Limit samples shown
                'characteristics': self._analyze_cluster_characteristics(samples)
            }
        
        return {
            'n_clusters': len(clusters),
            'clusters': cluster_analysis,
            'modality': modality
        }
    
    def _build_qdrant_filter(self, filters: Optional[Dict]) -> Optional[Filter]:
        """Build Qdrant filter from clinical parameters"""
        if not filters:
            return None
        
        conditions = []
        
        if 'class' in filters:
            conditions.append(
                FieldCondition(key="class", match=MatchValue(value=filters['class']))
            )
        
        if 'iris' in filters:
            conditions.append(
                FieldCondition(key="iris", match=MatchValue(value=filters['iris']))
            )
        
        if 'fpg_class' in filters:
            conditions.append(
                FieldCondition(key="fpg_class", match=MatchValue(value=filters['fpg_class']))
            )
        
        if 'gender' in filters:
            conditions.append(
                FieldCondition(key="gender", match=MatchValue(value=filters['gender']))
            )
        
        if 'sample_type' in filters:
            conditions.append(
                FieldCondition(key="sample_type", match=MatchValue(value=filters['sample_type']))
            )
        
        if 'bmi_min' in filters or 'bmi_max' in filters:
            bmi_range = {}
            if 'bmi_min' in filters:
                bmi_range['gte'] = filters['bmi_min']
            if 'bmi_max' in filters:
                bmi_range['lte'] = filters['bmi_max']
            conditions.append(
                FieldCondition(key="bmi", range=Range(**bmi_range))
            )
        
        if 'age_min' in filters or 'age_max' in filters:
            age_range = {}
            if 'age_min' in filters:
                age_range['gte'] = filters['age_min']
            if 'age_max' in filters:
                age_range['lte'] = filters['age_max']
            conditions.append(
                FieldCondition(key="age", range=Range(**age_range))
            )
        
        if not conditions:
            return None
        
        return Filter(must=conditions)
    
    def _apply_clinical_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply clinical filters to results"""
        filtered = []
        for result in results:
            payload = result['payload']
            match = True
            
            if 'class' in filters and payload.get('class') != filters['class']:
                match = False
            if 'iris' in filters and payload.get('iris') != filters['iris']:
                match = False
            if 'fpg_class' in filters and payload.get('fpg_class') != filters['fpg_class']:
                match = False
            if 'gender' in filters and payload.get('gender') != filters['gender']:
                match = False
            if 'sample_type' in filters and payload.get('sample_type') != filters['sample_type']:
                match = False
            if 'bmi_min' in filters and payload.get('bmi', 0) < filters['bmi_min']:
                match = False
            if 'bmi_max' in filters and payload.get('bmi', 999) > filters['bmi_max']:
                match = False
            if 'age_min' in filters and payload.get('age', 0) < filters['age_min']:
                match = False
            if 'age_max' in filters and payload.get('age', 999) > filters['age_max']:
                match = False
            
            if match:
                filtered.append(result)
        
        return filtered
    
    def _calculate_clinical_relevance(self, payload: Dict, filters: Dict) -> float:
        """Calculate clinical relevance score (0-1)"""
        score = 0.0
        matches = 0
        total = 0
        
        for key, value in filters.items():
            if key in ['bmi_min', 'bmi_max', 'age_min', 'age_max']:
                continue  # Handled separately
            
            total += 1
            if payload.get(key) == value:
                matches += 1
        
        if total > 0:
            score = matches / total
        
        return score
    
    def _calculate_therapeutic_stats(self, results: List[Dict]) -> Dict:
        """Calculate therapeutic statistics from results"""
        if not results:
            return {}
        
        stats = {
            'total': len(results),
            'iris_is': 0,
            'iris_ir': 0,
            'iris_unknown': 0,
            'diabetic': 0,
            'prediabetic': 0,
            'normal': 0,
            'bmi_values': [],
            'age_values': []
        }
        
        for result in results:
            payload = result['payload']
            
            iris = payload.get('iris', 'Unknown')
            if iris == 'IS':
                stats['iris_is'] += 1
            elif iris == 'IR':
                stats['iris_ir'] += 1
            else:
                stats['iris_unknown'] += 1
            
            class_val = payload.get('class', '')
            if 'Diabetic' in class_val:
                stats['diabetic'] += 1
            elif 'Prediabetic' in class_val:
                stats['prediabetic'] += 1
            else:
                stats['normal'] += 1
            
            if payload.get('bmi'):
                stats['bmi_values'].append(payload['bmi'])
            if payload.get('age'):
                stats['age_values'].append(payload['age'])
        
        # Calculate rates
        if stats['total'] > 0:
            stats['iris_is_rate'] = stats['iris_is'] / stats['total']
            stats['iris_ir_rate'] = stats['iris_ir'] / stats['total']
            stats['diabetic_rate'] = stats['diabetic'] / stats['total']
            stats['prediabetic_rate'] = stats['prediabetic'] / stats['total']
            stats['avg_bmi'] = np.mean(stats['bmi_values']) if stats['bmi_values'] else 0
            stats['avg_age'] = np.mean(stats['age_values']) if stats['age_values'] else 0
        
        return stats
    
    def _analyze_cluster_characteristics(self, samples: List[Dict]) -> Dict:
        """Analyze characteristics of a cluster"""
        if not samples:
            return {}
        
        characteristics = {
            'common_classes': defaultdict(int),
            'common_iris': defaultdict(int),
            'avg_bmi': [],
            'avg_age': []
        }
        
        for sample in samples:
            payload = sample.get('payload', {})
            
            if payload.get('class'):
                characteristics['common_classes'][payload['class']] += 1
            if payload.get('iris'):
                characteristics['common_iris'][payload['iris']] += 1
            if payload.get('bmi'):
                characteristics['avg_bmi'].append(payload['bmi'])
            if payload.get('age'):
                characteristics['avg_age'].append(payload['age'])
        
        # Summarize
        return {
            'dominant_class': max(characteristics['common_classes'].items(), key=lambda x: x[1])[0] if characteristics['common_classes'] else None,
            'dominant_iris': max(characteristics['common_iris'].items(), key=lambda x: x[1])[0] if characteristics['common_iris'] else None,
            'avg_bmi': np.mean(characteristics['avg_bmi']) if characteristics['avg_bmi'] else None,
            'avg_age': np.mean(characteristics['avg_age']) if characteristics['avg_age'] else None
        }


def main():
    """Example usage of advanced features"""
    rag = AdvancedMicrobiomeRAG()
    
    print("="*60)
    print("Advanced Features Demo")
    print("="*60)
    
    # 1. Variant Prioritizer
    print("\n[1] Variant Prioritizer")
    print("-"*60)
    prioritized = rag.variant_prioritizer(
        reference_query="diabetic patient with high BMI, insulin resistant",
        clinical_filters={'class': 'Diabetic', 'bmi_min': 25},
        limit=5
    )
    for i, result in enumerate(prioritized, 1):
        print(f"{i}. {result['payload']['sample_id']} - Score: {result['composite_score']:.3f}")
    
    # 2. Advanced Filtering
    print("\n[2] Advanced Filtering")
    print("-"*60)
    filtered = rag.advanced_filter(
        query="prediabetic patient",
        filters={'iris': 'IS', 'bmi_min': 25, 'bmi_max': 35},
        limit=5
    )
    for i, result in enumerate(filtered, 1):
        print(f"{i}. {result['payload']['sample_id']} - BMI: {result['payload'].get('bmi', 'N/A')}")
    
    # 3. Therapeutic Analysis
    print("\n[3] Therapeutic Analysis")
    print("-"*60)
    analysis = rag.therapeutic_analysis(
        treatment_profile="insulin sensitive diabetic patient",
        outcome_filter={'iris': 'IS'},
        limit=10
    )
    print(f"Total matches: {analysis['total_matches']}")
    print(f"Statistics: {analysis['statistics']}")
    print(f"Recommendations: {analysis['recommendations']}")
    
    # 4. Cluster Explorer
    print("\n[4] Cluster Explorer")
    print("-"*60)
    clusters = rag.cluster_explorer(
        query="diabetic samples",
        n_clusters=3,
        modality="text"
    )
    print(f"Found {clusters['n_clusters']} clusters")
    for cluster_id, cluster_info in clusters['clusters'].items():
        print(f"Cluster {cluster_id}: {cluster_info['size']} samples")
        print(f"  Characteristics: {cluster_info['characteristics']}")


if __name__ == "__main__":
    main()

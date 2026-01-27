"""
Knowledge Graph Builder for Microbiome Data
Constructs relationships from vector similarity and metadata
"""
from advanced_features import AdvancedMicrobiomeRAG
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
import json


class MicrobiomeKnowledgeGraph:
    """Build and explore knowledge graphs from microbiome data"""
    
    def __init__(self, rag: AdvancedMicrobiomeRAG):
        self.rag = rag
        self.graph = nx.Graph()
    
    def build_graph(
        self,
        sample_ids: Optional[List[str]] = None,
        similarity_threshold: float = 0.7,
        include_metadata_edges: bool = True
    ) -> nx.Graph:
        """
        Build knowledge graph from samples.
        
        Args:
            sample_ids: List of sample IDs to include (None = all)
            similarity_threshold: Minimum similarity for edges
            include_metadata_edges: Include edges from metadata relationships
        
        Returns:
            NetworkX graph
        """
        self.graph = nx.Graph()
        
        # Get samples
        if sample_ids is None:
            # Get all samples from text collection
            scroll_results = self.rag.qdrant_client.scroll(
                collection_name=self.rag.text_collection,
                limit=1000  # Limit for performance
            )
            samples = [r.payload for r in scroll_results[0]]
        else:
            samples = []
            for sample_id in sample_ids:
                info = self.rag.get_sample_info(sample_id)
                if info:
                    samples.append(info)
        
        # Add nodes
        for sample in samples:
            sample_id = sample.get('sample_id', 'unknown')
            self.graph.add_node(sample_id, **sample)
        
        # Add similarity edges (sequence-based)
        print(f"Building similarity edges...")
        for i, sample1 in enumerate(samples):
            sample_id1 = sample1.get('sample_id', 'unknown')
            
            # Search for similar sequences
            query = sample1.get('text_description', '')
            if query:
                similar = self.rag.search_sequence(query, limit=10)
                
                for result in similar:
                    sample_id2 = result['payload'].get('sample_id', 'unknown')
                    if sample_id1 != sample_id2 and result['score'] >= similarity_threshold:
                        self.graph.add_edge(
                            sample_id1,
                            sample_id2,
                            similarity=result['score'],
                            edge_type='sequence_similarity'
                        )
        
        # Add metadata-based edges
        if include_metadata_edges:
            print(f"Building metadata edges...")
            self._add_metadata_edges(samples)
        
        return self.graph
    
    def _add_metadata_edges(self, samples: List[Dict]):
        """Add edges based on metadata relationships"""
        # Group by subject
        subjects = {}
        for sample in samples:
            subject_id = sample.get('subject_id')
            sample_id = sample.get('sample_id')
            if subject_id and sample_id:
                if subject_id not in subjects:
                    subjects[subject_id] = []
                subjects[subject_id].append(sample_id)
        
        # Connect samples from same subject
        for subject_id, sample_ids in subjects.items():
            for i, sid1 in enumerate(sample_ids):
                for sid2 in sample_ids[i+1:]:
                    if not self.graph.has_edge(sid1, sid2):
                        self.graph.add_edge(
                            sid1,
                            sid2,
                            edge_type='same_subject',
                            subject_id=subject_id
                        )
        
        # Connect samples with same clinical class
        classes = {}
        for sample in samples:
            class_val = sample.get('class')
            sample_id = sample.get('sample_id')
            if class_val and sample_id:
                if class_val not in classes:
                    classes[class_val] = []
                classes[class_val].append(sample_id)
        
        for class_val, sample_ids in classes.items():
            for i, sid1 in enumerate(sample_ids):
                for sid2 in sample_ids[i+1:]:
                    if not self.graph.has_edge(sid1, sid2):
                        self.graph.add_edge(
                            sid1,
                            sid2,
                            edge_type='same_class',
                            class_value=class_val
                        )
    
    def get_central_samples(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most central samples (by degree centrality)"""
        centrality = nx.degree_centrality(self.graph)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_centrality[:top_n]
    
    def find_communities(self) -> Dict[int, List[str]]:
        """Find communities in the graph"""
        communities = nx.community.greedy_modularity_communities(self.graph)
        return {i: list(comm) for i, comm in enumerate(communities)}
    
    def get_path_between(self, sample_id1: str, sample_id2: str) -> List[str]:
        """Find shortest path between two samples"""
        try:
            path = nx.shortest_path(self.graph, sample_id1, sample_id2)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def export_graph(self, filename: str):
        """Export graph to JSON format"""
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    **self.graph.nodes[node]
                }
                for node in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    **self.graph.edges[edge]
                }
                for edge in self.graph.edges()
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Graph exported to {filename}")
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'number_of_components': nx.number_connected_components(self.graph)
        }


def main():
    """Example usage"""
    from advanced_features import AdvancedMicrobiomeRAG
    
    rag = AdvancedMicrobiomeRAG()
    kg = MicrobiomeKnowledgeGraph(rag)
    
    print("Building knowledge graph...")
    graph = kg.build_graph(
        similarity_threshold=0.75,
        include_metadata_edges=True
    )
    
    print(f"\nGraph Statistics:")
    stats = kg.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nMost Central Samples:")
    central = kg.get_central_samples(top_n=5)
    for sample_id, centrality_score in central:
        print(f"  {sample_id}: {centrality_score:.3f}")
    
    print(f"\nCommunities:")
    communities = kg.find_communities()
    for comm_id, samples in list(communities.items())[:3]:
        print(f"  Community {comm_id}: {len(samples)} samples")
        print(f"    Samples: {samples[:5]}...")


if __name__ == "__main__":
    main()

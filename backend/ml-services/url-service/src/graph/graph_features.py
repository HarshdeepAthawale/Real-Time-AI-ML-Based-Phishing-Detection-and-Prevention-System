"""Extract graph-based features"""
import networkx as nx
from typing import Dict


class GraphFeatureExtractor:
    """Extract features from domain relationship graphs"""
    
    def extract_features(self, graph: nx.Graph) -> Dict:
        """
        Extract graph features
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary of graph features
        """
        features = {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
            "is_connected": nx.is_connected(graph) if graph.number_of_nodes() > 1 else True
        }
        
        # Average degree
        if graph.number_of_nodes() > 0:
            degrees = dict(graph.degree())
            features["average_degree"] = sum(degrees.values()) / len(degrees)
        else:
            features["average_degree"] = 0.0
        
        # Clustering coefficient
        try:
            features["clustering_coefficient"] = nx.average_clustering(graph)
        except Exception:
            features["clustering_coefficient"] = 0.0
        
        return features

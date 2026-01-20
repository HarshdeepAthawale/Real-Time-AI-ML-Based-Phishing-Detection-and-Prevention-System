import networkx as nx
from typing import Dict, List
import numpy as np

class GraphFeatureExtractor:
    def __init__(self):
        pass
    
    def extract_features(self, graph: nx.DiGraph, node: str) -> Dict:
        """Extract graph-based features for a node"""
        if node not in graph:
            return {}
        
        features = {}
        
        # Basic degree features
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        total_degree = in_degree + out_degree
        
        features["in_degree"] = in_degree
        features["out_degree"] = out_degree
        features["total_degree"] = total_degree
        
        # Neighbor features
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))
        
        features["num_predecessors"] = len(predecessors)
        features["num_successors"] = len(successors)
        
        # Clustering coefficient
        try:
            clustering = nx.clustering(graph.to_undirected(), node)
            features["clustering_coefficient"] = clustering
        except:
            features["clustering_coefficient"] = 0.0
        
        # PageRank (if calculated)
        if hasattr(graph, 'pagerank'):
            features["pagerank"] = graph.pagerank.get(node, 0.0)
        
        return features
    
    def extract_global_features(self, graph: nx.DiGraph) -> Dict:
        """Extract global graph features"""
        features = {}
        
        features["num_nodes"] = graph.number_of_nodes()
        features["num_edges"] = graph.number_of_edges()
        features["density"] = nx.density(graph)
        
        # Check if graph is connected
        if graph.number_of_nodes() > 0:
            try:
                is_connected = nx.is_weakly_connected(graph)
                features["is_connected"] = is_connected
            except:
                features["is_connected"] = False
        
        return features

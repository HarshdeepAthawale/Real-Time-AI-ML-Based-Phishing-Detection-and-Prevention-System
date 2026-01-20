import networkx as nx
from typing import Dict, List, Set, Tuple
import numpy as np

class GraphUtils:
    @staticmethod
    def calculate_pagerank(graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate PageRank for nodes in graph"""
        try:
            pagerank = nx.pagerank(graph)
            return pagerank
        except:
            return {}
    
    @staticmethod
    def calculate_centrality(graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate betweenness centrality"""
        try:
            centrality = nx.betweenness_centrality(graph)
            return centrality
        except:
            return {}
    
    @staticmethod
    def find_communities(graph: nx.DiGraph) -> List[Set[str]]:
        """Find communities in graph using Louvain algorithm"""
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(graph.to_undirected())
            return communities
        except:
            return []
    
    @staticmethod
    def get_node_degree(graph: nx.DiGraph, node: str) -> Tuple[int, int]:
        """Get in-degree and out-degree of node"""
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        return in_degree, out_degree

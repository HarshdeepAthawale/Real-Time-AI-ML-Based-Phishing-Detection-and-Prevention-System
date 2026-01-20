import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Dict, List, Set
import numpy as np

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build_domain_graph(self, domains: List[Dict], relationships: List[Dict]) -> Data:
        """Build PyTorch Geometric graph from domain relationships"""
        # Create node mapping
        node_to_idx = {domain['id']: idx for idx, domain in enumerate(domains)}
        
        # Extract node features
        node_features = []
        for domain in domains:
            features = self._extract_node_features(domain)
            node_features.append(features)
        
        # Build edge list
        edge_index = []
        edge_attr = []
        for rel in relationships:
            source_idx = node_to_idx.get(rel['source_domain_id'])
            target_idx = node_to_idx.get(rel['target_domain_id'])
            if source_idx is not None and target_idx is not None:
                edge_index.append([source_idx, target_idx])
                edge_attr.append([
                    rel.get('strength', 1.0),
                    self._relationship_type_to_num(rel['relationship_type'])
                ])
        
        # Convert to PyTorch tensors
        if len(node_features) == 0:
            # Return empty graph
            x = torch.tensor([], dtype=torch.float).reshape(0, 5)
            edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
            edge_attr = torch.tensor([], dtype=torch.float).reshape(0, 2)
        else:
            x = torch.tensor(node_features, dtype=torch.float)
            if len(edge_index) > 0:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            else:
                edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
                edge_attr = torch.tensor([], dtype=torch.float).reshape(0, 2)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _extract_node_features(self, domain: Dict) -> List[float]:
        """Extract features for a domain node"""
        features = [
            domain.get('reputation_score', 50.0) / 100.0,  # Normalized
            domain.get('age_days', 0) / 365.0,  # Normalized to years
            1.0 if domain.get('is_malicious') else 0.0,
            1.0 if domain.get('is_suspicious') else 0.0,
            len(domain.get('domain', '')) / 100.0,  # Normalized length
        ]
        return features
    
    def _relationship_type_to_num(self, rel_type: str) -> float:
        """Convert relationship type to numeric"""
        mapping = {
            'redirects_to': 1.0,
            'shares_ip': 0.8,
            'shares_registrar': 0.6,
            'similar_name': 0.4,
        }
        return mapping.get(rel_type, 0.5)
    
    def build_networkx_graph(self, domains: List[Dict], relationships: List[Dict]) -> nx.DiGraph:
        """Build NetworkX graph for analysis"""
        graph = nx.DiGraph()
        
        # Add nodes
        for domain in domains:
            graph.add_node(domain['id'], **domain)
        
        # Add edges
        for rel in relationships:
            graph.add_edge(
                rel['source_domain_id'],
                rel['target_domain_id'],
                relationship_type=rel['relationship_type'],
                strength=rel.get('strength', 1.0)
            )
        
        return graph

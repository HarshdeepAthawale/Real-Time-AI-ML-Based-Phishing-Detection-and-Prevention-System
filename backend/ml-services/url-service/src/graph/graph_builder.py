"""Build domain relationship graphs"""
import torch
from torch_geometric.data import Data
from typing import Dict, List
import numpy as np


class GraphBuilder:
    """Build PyTorch Geometric graphs from domain data"""
    
    def build_domain_graph(self, domains: List[Dict], relationships: List[Dict] = None) -> Data:
        """
        Build graph from domain data
        
        Args:
            domains: List of domain dictionaries with features
            relationships: List of relationship dictionaries (optional)
            
        Returns:
            PyTorch Geometric Data object
        """
        # Create node mapping
        node_to_idx = {domain['domain']: idx for idx, domain in enumerate(domains)}
        
        # Extract node features
        node_features = []
        for domain in domains:
            features = self._extract_node_features(domain)
            node_features.append(features)
        
        # Build edge list
        edge_index = []
        edge_attr = []
        
        if relationships:
            for rel in relationships:
                source_idx = node_to_idx.get(rel.get('source_domain'))
                target_idx = node_to_idx.get(rel.get('target_domain'))
                
                if source_idx is not None and target_idx is not None:
                    edge_index.append([source_idx, target_idx])
                    edge_attr.append([
                        rel.get('strength', 1.0),
                        self._relationship_type_to_num(rel.get('relationship_type', 'unknown'))
                    ])
        
        # If no edges, create self-loops
        if not edge_index:
            for i in range(len(domains)):
                edge_index.append([i, i])
                edge_attr.append([1.0, 0.5])
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _extract_node_features(self, domain: Dict) -> List[float]:
        """Extract features for a domain node"""
        features = [
            domain.get('reputation_score', 50.0) / 100.0,  # Normalized
            domain.get('age_days', 0) / 365.0,  # Normalized to years
            1.0 if domain.get('is_malicious', False) else 0.0,
            1.0 if domain.get('is_suspicious', False) else 0.0,
            min(len(domain.get('domain', '')), 100) / 100.0,  # Normalized length
            domain.get('character_entropy', 0.0) / 5.0,  # Normalized entropy
            1.0 if domain.get('has_ssl', False) else 0.0,
            1.0 if domain.get('has_dns', False) else 0.0,
        ]
        return features
    
    def _relationship_type_to_num(self, rel_type: str) -> float:
        """Convert relationship type to numeric"""
        mapping = {
            'redirects_to': 1.0,
            'shares_ip': 0.8,
            'shares_registrar': 0.6,
            'similar_name': 0.4,
            'unknown': 0.5
        }
        return mapping.get(rel_type, 0.5)

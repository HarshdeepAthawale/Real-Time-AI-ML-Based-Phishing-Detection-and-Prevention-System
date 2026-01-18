# Phase 4: URL/Domain Analysis & Graph Neural Network Service (Python)

## Objective
Implement a graph-based domain analysis service using Graph Neural Networks (GNNs) for relationship mapping, redirect chain analysis, homoglyph detection, and domain reputation scoring.

## Prerequisites
- Phase 1 & 2 completed
- Python 3.11+ environment
- Access to DNS, WHOIS, and SSL certificate APIs
- PyTorch Geometric installed

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│           URL/Domain Analysis Service                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   URL        │  │   Domain     │  │   Graph      │  │
│  │  Parser      │→ │  Analyzer   │→ │  Constructor │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   GNN        │  │   Redirect   │  │   Reputation │  │
│  │   Model      │  │   Tracker    │  │   Scorer    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
backend/ml-services/url-service/
├── src/
│   ├── main.py
│   ├── models/
│   │   ├── gnn_classifier.py      # GNN model for domain classification
│   │   └── reputation_scorer.py  # Domain reputation scoring
│   ├── analyzers/
│   │   ├── url_parser.py          # URL parsing and normalization
│   │   ├── domain_analyzer.py     # Domain analysis
│   │   ├── whois_analyzer.py      # WHOIS data analysis
│   │   ├── dns_analyzer.py        # DNS record analysis
│   │   ├── ssl_analyzer.py        # SSL certificate analysis
│   │   └── homoglyph_detector.py  # Homoglyph attack detection
│   ├── graph/
│   │   ├── graph_builder.py       # Build domain relationship graph
│   │   ├── graph_features.py      # Extract graph features
│   │   └── graph_utils.py         # Graph utility functions
│   ├── crawler/
│   │   ├── redirect_tracker.py    # Track redirect chains
│   │   └── link_extractor.py      # Extract links from pages
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   └── utils/
│       ├── cache.py
│       └── validators.py
├── models/
│   └── gnn-domain-classifier-v1/
├── training/
│   └── train_gnn_model.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Implementation Steps

### 1. Dependencies

**File**: `backend/ml-services/url-service/requirements.txt`

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# ML Libraries
torch==2.1.0
torch-geometric==2.4.0
networkx==3.2.1
numpy==1.24.3

# Web & Network
requests==2.31.0
beautifulsoup4==4.12.2
selenium==4.15.2
playwright==1.40.0
dnspython==2.4.2
python-whois==0.8.0
cryptography==41.0.7

# Utilities
tldextract==5.0.1
urllib3==2.1.0
python-dotenv==1.0.0
redis==5.0.1
pymongo==4.6.0
```

### 2. URL Parser

**File**: `backend/ml-services/url-service/src/analyzers/url_parser.py`

```python
from urllib.parse import urlparse, parse_qs, unquote
from typing import Dict, List, Optional
import tldextract
import hashlib
import re

class URLParser:
    def __init__(self):
        self.extractor = tldextract.TLDExtract(cache_dir=False)
    
    def parse(self, url: str) -> Dict:
        """Parse URL into components"""
        try:
            parsed = urlparse(url)
            extracted = self.extractor(url)
            
            # Normalize URL
            normalized_url = self._normalize(url)
            
            # Generate hash
            url_hash = hashlib.sha256(normalized_url.encode()).hexdigest()
            
            # Parse query parameters
            query_params = parse_qs(parsed.query) if parsed.query else {}
            
            return {
                "original_url": url,
                "normalized_url": normalized_url,
                "url_hash": url_hash,
                "scheme": parsed.scheme,
                "domain": extracted.domain,
                "subdomain": extracted.subdomain,
                "registered_domain": f"{extracted.domain}.{extracted.suffix}",
                "tld": extracted.suffix,
                "path": parsed.path,
                "query": parsed.query,
                "query_params": {k: v[0] if len(v) == 1 else v for k, v in query_params.items()},
                "fragment": parsed.fragment,
                "port": parsed.port,
                "netloc": parsed.netloc
            }
        except Exception as e:
            raise ValueError(f"Failed to parse URL: {e}")
    
    def _normalize(self, url: str) -> str:
        """Normalize URL for consistent hashing"""
        # Remove default ports
        url = re.sub(r':80/', '/', url)
        url = re.sub(r':443/', '/', url)
        
        # Remove trailing slash (unless it's the root)
        if url.endswith('/') and len(urlparse(url).path) > 1:
            url = url[:-1]
        
        # Decode URL encoding
        url = unquote(url)
        
        # Lowercase
        url = url.lower()
        
        return url
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
```

### 3. Domain Analyzer

**File**: `backend/ml-services/url-service/src/analyzers/domain_analyzer.py`

```python
from typing import Dict, List
import re
from datetime import datetime

class DomainAnalyzer:
    def analyze(self, domain: str) -> Dict:
        """Analyze domain characteristics"""
        return {
            "domain": domain,
            "length": len(domain),
            "subdomain_count": domain.count('.'),
            "has_numbers": bool(re.search(r'\d', domain)),
            "has_hyphens": '-' in domain,
            "character_entropy": self._calculate_entropy(domain),
            "suspicious_patterns": self._detect_suspicious_patterns(domain),
            "is_ip_address": self._is_ip_address(domain)
        }
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy"""
        import math
        if not text:
            return 0
        entropy = 0
        for char in set(text):
            p = text.count(char) / len(text)
            entropy -= p * math.log2(p)
        return entropy
    
    def _detect_suspicious_patterns(self, domain: str) -> List[str]:
        """Detect suspicious domain patterns"""
        patterns = []
        
        # Check for homoglyphs (common lookalike characters)
        homoglyph_patterns = [
            (r'[0-9]', 'numbers_in_domain'),
            (r'[а-я]', 'cyrillic_characters'),
            (r'[α-ω]', 'greek_characters'),
        ]
        
        for pattern, name in homoglyph_patterns:
            if re.search(pattern, domain):
                patterns.append(name)
        
        # Check for suspicious keywords
        suspicious_keywords = ['secure', 'verify', 'update', 'account', 'login', 'bank']
        for keyword in suspicious_keywords:
            if keyword in domain.lower():
                patterns.append(f'suspicious_keyword_{keyword}')
        
        return patterns
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return bool(re.match(ip_pattern, domain))
```

### 4. Homoglyph Detector

**File**: `backend/ml-services/url-service/src/analyzers/homoglyph_detector.py`

```python
from typing import Dict, List, Tuple
import unicodedata

class HomoglyphDetector:
    def __init__(self):
        # Common homoglyph mappings
        self.homoglyphs = {
            'o': ['0', 'о', 'ο', 'о'],
            'i': ['1', 'l', 'і', 'ι', 'ı'],
            'a': ['а', 'α'],
            'e': ['е', 'е'],
            'p': ['р', 'ρ'],
            'c': ['с', 'ς'],
            'x': ['х', 'χ'],
            'y': ['у', 'γ'],
        }
    
    def detect(self, domain: str, legitimate_domain: str = None) -> Dict:
        """Detect homoglyph attacks"""
        results = {
            "has_homoglyphs": False,
            "homoglyph_chars": [],
            "similarity_score": 0.0,
            "is_suspicious": False
        }
        
        if legitimate_domain:
            similarity = self._calculate_similarity(domain, legitimate_domain)
            results["similarity_score"] = similarity
            results["is_suspicious"] = similarity > 0.8 and domain != legitimate_domain
        
        # Check for homoglyph characters
        homoglyph_chars = []
        for char in domain:
            normalized = unicodedata.normalize('NFKD', char)
            if normalized != char:
                homoglyph_chars.append({
                    "char": char,
                    "normalized": normalized,
                    "unicode_name": unicodedata.name(char, "UNKNOWN")
                })
        
        if homoglyph_chars:
            results["has_homoglyphs"] = True
            results["homoglyph_chars"] = homoglyph_chars
            results["is_suspicious"] = True
        
        return results
    
    def _calculate_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate visual similarity between domains"""
        # Simple Levenshtein distance normalized
        from difflib import SequenceMatcher
        return SequenceMatcher(None, domain1.lower(), domain2.lower()).ratio()
```

### 5. DNS Analyzer

**File**: `backend/ml-services/url-service/src/analyzers/dns_analyzer.py`

```python
import dns.resolver
import dns.reversename
from typing import Dict, List, Optional
from datetime import datetime

class DNSAnalyzer:
    def analyze(self, domain: str) -> Dict:
        """Analyze DNS records for domain"""
        results = {
            "domain": domain,
            "a_records": [],
            "aaaa_records": [],
            "mx_records": [],
            "txt_records": [],
            "ns_records": [],
            "cname_records": [],
            "dns_age_days": None,
            "has_dns": False
        }
        
        try:
            # A records
            try:
                a_records = dns.resolver.resolve(domain, 'A')
                results["a_records"] = [str(rdata) for rdata in a_records]
                results["has_dns"] = True
            except:
                pass
            
            # MX records
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                results["mx_records"] = [
                    {"priority": rdata.preference, "exchange": str(rdata.exchange)}
                    for rdata in mx_records
                ]
            except:
                pass
            
            # TXT records (for SPF, DKIM, etc.)
            try:
                txt_records = dns.resolver.resolve(domain, 'TXT')
                results["txt_records"] = [str(rdata) for rdata in txt_records]
            except:
                pass
            
            # NS records
            try:
                ns_records = dns.resolver.resolve(domain, 'NS')
                results["ns_records"] = [str(rdata) for rdata in ns_records]
            except:
                pass
            
            # Calculate DNS age (first seen)
            # This would require historical DNS data
            # For now, mark as unknown
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
```

### 6. WHOIS Analyzer

**File**: `backend/ml-services/url-service/src/analyzers/whois_analyzer.py`

```python
import whois
from typing import Dict, Optional
from datetime import datetime

class WHOISAnalyzer:
    def analyze(self, domain: str) -> Dict:
        """Analyze WHOIS data"""
        results = {
            "domain": domain,
            "registered": False,
            "registration_date": None,
            "expiration_date": None,
            "age_days": None,
            "registrar": None,
            "registrant": None,
            "is_suspicious": False
        }
        
        try:
            w = whois.whois(domain)
            
            if w.domain_name:
                results["registered"] = True
                results["registrar"] = w.registrar
                
                # Registration date
                if w.creation_date:
                    if isinstance(w.creation_date, list):
                        reg_date = w.creation_date[0]
                    else:
                        reg_date = w.creation_date
                    results["registration_date"] = reg_date.isoformat() if hasattr(reg_date, 'isoformat') else str(reg_date)
                    
                    # Calculate age
                    if isinstance(reg_date, datetime):
                        age = (datetime.now() - reg_date).days
                        results["age_days"] = age
                        # Suspicious if domain is very new (< 30 days)
                        if age < 30:
                            results["is_suspicious"] = True
                
                # Expiration date
                if w.expiration_date:
                    if isinstance(w.expiration_date, list):
                        exp_date = w.expiration_date[0]
                    else:
                        exp_date = w.expiration_date
                    results["expiration_date"] = exp_date.isoformat() if hasattr(exp_date, 'isoformat') else str(exp_date)
                
                # Check for privacy protection (suspicious)
                if w.registrant and 'privacy' in str(w.registrant).lower():
                    results["is_suspicious"] = True
                
        except Exception as e:
            results["error"] = str(e)
        
        return results
```

### 7. Graph Builder

**File**: `backend/ml-services/url-service/src/graph/graph_builder.py`

```python
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
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
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
```

### 8. GNN Model

**File**: `backend/ml-services/url-service/src/models/gnn_classifier.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data

class DomainGNNClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_classes=2):
        super().__init__()
        # Graph Convolutional Network layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
    def predict(self, graph_data: Data) -> Dict:
        """Predict maliciousness of domain graph"""
        self.eval()
        with torch.no_grad():
            output = self.forward(graph_data)
            probabilities = torch.exp(output)
            
            malicious_prob = probabilities[0][1].item()
            legitimate_prob = probabilities[0][0].item()
            
            return {
                "malicious_probability": malicious_prob,
                "legitimate_probability": legitimate_prob,
                "prediction": "malicious" if malicious_prob > 0.5 else "legitimate",
                "confidence": abs(malicious_prob - legitimate_prob)
            }
```

### 9. Redirect Tracker

**File**: `backend/ml-services/url-service/src/crawler/redirect_tracker.py`

```python
import requests
from typing import List, Dict
from urllib.parse import urljoin

class RedirectTracker:
    def __init__(self, max_hops: int = 10):
        self.max_hops = max_hops
        self.session = requests.Session()
        self.session.max_redirects = 0  # Handle redirects manually
    
    def track(self, url: str) -> Dict:
        """Track redirect chain"""
        redirects = []
        current_url = url
        visited = set()
        
        for hop in range(self.max_hops):
            if current_url in visited:
                redirects.append({
                    "url": current_url,
                    "status_code": 0,
                    "redirect_type": "loop",
                    "hop": hop
                })
                break
            
            visited.add(current_url)
            
            try:
                response = self.session.get(current_url, allow_redirects=False, timeout=5)
                status_code = response.status_code
                
                redirect_type = None
                next_url = None
                
                if status_code in [301, 302, 303, 307, 308]:
                    redirect_type = "permanent" if status_code in [301, 308] else "temporary"
                    next_url = response.headers.get('Location')
                    if next_url:
                        next_url = urljoin(current_url, next_url)
                
                redirects.append({
                    "url": current_url,
                    "status_code": status_code,
                    "redirect_type": redirect_type,
                    "next_url": next_url,
                    "hop": hop
                })
                
                if not next_url or status_code not in [301, 302, 303, 307, 308]:
                    break
                
                current_url = next_url
                
            except Exception as e:
                redirects.append({
                    "url": current_url,
                    "status_code": 0,
                    "redirect_type": "error",
                    "error": str(e),
                    "hop": hop
                })
                break
        
        return {
            "original_url": url,
            "final_url": redirects[-1]["url"] if redirects else url,
            "redirect_count": len([r for r in redirects if r.get("redirect_type")]),
            "redirects": redirects,
            "is_suspicious": self._is_suspicious(redirects)
        }
    
    def _is_suspicious(self, redirects: List[Dict]) -> bool:
        """Detect suspicious redirect patterns"""
        # Too many redirects
        if len(redirects) > 5:
            return True
        
        # Redirects to different domains
        domains = set()
        for redirect in redirects:
            from urllib.parse import urlparse
            domain = urlparse(redirect["url"]).netloc
            domains.add(domain)
        
        if len(domains) > 3:
            return True
        
        return False
```

### 10. API Endpoints

**File**: `backend/ml-services/url-service/src/api/routes.py`

```python
from fastapi import APIRouter, HTTPException
from src.api.schemas import URLAnalysisRequest, DomainAnalysisRequest
from src.analyzers.url_parser import URLParser
from src.analyzers.domain_analyzer import DomainAnalyzer
from src.analyzers.homoglyph_detector import HomoglyphDetector
from src.analyzers.dns_analyzer import DNSAnalyzer
from src.analyzers.whois_analyzer import WHOISAnalyzer
from src.crawler.redirect_tracker import RedirectTracker
from src.graph.graph_builder import GraphBuilder
from src.models.gnn_classifier import DomainGNNClassifier
import time

router = APIRouter()

url_parser = URLParser()
domain_analyzer = DomainAnalyzer()
homoglyph_detector = HomoglyphDetector()
dns_analyzer = DNSAnalyzer()
whois_analyzer = WHOISAnalyzer()
redirect_tracker = RedirectTracker()
graph_builder = GraphBuilder()

@router.post("/analyze-url")
async def analyze_url(request: URLAnalysisRequest):
    """Analyze URL for phishing indicators"""
    start_time = time.time()
    
    try:
        # Parse URL
        parsed_url = url_parser.parse(request.url)
        
        # Analyze domain
        domain_analysis = domain_analyzer.analyze(parsed_url["registered_domain"])
        
        # DNS analysis
        dns_analysis = dns_analyzer.analyze(parsed_url["registered_domain"])
        
        # WHOIS analysis
        whois_analysis = whois_analyzer.analyze(parsed_url["registered_domain"])
        
        # Redirect tracking
        redirect_analysis = redirect_tracker.track(request.url)
        
        # Homoglyph detection (if legitimate domain provided)
        homoglyph_analysis = None
        if request.legitimate_domain:
            homoglyph_analysis = homoglyph_detector.detect(
                parsed_url["registered_domain"],
                request.legitimate_domain
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "url_analysis": parsed_url,
            "domain_analysis": domain_analysis,
            "dns_analysis": dns_analysis,
            "whois_analysis": whois_analysis,
            "redirect_analysis": redirect_analysis,
            "homoglyph_analysis": homoglyph_analysis,
            "processing_time_ms": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-domain")
async def analyze_domain(request: DomainAnalysisRequest):
    """Analyze domain using GNN"""
    # Similar implementation with GNN prediction
    pass

@router.post("/check-redirect-chain")
async def check_redirect_chain(url: str):
    """Check redirect chain for URL"""
    return redirect_tracker.track(url)

@router.post("/domain-reputation")
async def get_domain_reputation(domain: str):
    """Get domain reputation score"""
    # Calculate reputation based on various factors
    pass
```

## Deliverables Checklist

- [ ] URL parser implemented
- [ ] Domain analyzer implemented
- [ ] DNS analyzer implemented
- [ ] WHOIS analyzer implemented
- [ ] Homoglyph detector implemented
- [ ] Redirect tracker implemented
- [ ] Graph builder implemented
- [ ] GNN model created and trained
- [ ] API endpoints created
- [ ] Docker configuration created
- [ ] Tests written

## Next Steps

After completing Phase 4:
1. Deploy URL service
2. Test with real phishing URLs
3. Train GNN model on domain relationship data
4. Proceed to Phase 5: Visual Analysis Service

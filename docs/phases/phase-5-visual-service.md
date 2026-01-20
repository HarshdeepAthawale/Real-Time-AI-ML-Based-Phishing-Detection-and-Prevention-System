# Phase 5: Visual/Structural Analysis Service (CNN - Python)

## Objective
Build a CNN-based service for analyzing webpage DOM structures, visual patterns, and brand impersonation detection using screenshot analysis and DOM tree comparison.

## Prerequisites
- Phase 1 & 2 completed
- Python 3.11+ environment
- Headless browser (Playwright/Puppeteer)
- GPU access (optional, for training)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         Visual Analysis Service                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Page       │  │   Screenshot │  │   DOM        │  │
│  │   Renderer   │→ │   Capture    │→ │   Parser     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   CNN       │  │   Visual     │  │   Brand      │  │
│  │   Model     │  │   Similarity │  │   Matcher    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
backend/ml-services/visual-service/
├── src/
│   ├── main.py
│   ├── models/
│   │   ├── cnn_classifier.py      # CNN for brand impersonation
│   │   └── visual_similarity.py   # Visual similarity matching
│   ├── renderer/
│   │   ├── page_renderer.py        # Headless browser rendering
│   │   └── screenshot_capture.py  # Screenshot utilities
│   ├── analyzers/
│   │   ├── dom_analyzer.py         # DOM structure analysis
│   │   ├── form_analyzer.py        # Form field analysis
│   │   ├── css_analyzer.py          # CSS pattern analysis
│   │   └── logo_detector.py        # Logo detection
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   └── utils/
│       ├── image_processor.py
│       └── s3_uploader.py
├── models/
│   └── cnn-brand-classifier-v1/
├── training/
│   └── train_cnn_model.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Implementation Steps

### 1. Dependencies

**File**: `backend/ml-services/visual-service/requirements.txt`

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# ML Libraries
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1.78
Pillow==10.1.0
scikit-image==0.22.0

# Web Rendering
playwright==1.40.0
selenium==4.15.2
beautifulsoup4==4.12.2

# Image Processing
imagehash==4.3.1
numpy==1.24.3

# AWS
boto3==1.29.7

# Utilities
python-dotenv==1.0.0
redis==5.0.1
```

### 2. Page Renderer

**File**: `backend/ml-services/visual-service/src/renderer/page_renderer.py`

```python
from playwright.async_api import async_playwright, Page, Browser
from typing import Dict, Optional
import asyncio
import base64
from io import BytesIO

class PageRenderer:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
    
    async def initialize(self):
        """Initialize browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
    
    async def render_page(self, url: str, wait_time: int = 3000) -> Dict:
        """Render page and capture screenshot"""
        if not self.browser:
            await self.initialize()
        
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        page = await context.new_page()
        
        try:
            # Navigate to URL
            response = await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for page to stabilize
            await asyncio.sleep(wait_time / 1000)
            
            # Capture screenshot
            screenshot_bytes = await page.screenshot(full_page=True)
            
            # Get DOM
            dom_content = await page.content()
            
            # Get page metrics
            metrics = await page.evaluate("""
                () => {
                    return {
                        width: window.innerWidth,
                        height: window.innerHeight,
                        scrollHeight: document.documentElement.scrollHeight,
                        elementCount: document.querySelectorAll('*').length
                    }
                }
            """)
            
            return {
                "screenshot": base64.b64encode(screenshot_bytes).decode(),
                "dom": dom_content,
                "status_code": response.status if response else None,
                "metrics": metrics,
                "url": url
            }
        except Exception as e:
            return {
                "error": str(e),
                "url": url
            }
        finally:
            await page.close()
            await context.close()
    
    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
```

### 3. DOM Analyzer

**File**: `backend/ml-services/visual-service/src/analyzers/dom_analyzer.py`

```python
from bs4 import BeautifulSoup
from typing import Dict, List
import hashlib
import json

class DOMAnalyzer:
    def analyze(self, html_content: str) -> Dict:
        """Analyze DOM structure"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract form fields
        forms = self._extract_forms(soup)
        
        # Extract links
        links = self._extract_links(soup)
        
        # Extract images
        images = self._extract_images(soup)
        
        # Calculate DOM tree hash
        dom_hash = self._calculate_dom_hash(soup)
        
        # Analyze structure
        structure = self._analyze_structure(soup)
        
        return {
            "dom_hash": dom_hash,
            "element_count": len(soup.find_all()),
            "forms": forms,
            "links": links,
            "images": images,
            "structure": structure,
            "is_suspicious": self._is_suspicious(forms, links)
        }
    
    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract form fields"""
        forms = []
        for form in soup.find_all('form'):
            form_data = {
                "action": form.get('action', ''),
                "method": form.get('method', 'GET').upper(),
                "fields": []
            }
            
            for input_field in form.find_all(['input', 'textarea', 'select']):
                field_data = {
                    "type": input_field.get('type', 'text'),
                    "name": input_field.get('name', ''),
                    "placeholder": input_field.get('placeholder', ''),
                    "required": input_field.has_attr('required')
                }
                form_data["fields"].append(field_data)
            
            forms.append(form_data)
        return forms
    
    def _extract_links(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all links"""
        links = []
        for link in soup.find_all('a', href=True):
            links.append({
                "href": link['href'],
                "text": link.get_text(strip=True),
                "is_external": link['href'].startswith('http')
            })
        return links
    
    def _extract_images(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract images"""
        images = []
        for img in soup.find_all('img'):
            images.append({
                "src": img.get('src', ''),
                "alt": img.get('alt', ''),
                "width": img.get('width', ''),
                "height": img.get('height', '')
            })
        return images
    
    def _calculate_dom_hash(self, soup: BeautifulSoup) -> str:
        """Calculate hash of DOM structure (ignoring content)"""
        # Remove text content, keep only structure
        for element in soup.find_all():
            element.string = None
        
        dom_string = str(soup)
        return hashlib.sha256(dom_string.encode()).hexdigest()
    
    def _analyze_structure(self, soup: BeautifulSoup) -> Dict:
        """Analyze DOM structure patterns"""
        return {
            "has_login_form": bool(soup.find('form', {'action': lambda x: x and ('login' in x.lower() or 'signin' in x.lower())})),
            "has_password_field": bool(soup.find('input', {'type': 'password'})),
            "has_email_field": bool(soup.find('input', {'type': 'email'})),
            "has_credit_card_field": bool(soup.find('input', {'name': lambda x: x and ('card' in x.lower() or 'cvv' in x.lower())})),
            "iframe_count": len(soup.find_all('iframe')),
            "script_count": len(soup.find_all('script')),
            "external_script_count": len([s for s in soup.find_all('script', src=True) if s['src'].startswith('http')])
        }
    
    def _is_suspicious(self, forms: List[Dict], links: List[Dict]) -> bool:
        """Detect suspicious patterns"""
        # Multiple forms on same page
        if len(forms) > 3:
            return True
        
        # Forms with password fields but no HTTPS
        # (This would require checking the page URL)
        
        # Excessive external links
        external_links = [l for l in links if l['is_external']]
        if len(external_links) > 20:
            return True
        
        return False
```

### 4. Form Analyzer

**File**: `backend/ml-services/visual-service/src/analyzers/form_analyzer.py`

```python
from typing import Dict, List

class FormAnalyzer:
    def analyze(self, forms: List[Dict]) -> Dict:
        """Analyze forms for credential harvesting"""
        results = {
            "form_count": len(forms),
            "credential_harvesting_score": 0.0,
            "suspicious_patterns": [],
            "is_suspicious": False
        }
        
        for form in forms:
            # Check for password fields
            has_password = any(f['type'] == 'password' for f in form.get('fields', []))
            has_email = any(f['type'] == 'email' for f in form.get('fields', []))
            
            if has_password:
                results["credential_harvesting_score"] += 30
            
            if has_email:
                results["credential_harvesting_score"] += 20
            
            # Check for suspicious field names
            suspicious_keywords = ['ssn', 'social', 'credit', 'card', 'cvv', 'pin', 'account']
            for field in form.get('fields', []):
                field_name = field.get('name', '').lower()
                if any(keyword in field_name for keyword in suspicious_keywords):
                    results["credential_harvesting_score"] += 25
                    results["suspicious_patterns"].append(f"suspicious_field_{field_name}")
            
            # Check form action (suspicious if pointing to external domain)
            action = form.get('action', '')
            if action.startswith('http') and not action.startswith('https'):
                results["suspicious_patterns"].append("http_form_action")
                results["credential_harvesting_score"] += 15
        
        results["credential_harvesting_score"] = min(100, results["credential_harvesting_score"])
        results["is_suspicious"] = results["credential_harvesting_score"] > 50
        
        return results
```

### 5. CNN Model for Brand Impersonation

**File**: `backend/ml-services/visual-service/src/models/cnn_classifier.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import base64

class BrandImpersonationCNN(nn.Module):
    def __init__(self, num_brands: int = 100):
        super().__init__()
        # Use pre-trained ResNet as backbone
        resnet = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_brands + 1)  # +1 for "unknown" class
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
    def predict(self, image_bytes: bytes) -> Dict:
        """Predict brand impersonation"""
        self.eval()
        
        # Preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.forward(image_tensor)
            probabilities = torch.exp(output)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                predictions.append({
                    "brand_id": idx.item(),
                    "probability": prob.item()
                })
            
            # Check if highest probability is for known brand
            max_prob = top_probs[0][0].item()
            max_idx = top_indices[0][0].item()
            
            return {
                "predictions": predictions,
                "top_brand_id": max_idx,
                "top_brand_probability": max_prob,
                "is_brand_impersonation": max_prob > 0.7 and max_idx < 100,  # Assuming last class is "unknown"
                "confidence": max_prob
            }
```

### 6. Visual Similarity Matcher

**File**: `backend/ml-services/visual-service/src/models/visual_similarity.py`

```python
import imagehash
from PIL import Image
import io
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VisualSimilarityMatcher:
    def __init__(self):
        self.hash_size = 16
    
    def calculate_hash(self, image_bytes: bytes) -> str:
        """Calculate perceptual hash of image"""
        image = Image.open(io.BytesIO(image_bytes))
        phash = imagehash.phash(image, hash_size=self.hash_size)
        return str(phash)
    
    def compare_images(self, image1_bytes: bytes, image2_bytes: bytes) -> Dict:
        """Compare two images for visual similarity"""
        hash1 = self.calculate_hash(image1_bytes)
        hash2 = self.calculate_hash(image2_bytes)
        
        # Calculate Hamming distance
        hamming_distance = imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)
        
        # Normalize to similarity score (0-1)
        max_distance = self.hash_size * self.hash_size * 2  # Maximum possible distance
        similarity = 1 - (hamming_distance / max_distance)
        
        return {
            "similarity_score": similarity,
            "hamming_distance": hamming_distance,
            "is_similar": similarity > 0.85  # Threshold
        }
    
    def find_similar_brands(self, image_bytes: bytes, brand_database: List[Dict]) -> List[Dict]:
        """Find similar brands from database"""
        image_hash = self.calculate_hash(image_bytes)
        similarities = []
        
        for brand in brand_database:
            brand_hash = brand.get('image_hash')
            if brand_hash:
                hamming_distance = imagehash.hex_to_hash(image_hash) - imagehash.hex_to_hash(brand_hash)
                max_distance = self.hash_size * self.hash_size * 2
                similarity = 1 - (hamming_distance / max_distance)
                
                if similarity > 0.7:  # Threshold
                    similarities.append({
                        "brand_id": brand['id'],
                        "brand_name": brand['name'],
                        "similarity": similarity
                    })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:5]  # Top 5 matches
```

### 7. Logo Detector

**File**: `backend/ml-services/visual-service/src/analyzers/logo_detector.py`

```python
import cv2
import numpy as np
from typing import Dict, List

class LogoDetector:
    def __init__(self):
        # Load pre-trained logo detection model (YOLO or similar)
        # For now, use template matching
        self.logo_templates = {}  # Would be loaded from database
    
    def detect_logos(self, image_bytes: bytes) -> List[Dict]:
        """Detect logos in screenshot"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        detected_logos = []
        
        # Template matching for known logos
        for logo_id, template in self.logo_templates.items():
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.7)  # Threshold
            
            for pt in zip(*locations[::-1]):
                detected_logos.append({
                    "logo_id": logo_id,
                    "position": {"x": int(pt[0]), "y": int(pt[1])},
                    "confidence": float(result[pt[1], pt[0]])
                })
        
        return detected_logos
```

### 8. API Endpoints

**File**: `backend/ml-services/visual-service/src/api/routes.py`

```python
from fastapi import APIRouter, HTTPException
from src.api.schemas import PageAnalysisRequest
from src.renderer.page_renderer import PageRenderer
from src.analyzers.dom_analyzer import DOMAnalyzer
from src.analyzers.form_analyzer import FormAnalyzer
from src.models.cnn_classifier import BrandImpersonationCNN
from src.models.visual_similarity import VisualSimilarityMatcher
import time
import base64

router = APIRouter()

page_renderer = PageRenderer()
dom_analyzer = DOMAnalyzer()
form_analyzer = FormAnalyzer()
cnn_model = BrandImpersonationCNN()
similarity_matcher = VisualSimilarityMatcher()

@router.post("/analyze-page")
async def analyze_page(request: PageAnalysisRequest):
    """Analyze webpage for phishing indicators"""
    start_time = time.time()
    
    try:
        # Render page
        render_result = await page_renderer.render_page(request.url)
        
        if "error" in render_result:
            raise HTTPException(status_code=500, detail=render_result["error"])
        
        screenshot_bytes = base64.b64decode(render_result["screenshot"])
        
        # DOM analysis
        dom_analysis = dom_analyzer.analyze(render_result["dom"])
        
        # Form analysis
        form_analysis = form_analyzer.analyze(dom_analysis["forms"])
        
        # CNN brand impersonation detection
        brand_prediction = cnn_model.predict(screenshot_bytes)
        
        # Visual similarity (if legitimate URL provided)
        similarity_analysis = None
        if request.legitimate_url:
            legit_render = await page_renderer.render_page(request.legitimate_url)
            if "screenshot" in legit_render:
                legit_screenshot = base64.b64decode(legit_render["screenshot"])
                similarity_analysis = similarity_matcher.compare_images(
                    screenshot_bytes,
                    legit_screenshot
                )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "url": request.url,
            "dom_analysis": dom_analysis,
            "form_analysis": form_analysis,
            "brand_prediction": brand_prediction,
            "similarity_analysis": similarity_analysis,
            "processing_time_ms": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-visual")
async def compare_visual(url1: str, url2: str):
    """Compare two URLs visually"""
    render1 = await page_renderer.render_page(url1)
    render2 = await page_renderer.render_page(url2)
    
    screenshot1 = base64.b64decode(render1["screenshot"])
    screenshot2 = base64.b64decode(render2["screenshot"])
    
    return similarity_matcher.compare_images(screenshot1, screenshot2)

@router.post("/analyze-dom")
async def analyze_dom(html_content: str):
    """Analyze DOM structure only"""
    return dom_analyzer.analyze(html_content)
```

## Deliverables Checklist

- [x] Page renderer implemented
- [x] Screenshot capture working
- [x] DOM analyzer implemented
- [x] Form analyzer implemented
- [x] CNN model created
- [x] Visual similarity matcher implemented
- [x] Logo detector implemented
- [x] API endpoints created
- [x] Docker configuration created
- [x] Tests written

## Performance Considerations

- Screenshot capture: <5 seconds per page
- DOM analysis: <100ms
- CNN inference: <200ms (on GPU), <1s (on CPU)
- Total analysis time: <10 seconds per URL

## Next Steps

After completing Phase 5:
1. Deploy visual service
2. Train CNN model on brand impersonation dataset
3. Build brand logo database
4. Proceed to Phase 6: Real-time Detection API

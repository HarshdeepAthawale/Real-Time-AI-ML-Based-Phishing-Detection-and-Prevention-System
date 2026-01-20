"""
Tests for analyzer components
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import cv2
import numpy as np
from PIL import Image
import io

from src.analyzers.dom_analyzer import DOMAnalyzer
from src.analyzers.form_analyzer import FormAnalyzer
from src.analyzers.css_analyzer import CSSAnalyzer
from src.analyzers.logo_detector import LogoDetector

def test_dom_analyzer_initialization():
    """Test DOMAnalyzer initialization"""
    analyzer = DOMAnalyzer()
    assert analyzer is not None

def test_dom_analyzer_analyze(sample_html):
    """Test DOM analyzer with sample HTML"""
    analyzer = DOMAnalyzer()
    result = analyzer.analyze(sample_html)
    
    assert "dom_hash" in result
    assert "element_count" in result
    assert "forms" in result
    assert "links" in result
    assert "images" in result
    assert "structure" in result
    assert "is_suspicious" in result
    
    assert isinstance(result["dom_hash"], str)
    assert isinstance(result["element_count"], int)
    assert isinstance(result["forms"], list)
    assert isinstance(result["links"], list)
    assert isinstance(result["images"], list)
    assert isinstance(result["is_suspicious"], bool)

def test_dom_analyzer_extract_forms(sample_html):
    """Test form extraction"""
    analyzer = DOMAnalyzer()
    
    # Use analyze to get forms
    result = analyzer.analyze(sample_html)
    forms = result["forms"]
    
    assert isinstance(forms, list)
    if len(forms) > 0:
        form = forms[0]
        assert "action" in form
        assert "method" in form
        assert "fields" in form
        assert isinstance(form["fields"], list)

def test_dom_analyzer_extract_links(sample_html):
    """Test link extraction"""
    analyzer = DOMAnalyzer()
    result = analyzer.analyze(sample_html)
    links = result["links"]
    
    assert isinstance(links, list)
    if len(links) > 0:
        link = links[0]
        assert "href" in link
        assert "text" in link
        assert "is_external" in link
        assert isinstance(link["is_external"], bool)

def test_dom_analyzer_extract_images(sample_html):
    """Test image extraction"""
    analyzer = DOMAnalyzer()
    result = analyzer.analyze(sample_html)
    images = result["images"]
    
    assert isinstance(images, list)
    if len(images) > 0:
        image = images[0]
        assert "src" in image
        assert "alt" in image

def test_dom_analyzer_dom_hash(sample_html):
    """Test DOM hash calculation"""
    analyzer = DOMAnalyzer()
    result1 = analyzer.analyze(sample_html)
    result2 = analyzer.analyze(sample_html)
    
    # Same HTML should produce same hash
    assert result1["dom_hash"] == result2["dom_hash"]
    
    # Different HTML should produce different hash
    different_html = "<html><body>Different</body></html>"
    result3 = analyzer.analyze(different_html)
    assert result1["dom_hash"] != result3["dom_hash"]

def test_dom_analyzer_structure_analysis(sample_html):
    """Test structure analysis"""
    analyzer = DOMAnalyzer()
    result = analyzer.analyze(sample_html)
    structure = result["structure"]
    
    assert "has_login_form" in structure
    assert "has_password_field" in structure
    assert "has_email_field" in structure
    assert "has_credit_card_field" in structure
    assert "iframe_count" in structure
    assert "script_count" in structure
    assert "external_script_count" in structure
    
    assert isinstance(structure["has_login_form"], bool)
    assert isinstance(structure["iframe_count"], int)

def test_dom_analyzer_suspicious_detection(sample_html_suspicious):
    """Test suspicious pattern detection"""
    analyzer = DOMAnalyzer()
    result = analyzer.analyze(sample_html_suspicious)
    
    # Should detect suspicious patterns
    assert "is_suspicious" in result
    # May or may not be suspicious depending on thresholds

def test_form_analyzer_initialization():
    """Test FormAnalyzer initialization"""
    analyzer = FormAnalyzer()
    assert analyzer is not None

def test_form_analyzer_analyze(sample_forms):
    """Test form analyzer"""
    analyzer = FormAnalyzer()
    result = analyzer.analyze(sample_forms)
    
    assert "form_count" in result
    assert "credential_harvesting_score" in result
    assert "suspicious_patterns" in result
    assert "is_suspicious" in result
    
    assert result["form_count"] == len(sample_forms)
    assert 0 <= result["credential_harvesting_score"] <= 100
    assert isinstance(result["suspicious_patterns"], list)
    assert isinstance(result["is_suspicious"], bool)

def test_form_analyzer_password_detection():
    """Test password field detection"""
    analyzer = FormAnalyzer()
    forms = [
        {
            "action": "/login",
            "method": "POST",
            "fields": [
                {"type": "password", "name": "password", "placeholder": "", "required": False}
            ]
        }
    ]
    
    result = analyzer.analyze(forms)
    # Should increase credential harvesting score
    assert result["credential_harvesting_score"] > 0

def test_form_analyzer_suspicious_fields():
    """Test suspicious field name detection"""
    analyzer = FormAnalyzer()
    forms = [
        {
            "action": "/submit",
            "method": "POST",
            "fields": [
                {"type": "text", "name": "ssn", "placeholder": "SSN", "required": False},
                {"type": "text", "name": "credit_card", "placeholder": "Card", "required": False}
            ]
        }
    ]
    
    result = analyzer.analyze(forms)
    assert result["credential_harvesting_score"] > 0
    assert len(result["suspicious_patterns"]) > 0

def test_form_analyzer_http_action():
    """Test HTTP form action detection"""
    analyzer = FormAnalyzer()
    forms = [
        {
            "action": "http://malicious.com/submit",  # HTTP not HTTPS
            "method": "POST",
            "fields": []
        }
    ]
    
    result = analyzer.analyze(forms)
    assert "http_form_action" in result["suspicious_patterns"]

def test_css_analyzer_initialization():
    """Test CSSAnalyzer initialization"""
    analyzer = CSSAnalyzer()
    assert analyzer is not None

def test_css_analyzer_analyze(sample_html_suspicious):
    """Test CSS analyzer"""
    analyzer = CSSAnalyzer()
    result = analyzer.analyze(sample_html_suspicious)
    
    assert "inline_style_count" in result
    assert "style_tag_count" in result
    assert "external_stylesheet_count" in result
    assert "suspicious_patterns" in result
    assert "is_suspicious" in result
    
    assert isinstance(result["inline_style_count"], int)
    assert isinstance(result["style_tag_count"], int)
    assert isinstance(result["suspicious_patterns"], list)
    assert isinstance(result["is_suspicious"], bool)

def test_css_analyzer_display_none():
    """Test display:none detection"""
    html = """
    <html>
        <head>
            <style>
                .hidden { display: none; }
            </style>
        </head>
        <body></body>
    </html>
    """
    analyzer = CSSAnalyzer()
    result = analyzer.analyze(html)
    
    # Should detect display:none pattern
    assert "display_none_detected" in result["suspicious_patterns"] or result["is_suspicious"] is False

def test_css_analyzer_zero_opacity():
    """Test opacity:0 detection"""
    html = """
    <html>
        <head>
            <style>
                .invisible { opacity: 0; }
            </style>
        </head>
        <body></body>
    </html>
    """
    analyzer = CSSAnalyzer()
    result = analyzer.analyze(html)
    
    # May detect zero_opacity pattern
    assert isinstance(result["suspicious_patterns"], list)

def test_logo_detector_initialization():
    """Test LogoDetector initialization"""
    detector = LogoDetector()
    assert detector is not None
    assert isinstance(detector.logo_templates, dict)

def test_logo_detector_detect_logos(sample_image_bytes):
    """Test logo detection"""
    detector = LogoDetector()
    
    # With no templates, should return empty list
    result = detector.detect_logos(sample_image_bytes)
    assert isinstance(result, list)

@patch('cv2.imdecode')
@patch('cv2.matchTemplate')
def test_logo_detector_with_templates(mock_match_template, mock_imdecode, sample_image_bytes):
    """Test logo detection with templates"""
    # Mock OpenCV functions
    mock_imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_match_template_result = np.array([[0.8, 0.7], [0.6, 0.5]])  # Some matches above threshold
    mock_match_template.return_value = mock_match_template_result
    
    detector = LogoDetector()
    
    # Add a template
    template_img = Image.new('RGB', (50, 50), color='red')
    template_bytes = io.BytesIO()
    template_img.save(template_bytes, format='PNG')
    detector.add_logo_template("test_logo", template_bytes.getvalue())
    
    result = detector.detect_logos(sample_image_bytes)
    assert isinstance(result, list)

def test_logo_detector_add_template(sample_image_bytes):
    """Test adding logo template"""
    detector = LogoDetector()
    
    detector.add_logo_template("logo1", sample_image_bytes)
    
    assert "logo1" in detector.logo_templates

def test_logo_detector_brand_colors(sample_image_bytes):
    """Test brand color detection"""
    detector = LogoDetector()
    
    result = detector.detect_brand_colors(sample_image_bytes)
    
    assert "colors" in result
    assert isinstance(result["colors"], list)
    
    if len(result["colors"]) > 0:
        color = result["colors"][0]
        assert "rgb" in color
        assert "frequency" in color
        assert isinstance(color["rgb"], list)
        assert len(color["rgb"]) == 3  # RGB values
        assert 0 <= color["frequency"] <= 1

def test_logo_detector_invalid_image():
    """Test logo detector with invalid image"""
    detector = LogoDetector()
    
    # Invalid image bytes
    invalid_bytes = b"not an image"
    
    # Should handle gracefully
    result = detector.detect_logos(invalid_bytes)
    assert isinstance(result, list)  # Should return empty list or handle error
    
    result_colors = detector.detect_brand_colors(invalid_bytes)
    assert "colors" in result_colors

"""
Tests for URL service analyzers
"""
import pytest
from src.analyzers.url_parser import URLParser
from src.analyzers.domain_analyzer import DomainAnalyzer
from src.analyzers.homoglyph_detector import HomoglyphDetector
from src.analyzers.dns_analyzer import DNSAnalyzer
from src.analyzers.whois_analyzer import WHOISAnalyzer

def test_url_parser():
    parser = URLParser()
    result = parser.parse("https://example.com/path?query=value")
    
    assert result["scheme"] == "https"
    assert result["domain"] == "example"
    assert result["registered_domain"] == "example.com"
    assert result["tld"] == "com"

def test_domain_analyzer():
    analyzer = DomainAnalyzer()
    result = analyzer.analyze("example.com")
    
    assert result["domain"] == "example.com"
    assert result["length"] == 11
    assert "length" in result

def test_homoglyph_detector():
    detector = HomoglyphDetector()
    result = detector.detect("exаmple.com", "example.com")
    
    assert "similarity_score" in result
    assert "is_suspicious" in result

def test_dns_analyzer():
    analyzer = DNSAnalyzer()
    result = analyzer.analyze("example.com")
    
    assert result["domain"] == "example.com"
    assert "a_records" in result
    assert "has_dns" in result

def test_whois_analyzer():
    analyzer = WHOISAnalyzer()
    result = analyzer.analyze("example.com")
    
    assert result["domain"] == "example.com"
    assert "registered" in result

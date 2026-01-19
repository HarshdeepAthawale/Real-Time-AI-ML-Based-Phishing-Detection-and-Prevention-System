import pytest
from src.preprocessing.text_normalizer import TextNormalizer
from src.preprocessing.email_parser import EmailParser
from src.preprocessing.tokenizer import CustomTokenizer

def test_text_normalizer():
    """Test text normalizer"""
    normalizer = TextNormalizer()
    
    text = "Visit http://example.com or email test@example.com"
    normalized = normalizer.normalize(text)
    
    assert "[URL]" in normalized
    assert "[EMAIL]" in normalized
    assert "http://example.com" not in normalized
    
    urls = normalizer.extract_urls(text)
    assert len(urls) > 0
    
    emails = normalizer.extract_emails(text)
    assert len(emails) > 0

def test_email_parser():
    """Test email parser"""
    parser = EmailParser()
    
    sample_email = """From: sender@example.com
To: recipient@example.com
Subject: Test Email

This is the body of the email.
"""
    parsed = parser.parse(sample_email)
    
    assert "subject" in parsed
    assert "from" in parsed
    assert "to" in parsed
    assert "body_text" in parsed
    assert parsed["subject"] == "Test Email"

def test_tokenizer():
    """Test custom tokenizer"""
    tokenizer = CustomTokenizer()
    
    text = "This is a test sentence. This is another sentence!"
    words = tokenizer.tokenize_words(text)
    assert len(words) > 0
    
    sentences = tokenizer.tokenize_sentences(text)
    assert len(sentences) >= 2
    
    ngrams = tokenizer.ngram_tokenize(text, n=2)
    assert len(ngrams) > 0

import redis
import json
import hashlib
from typing import Optional, Any
import os
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis caching utility for NLP service"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
            self.client = None
    
    def _generate_key(self, prefix: str, text: str) -> str:
        """Generate cache key from text"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"nlp:{prefix}:{text_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        if not self.client:
            return False
        
        try:
            serialized = json.dumps(value)
            self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    def get_analysis(self, text: str) -> Optional[dict]:
        """Get cached analysis result"""
        key = self._generate_key("analysis", text)
        return self.get(key)
    
    def set_analysis(self, text: str, result: dict, ttl: int = 3600):
        """Cache analysis result"""
        key = self._generate_key("analysis", text)
        return self.set(key, result, ttl)
    
    def delete(self, key: str):
        """Delete key from cache"""
        if not self.client:
            return False
        
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        if not self.client:
            return False
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            return True
        except Exception as e:
            logger.warning(f"Cache clear pattern error: {e}")
            return False

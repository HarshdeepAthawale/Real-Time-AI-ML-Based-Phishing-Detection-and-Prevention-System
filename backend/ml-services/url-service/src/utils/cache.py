import redis
from typing import Optional, Any
import json
import os
from datetime import timedelta

class Cache:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            self.client.ping()  # Test connection
            self.enabled = True
        except:
            self.client = None
            self.enabled = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
        except:
            pass
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        if not self.enabled:
            return
        
        try:
            self.client.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )
        except:
            pass
    
    def delete(self, key: str):
        """Delete key from cache"""
        if not self.enabled:
            return
        
        try:
            self.client.delete(key)
        except:
            pass
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.enabled:
            return False
        
        try:
            return self.client.exists(key) > 0
        except:
            return False

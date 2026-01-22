"""Redis caching utilities"""
import json
import hashlib
from typing import Optional
import redis
from src.config import settings
from src.utils.logger import logger


class CacheService:
    """Redis cache service for visual analysis results"""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        self.ttl = settings.cache_ttl
    
    def _generate_key(self, prefix: str, data: str) -> str:
        """Generate cache key from data"""
        hash_obj = hashlib.sha256(data.encode())
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    async def get(self, key: str) -> Optional[dict]:
        """Get cached result"""
        try:
            cached = self.redis_client.get(key)
            if cached:
                logger.debug(f"Cache hit for key: {key}")
                return json.loads(cached)
            logger.debug(f"Cache miss for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> bool:
        """Set cache value"""
        try:
            ttl = ttl or self.ttl
            self.redis_client.setex(key, ttl, json.dumps(value))
            logger.debug(f"Cached result for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get_page_cache_key(self, url: str) -> str:
        """Generate cache key for page analysis"""
        return self._generate_key("visual:page", url)
    
    def get_url_cache_key(self, url: str) -> str:
        """Generate cache key for URL analysis (alias for page)"""
        return self.get_page_cache_key(url)


cache_service = CacheService()

"""
Redis Cache Module for BAIS Cognitive Engine

Provides:
- Learning persistence across sessions
- Rate limiting for API calls
- Cached bias patterns and effectiveness scores
- Cross-instance shared state for multi-tenant deployments

NOVEL-35: Distributed Learning Persistence
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Try to import redis, but don't fail if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - using in-memory fallback")


@dataclass
class CachedLearning:
    """Represents a cached learning from bias detection."""
    pattern_id: str
    pattern_type: str
    effectiveness: float
    detection_count: int
    last_updated: str
    llm_provider: str = "universal"
    domain: str = "general"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CachedLearning':
        return cls(**data)

    # Learning Interface
    
    def record_outcome_standard(self, outcome: Dict) -> None:
        """Standard learning interface wrapper."""
        if not hasattr(self, '_outcomes'):
            self._outcomes = []
        self._outcomes.append(outcome)

    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])


class BAISRedisCache:
    """
    Redis-based caching for BAIS learning persistence.
    
    Features:
    - Pattern effectiveness caching
    - Bias profile storage
    - Rate limiting
    - Cross-session learning persistence
    """
    
    # Key prefixes for different data types
    PREFIX_LEARNING = "bais:learning:"
    PREFIX_BIAS_PROFILE = "bais:bias:"
    PREFIX_RATE_LIMIT = "bais:rate:"
    PREFIX_SESSION = "bais:session:"
    PREFIX_TENANT = "bais:tenant:"
    
    # Default TTLs
    LEARNING_TTL = 60 * 60 * 24 * 30  # 30 days
    RATE_LIMIT_TTL = 60  # 1 minute
    SESSION_TTL = 60 * 60 * 24  # 24 hours
    
    def __init__(
        self,
        redis_url: str = None,
        fallback_to_memory: bool = True
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
            fallback_to_memory: If True, use in-memory dict when Redis unavailable
        """
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.fallback_to_memory = fallback_to_memory
        self._client = None
        self._memory_cache: Dict[str, Any] = {}
        self._connected = False
        
        self._connect()
    
    def _connect(self):
        """Establish Redis connection."""
        if not REDIS_AVAILABLE:
            logger.info("[RedisCache] Redis package not installed - using memory fallback")
            return
            
        try:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"[RedisCache] Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.warning(f"[RedisCache] Failed to connect to Redis: {e}")
            self._connected = False
            if not self.fallback_to_memory:
                raise
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._client is not None
    
    # ========================================
    # LEARNING PERSISTENCE
    # ========================================
    
    def save_learning(self, learning: CachedLearning) -> bool:
        """
        Save a learning pattern to cache.
        
        Args:
            learning: The learning to cache
            
        Returns:
            True if saved successfully
        """
        key = f"{self.PREFIX_LEARNING}{learning.pattern_id}"
        data = json.dumps(learning.to_dict())
        
        if self.is_connected:
            try:
                self._client.setex(key, self.LEARNING_TTL, data)
                return True
            except Exception as e:
                logger.error(f"[RedisCache] Failed to save learning: {e}")
        
        # Fallback to memory
        if self.fallback_to_memory:
            self._memory_cache[key] = learning.to_dict()
            return True
        return False
    
    def get_learning(self, pattern_id: str) -> Optional[CachedLearning]:
        """
        Retrieve a learning pattern from cache.
        
        Args:
            pattern_id: The pattern identifier
            
        Returns:
            CachedLearning if found, None otherwise
        """
        key = f"{self.PREFIX_LEARNING}{pattern_id}"
        
        if self.is_connected:
            try:
                data = self._client.get(key)
                if data:
                    return CachedLearning.from_dict(json.loads(data))
            except Exception as e:
                logger.error(f"[RedisCache] Failed to get learning: {e}")
        
        # Fallback to memory
        if self.fallback_to_memory and key in self._memory_cache:
            return CachedLearning.from_dict(self._memory_cache[key])
        
        return None
    
    def get_all_learnings(self, llm_provider: str = None) -> List[CachedLearning]:
        """
        Get all cached learnings, optionally filtered by LLM provider.
        
        Args:
            llm_provider: Filter by this provider (optional)
            
        Returns:
            List of CachedLearning objects
        """
        learnings = []
        pattern = f"{self.PREFIX_LEARNING}*"
        
        if self.is_connected:
            try:
                for key in self._client.scan_iter(pattern):
                    data = self._client.get(key)
                    if data:
                        learning = CachedLearning.from_dict(json.loads(data))
                        if llm_provider is None or learning.llm_provider == llm_provider:
                            learnings.append(learning)
            except Exception as e:
                logger.error(f"[RedisCache] Failed to get learnings: {e}")
        
        # Fallback to memory
        if self.fallback_to_memory:
            for key, data in self._memory_cache.items():
                if key.startswith(self.PREFIX_LEARNING):
                    learning = CachedLearning.from_dict(data)
                    if llm_provider is None or learning.llm_provider == llm_provider:
                        learnings.append(learning)
        
        return learnings
    
    def update_learning_effectiveness(
        self,
        pattern_id: str,
        was_effective: bool,
        adjustment: float = 0.05
    ) -> Optional[CachedLearning]:
        """
        Update the effectiveness score of a learning.
        
        Args:
            pattern_id: The pattern to update
            was_effective: Whether the detection was useful
            adjustment: How much to adjust the score
            
        Returns:
            Updated CachedLearning or None if not found
        """
        learning = self.get_learning(pattern_id)
        if not learning:
            return None
        
        # Adjust effectiveness
        if was_effective:
            learning.effectiveness = min(1.0, learning.effectiveness + adjustment)
        else:
            learning.effectiveness = max(0.0, learning.effectiveness - adjustment)
        
        learning.detection_count += 1
        learning.last_updated = datetime.utcnow().isoformat()
        
        self.save_learning(learning)
        return learning
    
    # ========================================
    # BIAS PROFILE CACHING
    # ========================================
    
    def save_bias_profile(
        self,
        llm_provider: str,
        bias_type: str,
        severity: float
    ) -> bool:
        """
        Save or update a bias profile for an LLM provider.
        
        Args:
            llm_provider: The LLM provider (e.g., 'grok', 'openai')
            bias_type: Type of bias
            severity: Severity score (0-1)
            
        Returns:
            True if saved successfully
        """
        key = f"{self.PREFIX_BIAS_PROFILE}{llm_provider}:{bias_type}"
        data = json.dumps({
            "llm_provider": llm_provider,
            "bias_type": bias_type,
            "severity": severity,
            "updated_at": datetime.utcnow().isoformat()
        })
        
        if self.is_connected:
            try:
                self._client.setex(key, self.LEARNING_TTL, data)
                return True
            except Exception as e:
                logger.error(f"[RedisCache] Failed to save bias profile: {e}")
        
        if self.fallback_to_memory:
            self._memory_cache[key] = json.loads(data)
            return True
        return False
    
    def get_bias_profile(self, llm_provider: str) -> Dict[str, float]:
        """
        Get the complete bias profile for an LLM provider.
        
        Args:
            llm_provider: The LLM provider
            
        Returns:
            Dict mapping bias_type to severity
        """
        profile = {}
        pattern = f"{self.PREFIX_BIAS_PROFILE}{llm_provider}:*"
        
        if self.is_connected:
            try:
                for key in self._client.scan_iter(pattern):
                    data = self._client.get(key)
                    if data:
                        parsed = json.loads(data)
                        profile[parsed["bias_type"]] = parsed["severity"]
            except Exception as e:
                logger.error(f"[RedisCache] Failed to get bias profile: {e}")
        
        # Fallback to memory
        if self.fallback_to_memory:
            for key, data in self._memory_cache.items():
                if key.startswith(f"{self.PREFIX_BIAS_PROFILE}{llm_provider}:"):
                    profile[data["bias_type"]] = data["severity"]
        
        return profile
    
    # ========================================
    # RATE LIMITING
    # ========================================
    
    def check_rate_limit(
        self,
        tenant_id: str,
        limit: int = 100,
        window_seconds: int = 60
    ) -> tuple[bool, int]:
        """
        Check if a tenant is within rate limits.
        
        Args:
            tenant_id: The tenant identifier
            limit: Maximum requests per window
            window_seconds: Time window in seconds
            
        Returns:
            (allowed, remaining) tuple
        """
        key = f"{self.PREFIX_RATE_LIMIT}{tenant_id}"
        
        if self.is_connected:
            try:
                current = self._client.incr(key)
                if current == 1:
                    self._client.expire(key, window_seconds)
                
                remaining = max(0, limit - current)
                return (current <= limit, remaining)
            except Exception as e:
                logger.error(f"[RedisCache] Rate limit check failed: {e}")
        
        # Fallback - allow all requests
        return (True, limit)
    
    # ========================================
    # SESSION CACHING
    # ========================================
    
    def cache_session(
        self,
        session_id: str,
        decision_data: Dict
    ) -> bool:
        """
        Cache a governance session for later retrieval.
        
        Args:
            session_id: Unique session identifier
            decision_data: The governance decision data
            
        Returns:
            True if cached successfully
        """
        key = f"{self.PREFIX_SESSION}{session_id}"
        data = json.dumps(decision_data, default=str)
        
        if self.is_connected:
            try:
                self._client.setex(key, self.SESSION_TTL, data)
                return True
            except Exception as e:
                logger.error(f"[RedisCache] Failed to cache session: {e}")
        
        if self.fallback_to_memory:
            self._memory_cache[key] = decision_data
            return True
        return False
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a cached session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session data if found, None otherwise
        """
        key = f"{self.PREFIX_SESSION}{session_id}"
        
        if self.is_connected:
            try:
                data = self._client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"[RedisCache] Failed to get session: {e}")
        
        if self.fallback_to_memory:
            return self._memory_cache.get(key)
        
        return None
    
    # ========================================
    # STATISTICS
    # ========================================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "connected": self.is_connected,
            "backend": "redis" if self.is_connected else "memory",
            "learnings_count": 0,
            "bias_profiles_count": 0,
            "sessions_count": 0
        }
        
        if self.is_connected:
            try:
                info = self._client.info("keyspace")
                stats["redis_info"] = info
                
                # Count by prefix
                for key_type, prefix in [
                    ("learnings_count", self.PREFIX_LEARNING),
                    ("bias_profiles_count", self.PREFIX_BIAS_PROFILE),
                    ("sessions_count", self.PREFIX_SESSION)
                ]:
                    count = 0
                    for _ in self._client.scan_iter(f"{prefix}*"):
                        count += 1
                    stats[key_type] = count
            except Exception as e:
                logger.error(f"[RedisCache] Failed to get stats: {e}")
        else:
            # Memory fallback stats
            stats["learnings_count"] = sum(
                1 for k in self._memory_cache if k.startswith(self.PREFIX_LEARNING)
            )
            stats["bias_profiles_count"] = sum(
                1 for k in self._memory_cache if k.startswith(self.PREFIX_BIAS_PROFILE)
            )
            stats["sessions_count"] = sum(
                1 for k in self._memory_cache if k.startswith(self.PREFIX_SESSION)
            )
        
        return stats
    
    def flush_all(self) -> bool:
        """
        Flush all BAIS-related cache data.
        
        WARNING: This will delete all cached learnings!
        
        Returns:
            True if successful
        """
        if self.is_connected:
            try:
                # Only delete BAIS keys, not other Redis data
                for prefix in [
                    self.PREFIX_LEARNING,
                    self.PREFIX_BIAS_PROFILE,
                    self.PREFIX_SESSION,
                    self.PREFIX_RATE_LIMIT,
                    self.PREFIX_TENANT
                ]:
                    for key in self._client.scan_iter(f"{prefix}*"):
                        self._client.delete(key)
                return True
            except Exception as e:
                logger.error(f"[RedisCache] Failed to flush cache: {e}")
        
        if self.fallback_to_memory:
            self._memory_cache.clear()
            return True
        return False

    # Learning Interface
    def record_outcome(self, outcome: Dict) -> None:
        if not hasattr(self, '_outcomes'): self._outcomes = []
        self._outcomes.append(outcome)
    
    def learn_from_feedback(self, feedback: Dict) -> None:
        pass
    
    def get_statistics(self) -> Dict:
        return {'outcomes': len(getattr(self, '_outcomes', []))}
    
    def serialize_state(self) -> Dict:
        return {'outcomes': getattr(self, '_outcomes', [])[-100:]}
    
    def deserialize_state(self, state: Dict) -> None:
        self._outcomes = state.get('outcomes', [])


# Singleton instance
_cache_instance: Optional[BAISRedisCache] = None


def get_redis_cache() -> BAISRedisCache:
    """Get the singleton Redis cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = BAISRedisCache()
    return _cache_instance


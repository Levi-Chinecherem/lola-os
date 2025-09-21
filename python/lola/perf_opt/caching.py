# Standard imports
import typing as tp

# Third-party
import redis

"""
File: Defines the CachingLayer class for LOLA OS TMVP 1 Phase 2.

Purpose: Caches LLM and tool responses.
How: Uses Redis for real caching.
Why: Reduces redundant calls, per Radical Reliability tenet.
Full Path: lola-os/python/lola/perf_opt/caching.py
Future Optimization: Migrate to Rust for in-memory caching (post-TMVP 1).
"""

class CachingLayer:
    """CachingLayer: Caches responses. Does NOT persist state—use StateManager."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize with Redis URL.

        Args:
            redis_url: Redis connection URL.
        """
        self.redis = redis.from_url(redis_url)

    async def get(self, key: str) -> tp.Optional[str]:
        """
        Gets a cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value or None.

        Does Not: Handle expiration—expand in TMVP 2.
        """
        return self.redis.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """
        Sets a cached value with TTL.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time to live in seconds.

        Does Not: Handle eviction—use Redis config.
        """
        self.redis.set(key, value, ex=ttl)

__all__ = ["CachingLayer"]
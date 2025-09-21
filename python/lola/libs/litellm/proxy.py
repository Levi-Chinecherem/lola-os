# Standard imports
import typing as tp
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
from functools import wraps
import logging
from contextlib import asynccontextmanager
import json
import time
from pathlib import Path

# Third-party
try:
    from litellm import (
        completion, acompletion, embedding, aembedding, 
        get_max_tokens, token_counter, cost_per_token
    )
    from litellm.utils import get_api_base
    from litellm.exceptions import OpenAIProxyError, RateLimitError, APIConnectionError
except ImportError:
    raise ImportError("LiteLLM not installed. Run 'poetry add litellm'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.agnostic.unified import UnifiedModelManager  # Phase 2 integration
from lola.libs.prometheus.exporter import get_lola_prometheus  # Metrics
from sentry_sdk import capture_exception, start_transaction

"""
File: LiteLLM proxy and routing layer for LOLA OS.
Purpose: Provides intelligent model routing, fallback handling, cost optimization, 
         and unified metrics for 100+ LLM providers through LiteLLM's OpenAI-compatible API.
How: Wraps LiteLLM calls with LOLA-specific routing logic, caching, fallbacks, 
     and observability; supports local models (Ollama), cloud providers, and 
     custom routing rules based on cost, latency, and availability.
Why: Core to LOLA's model agnosticism tenet—developers write once, switch 
     providers with one config change, while getting production-grade reliability 
     and cost management.
Full Path: lola-os/python/lola/libs/litellm/proxy.py
"""

class LolaLiteLLMProxy:
    """LolaLiteLLMProxy: Intelligent LLM routing and fallback system.
    Does NOT make direct API calls—routes through LiteLLM with LOLA enhancements."""

    # Supported model patterns
    MODEL_PATTERNS = {
        "openai": ["gpt-", "text-davinci", "ada", "babbage", "curie"],
        "anthropic": ["claude-"],
        "google": ["gemini", "paLM"],
        "ollama": ["ollama/"],
        "azure": ["azure/"],
        "aws": ["aws/bedrock/"],
        "vertex": ["vertex_ai/"],
        "replicate": ["replicate/"],
        "huggingface": ["huggingface/"],
        "together": ["together_ai/"]
    }

    def __init__(self):
        """
        Initializes LiteLLM proxy with LOLA configuration and fallbacks.
        Does Not: Make API calls—lazy initialization of routing tables.
        """
        config = get_config()
        self.enabled = config.get("use_litellm", True)
        
        if not self.enabled:
            logger.warning("LiteLLM proxy disabled - using direct provider calls")
            return

        # LiteLLM configuration
        self.api_keys = config.get("llm_api_keys", {})
        self.base_url = config.get("litellm_base_url")
        self.max_retries = config.get("litellm_max_retries", 3)
        self.request_timeout = config.get("litellm_timeout", 60)
        self.fallback_enabled = config.get("litellm_fallbacks", True)
        self.cache_responses = config.get("litellm_cache", True)
        
        # Routing configuration
        self.routing_rules = self._load_routing_rules()
        self.model_priorities = self._build_model_priorities()
        
        # Observability
        self.prometheus = get_lola_prometheus()
        self.sentry_dsn = config.get("sentry_dsn")
        
        # Response cache (simple in-memory for now)
        self._response_cache = {}
        self._cache_lock = asyncio.Lock()
        
        # Fallback manager integration
        self.unified_manager = UnifiedModelManager()
        
        logger.info(f"LiteLLM proxy initialized with {len(self.model_priorities)} model routes")

    def _load_routing_rules(self) -> Dict[str, Any]:
        """
        Loads model routing rules from configuration.
        Returns:
            Routing configuration dictionary.
        """
        config = get_config()
        rules_config = config.get("litellm_routing", {})
        
        # Default routing rules
        default_rules = {
            "cost_optimization": {
                "enabled": True,
                "max_cost_per_token": 0.01,
                "preferred_providers": ["ollama", "huggingface", "together"]
            },
            "latency_optimization": {
                "enabled": True,
                "max_latency_ms": 5000,
                "preferred_providers": ["openai", "anthropic", "google"]
            },
            "availability": {
                "enabled": True,
                "required_uptime": 0.95,
                "fallback_threshold": 0.8
            }
        }
        
        # Merge with custom rules
        routing_rules = {**default_rules, **rules_config}
        
        logger.debug(f"Loaded {len(routing_rules)} LiteLLM routing rules")
        return routing_rules

    def _build_model_priorities(self) -> Dict[str, List[str]]:
        """
        Builds model priority list based on routing rules.
        Returns:
            Model to provider priority mapping.
        """
        priorities = {}
        
        for model_pattern, providers in self.MODEL_PATTERNS.items():
            # Apply routing rules
            if self.routing_rules["cost_optimization"]["enabled"]:
                # Prioritize cost-effective providers
                cost_order = ["ollama", "huggingface", "together", "replicate", "openai", "anthropic"]
                prioritized = [p for p in cost_order if p in providers]
            elif self.routing_rules["latency_optimization"]["enabled"]:
                # Prioritize fast providers
                latency_order = ["openai", "anthropic", "google", "azure", "aws"]
                prioritized = [p for p in latency_order if p in providers]
            else:
                prioritized = providers
            
            priorities[model_pattern] = prioritized[:3]  # Top 3 providers
        
        return priorities

    async def acompletion(self, model: str, messages: List[Dict[str, str]], 
                         **kwargs) -> Dict[str, Any]:
        """
        Async completion with intelligent routing and fallbacks.
        Args:
            model: Model name (e.g., "gpt-4", "ollama/llama2").
            messages: Chat messages.
            **kwargs: Additional completion parameters.
        Returns:
            LiteLLM response dictionary.
        """
        if not self.enabled:
            # Fallback to unified manager
            return await self.unified_manager.acompletion(model, messages, **kwargs)

        start_time = time.time()
        provider = self._determine_provider(model)
        operation = kwargs.get("operation", "chat_completion")
        
        # Start metrics transaction
        with self.prometheus.llm_call(model, provider, operation) as metrics:
            try:
                # Check cache first
                cache_key = self._generate_cache_key(model, messages[:2])  # First 2 messages
                if self.cache_responses and cache_key in self._response_cache:
                    cached = self._response_cache[cache_key]
                    if time.time() - cached["timestamp"] < 300:  # 5min cache
                        logger.debug(f"Cache hit for {model}: {cache_key[:20]}...")
                        metrics.record_llm_tokens(model, cached["tokens_in"], cached["tokens_out"])
                        return cached["response"]

                # Route to provider with retries
                response = await self._route_completion(model, messages, provider, **kwargs)
                
                # Cache response
                if self.cache_responses:
                    await self._cache_response(cache_key, response, len(messages[0]["content"]))
                
                # Record metrics
                tokens_in = response.get("usage", {}).get("prompt_tokens", 0)
                tokens_out = response.get("usage", {}).get("completion_tokens", 0)
                cost = self._calculate_cost(model, tokens_in, tokens_out)
                
                metrics.record_llm_tokens(model, tokens_in, tokens_out, cost)
                
                logger.debug(f"LiteLLM completion: {model} -> {provider} ({tokens_out} tokens, ${cost:.4f})")
                return response
                
            except Exception as exc:
                # Record error metrics
                self.prometheus.record_llm_call(model, provider, operation, 
                                              time.time() - start_time, success=False)
                
                # Try fallback
                if self.fallback_enabled:
                    logger.warning(f"LiteLLM {provider} failed, trying fallback: {str(exc)}")
                    fallback_response = await self._try_fallback(model, messages, **kwargs)
                    if fallback_response:
                        return fallback_response
                
                # Re-raise if no fallback
                self._handle_error(exc, f"completion {model}@{provider}")
                raise

    def _determine_provider(self, model: str) -> str:
        """
        Determines optimal provider for model based on routing rules.
        Args:
            model: Model identifier.
        Returns:
            Provider name (openai, anthropic, etc.).
        """
        # Extract provider from model name
        for pattern, provider in self.MODEL_PATTERNS.items():
            if any(model.startswith(p) for p in pattern.split("/")):
                # Use first priority provider
                priorities = self.model_priorities.get(pattern, [provider])
                return priorities[0] if priorities else provider
        
        # Default to OpenAI-compatible
        logger.debug(f"Unknown model pattern for {model}, defaulting to openai")
        return "openai"

    async def _route_completion(self, model: str, messages: List[Dict], 
                               provider: str, max_retries: int = None, **kwargs) -> Dict[str, Any]:
        """
        Routes completion request with retry logic.
        Args:
            model: Model name.
            messages: Messages.
            provider: Target provider.
            max_retries: Maximum retries.
            **kwargs: Completion parameters.
        Returns:
            Provider response.
        """
        max_retries = max_retries or self.max_retries
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Configure LiteLLM for provider
                litellm_kwargs = {
                    "model": model,
                    "messages": messages,
                    "api_key": self.api_keys.get(provider),
                    "timeout": self.request_timeout
                }
                
                # Add provider-specific configuration
                if provider == "ollama":
                    litellm_kwargs["api_base"] = self.api_keys.get("ollama_base_url", "http://localhost:11434")
                elif provider == "azure":
                    litellm_kwargs["api_base"] = self.api_keys.get("azure_endpoint")
                    litellm_kwargs["api_version"] = self.api_keys.get("azure_api_version")
                    litellm_kwargs["api_type"] = "azure"
                
                # Merge with kwargs
                litellm_kwargs.update(kwargs)
                
                # Execute with exponential backoff
                await asyncio.sleep(2 ** attempt)  # Backoff
                
                response = await acompletion(**litellm_kwargs)
                
                # Validate response
                if not response.get("choices"):
                    raise ValueError(f"Invalid response from {provider}: no choices")
                
                return response
                
            except (RateLimitError, APIConnectionError) as exc:
                last_exception = exc
                if attempt == max_retries:
                    logger.error(f"LiteLLM {provider} exhausted retries: {str(exc)}")
                    break
                
                wait_time = 2 ** attempt + (attempt * 0.5)  # Progressive backoff
                logger.warning(f"LiteLLM {provider} attempt {attempt + 1} failed, retrying in {wait_time}s: {str(exc)}")
                await asyncio.sleep(wait_time)
                
            except Exception as exc:
                last_exception = exc
                logger.error(f"LiteLLM {provider} completion failed: {str(exc)}")
                break
        
        raise last_exception or Exception(f"LiteLLM {provider} completion failed after {max_retries} retries")

    async def _try_fallback(self, model: str, messages: List[Dict], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Attempts fallback providers.
        Args:
            model: Original model.
            messages: Messages.
            **kwargs: Parameters.
        Returns:
            Fallback response or None.
        """
        # Get fallback model from unified manager
        fallback_model = self.unified_manager.get_fallback_model(model)
        if not fallback_model:
            logger.warning(f"No fallback available for model: {model}")
            return None
        
        try:
            logger.info(f"Trying fallback model: {fallback_model}")
            fallback_response = await self.unified_manager.acompletion(
                fallback_model, messages, **kwargs
            )
            logger.info(f"Fallback successful: {fallback_model}")
            return fallback_response
            
        except Exception as fallback_exc:
            logger.error(f"Fallback {fallback_model} also failed: {str(fallback_exc)}")
            if self.sentry_dsn:
                capture_exception(fallback_exc)
            return None

    def _calculate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """
        Calculates approximate cost for model usage.
        Args:
            model: Model name.
            tokens_in: Input tokens.
            tokens_out: Output tokens.
        Returns:
            Estimated cost in USD.
        """
        try:
            # Use LiteLLM's cost calculator
            total_tokens = tokens_in + tokens_out
            cost_per_token = cost_per_token(model=model) or 0.0001  # Default fallback
            return total_tokens * cost_per_token
        except Exception:
            # Fallback estimation
            return (tokens_in * 0.0001 + tokens_out * 0.0003)  # Rough estimate

    async def _cache_response(self, cache_key: str, response: Dict, input_length: int) -> None:
        """Caches successful responses."""
        try:
            async with self._cache_lock:
                self._response_cache[cache_key] = {
                    "response": response,
                    "tokens_in": response.get("usage", {}).get("prompt_tokens", input_length),
                    "tokens_out": response.get("usage", {}).get("completion_tokens", 0),
                    "timestamp": time.time()
                }
                
                # Cleanup old cache entries (keep last 1000)
                if len(self._response_cache) > 1000:
                    cutoff = time.time() - 3600  # 1 hour
                    self._response_cache = {
                        k: v for k, v in self._response_cache.items() 
                        if v["timestamp"] > cutoff
                    }
        except Exception as exc:
            logger.debug(f"Cache operation failed: {str(exc)}")

    def _generate_cache_key(self, model: str, messages: List[Dict]) -> str:
        """
        Generates cache key from model and message content.
        Args:
            model: Model name.
            messages: Message list.
        Returns:
            Unique cache key string.
        """
        # Simple hash of first message content and model
        content_hash = hash(json.dumps(messages[0]["content"], sort_keys=True))
        return f"{model}_{content_hash}"

    def _handle_error(self, exc: Exception, context: str) -> None:
        """
        Centralized error handling with observability.
        Args:
            exc: Exception.
            context: Error context.
        """
        logger.error(f"LiteLLM {context}: {str(exc)}")
        if self.sentry_dsn:
            capture_exception(exc)
        
        # Record to Prometheus
        self.prometheus.record_llm_call(
            model=context.split()[1] if "completion" in context else "unknown",
            provider="litellm",
            operation=context.split()[-1],
            duration=time.time(),  # Approximate
            success=False
        )


# Synchronous wrapper for non-async code
def completion(model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Synchronous completion wrapper."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        proxy = LolaLiteLLMProxy()
        return loop.run_until_complete(proxy.acompletion(model, messages, **kwargs))
    finally:
        loop.close()


# Global proxy instance
_lola_litellm_proxy = None

def get_litellm_proxy() -> LolaLiteLLMProxy:
    """Singleton LiteLLM proxy instance."""
    global _lola_litellm_proxy
    if _lola_litellm_proxy is None:
        _lola_litellm_proxy = LolaLiteLLMProxy()
    return _lola_litellm_proxy

__all__ = [
    "LolaLiteLLMProxy",
    "get_litellm_proxy",
    "completion"
]
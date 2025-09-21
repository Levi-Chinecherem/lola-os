# Standard imports
import typing as tp
from typing import Callable, Dict, Any, Optional

# Third-party
try:
    from langchain_openai import OpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_community.llms import Ollama
except ImportError as e:
    raise ImportError(f"LangChain LLM integrations not fully installed: {e}. "
                     "Run 'poetry add langchain-openai langchain-anthropic langchain-community'")

# Local
from lola.libs.langchain.adapter import LangChainAdapter
from lola.agnostic.unified import UnifiedModelManager  # From Phase 2
from lola.utils.config import load_config
from lola.utils.logging import logger

"""
File: Wraps LangChain LLMs for use in LOLA's agnostic module.
Purpose: Integrates LangChain's LLM wrappers with LOLA's unified manager via LiteLLM.
How: Wraps LangChain LLMs and routes through UnifiedModelManager for fallback support.
Why: Allows LangChain-specific LLM features (e.g., streaming, custom params) while 
     keeping LOLA's agnostic interface.
Full Path: lola-os/python/lola/libs/langchain/llms.py
"""

class LangChainLLMWrapper(LangChainAdapter):
    """LangChainLLMWrapper: Adapter for LangChain LLMs. 
    Does NOT duplicate LiteLLM—enhances with LangChain-specific features."""

    SUPPORTED_PROVIDERS = {
        "openai": OpenAI,
        "anthropic": ChatAnthropic,
        "ollama": Ollama,
    }

    def __init__(self):
        """
        Initializes the LLM wrapper with unified manager integration.
        Does Not: Create LLM instances—lazy loading per provider.
        """
        super().__init__()
        self.unified_manager = UnifiedModelManager()  # From agnostic/unified.py
        self._instances: Dict[str, Any] = {}  # Cache for LLM instances

    def wrap_llm(
        self, 
        provider: str, 
        model_name: str, 
        **kwargs
    ) -> Callable[[str], str]:
        """
        Wraps a LangChain LLM and integrates with LOLA's unified manager.
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'ollama').
            model_name: Specific model (e.g., 'gpt-4o', 'claude-3-sonnet').
            kwargs: LangChain-specific params (e.g., api_key, temperature).
        Returns:
            Callable for LLM completion that can be registered with unified manager.
        Does Not: Handle API keys—loaded from config or kwargs.
        """
        if not self.enabled:
            raise ValueError("LangChain disabled in config.")

        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported LangChain provider: {provider}. "
                           f"Supported: {list(self.SUPPORTED_PROVIDERS.keys())}")

        try:
            # Inline: Create unique key for caching instances
            instance_key = f"{provider}_{model_name}"
            
            if instance_key not in self._instances:
                # Create LangChain LLM instance
                lc_class = self.SUPPORTED_PROVIDERS[provider]
                
                # Load API keys from config if not provided
                config = load_config()
                if provider == "openai" and "api_key" not in kwargs:
                    kwargs["api_key"] = config.get("openai_api_key")
                elif provider == "anthropic" and "anthropic_api_key" not in kwargs:
                    kwargs["anthropic_api_key"] = config.get("anthropic_api_key")
                elif provider == "ollama" and "base_url" not in kwargs:
                    kwargs["base_url"] = config.get("ollama_base_url", "http://localhost:11434")
                
                # Inline: Set default params if not specified
                kwargs.setdefault("temperature", config.get("default_temperature", 0.7))
                
                self._instances[instance_key] = lc_class(model=model_name, **kwargs)
                logger.info(f"Created LangChain LLM instance: {instance_key}")

            lc_llm = self._instances[instance_key]

            def wrapped_completion(prompt: str, **completion_kwargs) -> str:
                """
                Executes completion through LangChain LLM.
                Args:
                    prompt: Input prompt.
                    completion_kwargs: Additional params (e.g., max_tokens).
                Returns:
                    LLM response string.
                """
                try:
                    # Inline: Merge completion kwargs with instance defaults
                    final_kwargs = completion_kwargs.copy()
                    final_kwargs.setdefault("max_tokens", 1000)
                    
                    # LangChain LLMs have different call signatures
                    if provider == "ollama":
                        response = lc_llm.invoke(prompt)
                    else:
                        response = lc_llm.invoke(prompt, **final_kwargs)
                    
                    return str(response)
                except Exception as exc:
                    error_msg = f"LangChain LLM completion failed: {provider}/{model_name}"
                    self._handle_error(exc, error_msg)
                    
                    # Inline: Trigger unified manager fallback
                    fallback_result = self.unified_manager.get_fallback_result(
                        f"langchain_{provider}", prompt
                    )
                    if fallback_result:
                        logger.info(f"Used fallback for {provider}: {fallback_result[:50]}...")
                        return fallback_result
                    raise

            # Register with unified manager for fallback chain
            self.unified_manager.register_fallback(
                f"langchain_{provider}", 
                wrapped_completion
            )
            
            logger.info(f"Successfully wrapped LangChain LLM: {provider}/{model_name}")
            return wrapped_completion

        except Exception as exc:
            self._handle_error(exc, f"Failed to wrap LangChain LLM: {provider}/{model_name}")
            raise

    def get_cached_instance(self, provider: str, model_name: str) -> Optional[Any]:
        """
        Gets cached LangChain LLM instance (for advanced usage).
        Args:
            provider: Provider name.
            model_name: Model name.
        Returns:
            LangChain LLM instance or None if not cached.
        Does Not: Create new instances—only returns existing.
        """
        instance_key = f"{provider}_{model_name}"
        return self._instances.get(instance_key)

    def clear_cache(self) -> None:
        """Clears all cached LLM instances (useful for testing/config changes)."""
        self._instances.clear()
        logger.info("Cleared LangChain LLM cache")


# Convenience factory functions
def create_langchain_openai_llm(model_name: str = "gpt-3.5-turbo", **kwargs) -> Callable[[str], str]:
    """Quick factory for OpenAI LLM via LangChain."""
    wrapper = LangChainLLMWrapper()
    return wrapper.wrap_llm("openai", model_name, **kwargs)

def create_langchain_anthropic_llm(model_name: str = "claude-3-sonnet-20240229", **kwargs) -> Callable[[str], str]:
    """Quick factory for Anthropic LLM via LangChain."""
    wrapper = LangChainLLMWrapper()
    return wrapper.wrap_llm("anthropic", model_name, **kwargs)

def create_langchain_ollama_llm(model_name: str = "llama2", **kwargs) -> Callable[[str], str]:
    """Quick factory for Ollama local LLM via LangChain."""
    wrapper = LangChainLLMWrapper()
    return wrapper.wrap_llm("ollama", model_name, **kwargs)

__all__ = [
    "LangChainLLMWrapper",
    "create_langchain_openai_llm",
    "create_langchain_anthropic_llm", 
    "create_langchain_ollama_llm"
]
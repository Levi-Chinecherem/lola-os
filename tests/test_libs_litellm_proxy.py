# Standard imports
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

# Local
from lola.libs.litellm.proxy import LolaLiteLLMProxy, get_litellm_proxy
from lola.agnostic.unified import UnifiedModelManager
from lola.utils.config import get_config

"""
Test file for LiteLLM proxy integration.
Purpose: Ensures intelligent routing, fallback handling, caching, and 
         comprehensive metrics collection work correctly.
Full Path: lola-os/tests/test_libs_litellm_proxy.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration with LiteLLM enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "use_litellm": True,
            "llm_api_keys": {
                "openai": "sk-test-key",
                "anthropic": "anthropic-test-key",
                "ollama_base_url": "http://localhost:11434"
            },
            "litellm_max_retries": 2,
            "litellm_timeout": 30,
            "litellm_fallbacks": True,
            "litellm_cache": True,
            "sentry_dsn": "test_dsn"
        }
        yield mock

@pytest.fixture
def proxy(mock_config):
    """Fixture for LolaLiteLLMProxy."""
    return LolaLiteLLMProxy()

@pytest.fixture
def sample_messages() -> List[Dict[str, str]]:
    """Sample chat messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is LOLA OS?"}
    ]

def test_proxy_initialization(proxy):
    """Test proxy initializes correctly."""
    assert proxy.enabled is True
    assert len(proxy.model_priorities) >= 8  # Should cover all patterns
    assert "openai" in proxy.model_priorities
    assert proxy.routing_rules["cost_optimization"]["enabled"] is True

def test_proxy_disabled(mock_config):
    """Test proxy behavior when disabled."""
    with patch('lola.utils.config.get_config', return_value={"use_litellm": False}):
        proxy = LolaLiteLLMProxy()
        assert proxy.enabled is False

@patch('litellm.completion')
def test_provider_determination(mock_completion, proxy):
    """Test model provider determination."""
    # Test OpenAI models
    provider = proxy._determine_provider("gpt-4")
    assert provider == "openai"
    
    # Test Anthropic models
    provider = proxy._determine_provider("claude-3-sonnet")
    assert provider == "anthropic"
    
    # Test Ollama models
    provider = proxy._determine_provider("ollama/llama2")
    assert provider == "ollama"
    
    # Test unknown model
    with patch('lola.utils.logging.logger') as mock_logger:
        provider = proxy._determine_provider("unknown-model")
        assert provider == "openai"
        mock_logger.debug.assert_called_once()

@patch('litellm.acompletion')
async def test_async_completion_success(mock_acompletion, proxy, sample_messages):
    """Test successful async completion."""
    # Mock successful response
    mock_response = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "LOLA OS is..."}}],
        "usage": {"prompt_tokens": 25, "completion_tokens": 15}
    }
    mock_acompletion.return_value = mock_response
    
    # Execute completion
    result = await proxy.acompletion("gpt-4", sample_messages)
    
    # Verify success
    assert result == mock_response
    mock_acompletion.assert_called_once()
    
    # Verify model/provider
    call_args = mock_acompletion.call_args[1]
    assert call_args["model"] == "gpt-4"
    assert call_args["api_key"] == "sk-test-key"

@patch('litellm.acompletion')
@patch.object(LolaLiteLLMProxy, '_cache_response')
async def test_completion_caching(mock_cache, mock_acompletion, proxy, sample_messages):
    """Test response caching works."""
    # First call - should cache
    mock_response = {
        "id": "cached-response",
        "choices": [{"message": {"content": "Cached answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }
    mock_acompletion.return_value = mock_response
    
    result1 = await proxy.acompletion("gpt-4", sample_messages)
    assert mock_cache.call_count == 1
    
    # Second call with same content - should hit cache
    mock_acompletion.return_value = {"choices": [{"message": {"content": "Different answer"}}]}
    result2 = await proxy.acompletion("gpt-4", sample_messages)
    
    # Should return cached response
    assert result2["id"] == "cached-response"
    assert mock_acompletion.call_count == 1  # Should only call once

@patch('litellm.acompletion')
async def test_completion_retries(mock_acompletion, proxy, sample_messages):
    """Test retry logic with rate limiting."""
    # Mock rate limit error on first two attempts
    mock_acompletion.side_effect = [
        Exception("Rate limit exceeded"),
        Exception("Still rate limited"),
        {"id": "success", "choices": [{"message": {"content": "Success"}}]}  # Third attempt succeeds
    ]
    
    result = await proxy.acompletion("gpt-4", sample_messages, max_retries=2)
    
    # Verify success after retries
    assert result["id"] == "success"
    assert mock_acompletion.call_count == 3  # All three attempts made

@patch('litellm.acompletion')
async def test_completion_fallback(mock_acompletion, proxy, sample_messages, mocker):
    """Test fallback to alternative model."""
    # Mock primary provider failure
    mock_acompletion.side_effect = Exception("Provider down")
    
    # Mock unified manager fallback
    mock_unified = Mock(spec=UnifiedModelManager)
    mock_fallback_response = {"id": "fallback", "choices": [{"message": {"content": "Fallback answer"}}]}
    mock_unified.acompletion.return_value = mock_fallback_response
    
    mocker.patch.object(proxy, 'unified_manager', mock_unified)
    
    result = await proxy.acompletion("gpt-4", sample_messages)
    
    # Verify fallback used
    assert result == mock_fallback_response
    mock_unified.acompletion.assert_called_once_with("gpt-3.5-turbo", sample_messages)

def test_cost_calculation(proxy):
    """Test cost calculation functionality."""
    # Test with LiteLLM cost function
    with patch('litellm.cost_per_token') as mock_cost:
        mock_cost.return_value = 0.002  # $0.002 per token
        
        cost = proxy._calculate_cost("gpt-4", tokens_in=100, tokens_out=50)
        assert cost == 0.15  # 150 tokens * $0.002
        
        mock_cost.assert_called_once_with(model="gpt-4")
    
    # Test fallback calculation
    with patch('litellm.cost_per_token', side_effect=Exception("Cost function failed")):
        cost = proxy._calculate_cost("unknown-model", tokens_in=100, tokens_out=50)
        # Should use fallback: input $0.0001 + output $0.0003 = $0.04
        assert cost == 0.04

def test_cache_key_generation(proxy, sample_messages):
    """Test cache key generation consistency."""
    key1 = proxy._generate_cache_key("gpt-4", sample_messages)
    key2 = proxy._generate_cache_key("gpt-4", sample_messages)
    
    assert key1 == key2  # Should be deterministic
    assert key1.startswith("gpt-4_")
    assert len(key1) > 10

@patch('litellm.acompletion')
async def test_synchronous_wrapper(mock_acompletion, sample_messages):
    """Test synchronous completion wrapper."""
    from lola.libs.litellm.proxy import completion
    
    # Mock async response
    mock_response = {"id": "sync-test", "choices": [{"message": {"content": "Sync answer"}}]}
    mock_acompletion.return_value = mock_response
    
    # Call synchronous wrapper
    result = completion("gpt-4", sample_messages)
    
    assert result == mock_response
    mock_acompletion.assert_called_once()

def test_proxy_singleton():
    """Test singleton pattern works."""
    proxy1 = get_litellm_proxy()
    proxy2 = get_litellm_proxy()
    
    assert proxy1 is proxy2  # Same instance

def test_disabled_proxy_fallback(mock_config):
    """Test proxy falls back to unified manager when disabled."""
    with patch('lola.utils.config.get_config', return_value={"use_litellm": False}):
        proxy = LolaLiteLLMProxy()
        
        # Should delegate to unified manager
        with patch('lola.agnostic.unified.UnifiedModelManager') as mock_unified:
            mock_unified_instance = Mock()
            mock_unified.return_value = mock_unified_instance
            mock_unified_instance.acompletion.return_value = {"id": "unified-fallback"}
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(proxy.acompletion("gpt-4", []))
                assert result == {"id": "unified-fallback"}
                mock_unified_instance.acompletion.assert_called_once()
            finally:
                loop.close()

# Integration test with Prometheus metrics
@patch('lola.libs.prometheus.exporter.get_lola_prometheus')
async def test_prometheus_integration(mock_prometheus, proxy, sample_messages):
    """Test Prometheus metrics integration."""
    mock_exporter = Mock()
    mock_llm_call = Mock()
    mock_exporter.llm_call.return_value.__enter__.return_value = mock_llm_call
    mock_prometheus.return_value = mock_exporter
    
    await proxy.acompletion("gpt-4", sample_messages)
    
    # Verify metrics context was used
    mock_exporter.llm_call.assert_called_once_with("gpt-4", "openai", "chat_completion")
    mock_llm_call.__exit__.assert_called_once()

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(proxy):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_litellm_proxy.py -v --cov=lola/libs/litellm --cov-report=html
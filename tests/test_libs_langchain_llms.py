# Standard imports
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Callable

# Local
from lola.libs.langchain.llms import LangChainLLMWrapper
from lola.agnostic.unified import UnifiedModelManager
from lola.utils.config import get_config

"""
Test file for LangChain LLM wrapper.
Purpose: Ensures LangChain LLMs integrate properly with LOLA's unified manager.
Full Path: lola-os/tests/test_libs_langchain_llms.py
"""

@pytest.fixture
def mock_config():
    """Mock config with all providers enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "use_langchain": True,
            "openai_api_key": "fake_openai_key",
            "anthropic_api_key": "fake_anthropic_key",
            "ollama_base_url": "http://localhost:11434",
            "default_temperature": 0.7
        }
        yield mock

@pytest.fixture
def wrapper(mock_config):
    """Fixture for LangChainLLMWrapper."""
    return LangChainLLMWrapper()

@pytest.fixture
def mock_unified_manager():
    """Mock UnifiedModelManager for fallback testing."""
    with patch('lola.agnostic.unified.UnifiedModelManager') as mock:
        instance = mock.return_value
        instance.get_fallback_result.return_value = "fallback response"
        yield instance

def test_wrapper_initialization(wrapper):
    """Test LLM wrapper initializes correctly."""
    assert wrapper.enabled is True
    assert isinstance(wrapper.unified_manager, UnifiedModelManager)
    assert len(wrapper._instances) == 0

def test_supported_providers():
    """Test all supported providers are available."""
    expected = ["openai", "anthropic", "ollama"]
    assert set(wrapper.SUPPORTED_PROVIDERS.keys()) == set(expected)

def test_unknown_provider_raises(wrapper):
    """Test unknown provider raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported LangChain provider"):
        wrapper.wrap_llm("unknown_provider", "model")

@patch('sentry_sdk.capture_exception')
def test_llm_creation_error_handling(mock_sentry, wrapper):
    """Test LLM creation handles errors with Sentry logging."""
    with patch('lola.libs.langchain.llms.OpenAI') as mock_openai:
        mock_openai.side_effect = Exception("API connection failed")
        
        with pytest.raises(Exception):
            wrapper.wrap_llm("openai", "gpt-4")
        
        mock_sentry.assert_called_once()

@patch('lola.libs.langchain.llms.OpenAI')
def test_openai_llm_creation(mock_openai_class, wrapper, mock_config):
    """Test OpenAI LLM creation with config integration."""
    mock_llm_instance = MagicMock()
    mock_openai_class.return_value = mock_llm_instance
    
    completion_fn = wrapper.wrap_llm("openai", "gpt-3.5-turbo")
    
    assert callable(completion_fn)
    assert len(wrapper._instances) == 1
    assert "openai_gpt-3.5-turbo" in wrapper._instances
    mock_openai_class.assert_called_once_with(
        model="gpt-3.5-turbo",
        api_key="fake_openai_key",
        temperature=0.7
    )

@patch('lola.libs.langchain.llms.ChatAnthropic')
def test_anthropic_llm_creation(mock_anthropic_class, wrapper, mock_config):
    """Test Anthropic LLM creation."""
    mock_llm_instance = MagicMock()
    mock_anthropic_class.return_value = mock_llm_instance
    
    completion_fn = wrapper.wrap_llm("anthropic", "claude-3-sonnet-20240229")
    
    assert callable(completion_fn)
    mock_anthropic_class.assert_called_once_with(
        model="claude-3-sonnet-20240229",
        anthropic_api_key="fake_anthropic_key",
        temperature=0.7
    )

@patch('lola.libs.langchain.llms.Ollama')
def test_ollama_llm_creation(mock_ollama_class, wrapper, mock_config):
    """Test Ollama local LLM creation."""
    mock_llm_instance = MagicMock()
    mock_ollama_class.return_value = mock_llm_instance
    
    completion_fn = wrapper.wrap_llm("ollama", "llama2")
    
    assert callable(completion_fn)
    mock_ollama_class.assert_called_once_with(
        model="llama2",
        base_url="http://localhost:11434",
        temperature=0.7
    )

def test_completion_success(wrapper, mock_unified_manager):
    """Test successful LLM completion."""
    # Mock a working OpenAI instance
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Hello from LangChain!"
    
    wrapper._instances["openai_gpt-3.5-turbo"] = mock_llm
    
    completion_fn = wrapper.wrap_llm("openai", "gpt-3.5-turbo")
    result = completion_fn("Hello, how are you?")
    
    assert result == "Hello from LangChain!"
    mock_llm.invoke.assert_called_once_with("Hello, how are you?")

def test_completion_with_fallback(wrapper, mock_unified_manager):
    """Test completion uses fallback when LangChain fails."""
    # Mock failing LLM
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("API timeout")
    
    wrapper._instances["openai_gpt-3.5-turbo"] = mock_llm
    
    completion_fn = wrapper.wrap_llm("openai", "gpt-3.5-turbo")
    
    with pytest.raises(Exception):
        completion_fn("test prompt")
    
    # Verify fallback was attempted
    mock_unified_manager.get_fallback_result.assert_called_once_with(
        "langchain_openai", "test prompt"
    )

def test_cache_management(wrapper):
    """Test LLM instance caching works."""
    # First creation populates cache
    completion1 = wrapper.wrap_llm("openai", "gpt-3.5-turbo")
    assert len(wrapper._instances) == 1
    
    # Second call should use cached instance
    completion2 = wrapper.wrap_llm("openai", "gpt-3.5-turbo")
    assert len(wrapper._instances) == 1  # Still one instance
    assert completion1 != completion2  # Different function objects but same instance

def test_clear_cache(wrapper):
    """Test cache clearing works."""
    # Populate cache
    wrapper.wrap_llm("openai", "gpt-3.5-turbo")
    assert len(wrapper._instances) == 1
    
    # Clear cache
    wrapper.clear_cache()
    assert len(wrapper._instances) == 0

# Integration test for factory functions
def test_factory_functions():
    """Test convenience factory functions."""
    from lola.libs.langchain.llms import create_langchain_openai_llm
    
    completion_fn = create_langchain_openai_llm("gpt-3.5-turbo")
    assert callable(completion_fn)

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(wrapper):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_langchain_llms.py -v --cov=lola/libs/langchain/llms --cov-report=html
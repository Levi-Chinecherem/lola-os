# Standard imports
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import os
import shutil

# Local
from lola.libs.ollama.runner import LolaOllamaRunner, get_ollama_runner
from lola.utils.config import get_config

"""
Test file for Ollama runner integration.
Purpose: Ensures Ollama lifecycle management, model pulling, and health 
         monitoring work correctly with various configurations.
Full Path: lola-os/tests/test_libs_ollama_runner.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration for Ollama testing."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "ollama_base_url": "http://localhost:11434",
            "ollama_models_dir": str(tempfile.mkdtemp()),
            "ollama_auto_install": True,
            "ollama_use_docker": False,  # Test native first
            "ollama_gpu_support": False,
            "ollama_cache_size_gb": 2,
            "sentry_dsn": "test_dsn"
        }
        yield mock

@pytest.fixture
def runner(mock_config):
    """Fixture for LolaOllamaRunner."""
    return LolaOllamaRunner()

@pytest.fixture
def temp_models_dir():
    """Temporary models directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_runner_initialization(runner, temp_models_dir):
    """Test runner initializes with correct configuration."""
    assert runner.ollama_url == "http://localhost:11434"
    assert Path(runner.models_dir).exists() or runner.auto_install
    assert runner.use_docker is False  # From mock config
    assert runner.gpu_support is False

def test_system_detection(runner):
    """Test system capability detection."""
    assert runner.architecture in ["x86_64", "arm64", "amd64"]  # Common architectures
    assert runner.os in ["linux", "darwin", "windows"]
    
    # GPU detection (mocked)
    with patch('lola.libs.ollama.runner.subprocess') as mock_subprocess:
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GPU\n"
        mock_subprocess.run.return_value = mock_result
        
        runner._detect_nvidia_gpu()
        assert runner.nvidia_gpu is True

def test_docker_detection(runner):
    """Test Docker availability detection."""
    with patch('lola.libs.ollama.runner.docker') as mock_docker:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_docker.from_env.return_value = mock_client
        
        runner._check_docker()
        assert runner.docker_available is True
    
    with patch('lola.libs.ollama.runner.docker') as mock_docker:
        mock_docker.from_env.side_effect = Exception("Docker not found")
        
        runner._check_docker()
        assert runner.docker_available is False

@patch('lola.libs.ollama.runner.requests')
async def test_health_check_success(mock_requests, runner):
    """Test successful health check."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "llama2"}]}
    mock_requests.get.return_value = mock_response
    
    is_healthy = await runner.is_healthy()
    assert is_healthy is True
    mock_requests.get.assert_called_once_with(
        "http://localhost:11434/api/tags", timeout=10
    )

@patch('lola.libs.ollama.runner.requests')
async def test_health_check_failure(mock_requests, runner):
    """Test health check failure."""
    mock_requests.get.side_effect = requests.exceptions.RequestException("Connection failed")
    
    is_healthy = await runner.is_healthy()
    assert is_healthy is False

@patch('lola.libs.ollama.runner.requests')
async def test_model_pull_success(mock_requests, runner, temp_models_dir):
    """Test successful model pull."""
    # Mock streaming response
    mock_response = Mock()
    mock_response.status_code = 200
    
    # Mock JSON lines for streaming
    json_lines = [
        json.dumps({"status": "pulling manifest"}),
        json.dumps({"status": "pulling 1.0.0"}),
        json.dumps({"status": "verifying sha256"}),
        json.dumps({"status": "success"})
    ]
    
    def iter_lines_side_effect():
        for line in json_lines:
            yield line.encode()
    
    mock_response.iter_lines = Mock(side_effect=iter_lines_side_effect)
    mock_requests.post.return_value = mock_response
    
    # Test pull
    success = await runner.pull_model("llama2", timeout=10)
    assert success is True
    mock_requests.post.assert_called_once_with(
        "http://localhost:11434/api/pull",
        json={"name": "llama2"},
        stream=True,
        timeout=10
    )

@patch('lola.libs.ollama.runner.requests')
async def test_model_pull_progress(mock_requests, runner, caplog):
    """Test model pull progress logging."""
    mock_response = Mock()
    mock_response.status_code = 200
    
    # Progress updates
    progress_lines = [
        json.dumps({"total": 100, "completed": 25, "status": "downloading"}),
        json.dumps({"total": 100, "completed": 75, "status": "downloading"}),
        json.dumps({"status": "success"})
    ]
    
    def iter_lines_with_progress():
        for line in progress_lines:
            yield line.encode()
            # Simulate 5-second updates
            time.sleep(0.1)  # Reduced for test
    
    mock_response.iter_lines = Mock(side_effect=iter_lines_with_progress)
    mock_requests.post.return_value = mock_response
    
    # Run with shorter timeout for test
    with caplog.at_level("INFO"):
        success = await asyncio.wait_for(
            runner.pull_model("mistral", timeout=5), 
            timeout=2  # Short timeout for test
        )
    
    # Verify progress logging
    assert "Model pull progress: 25.0%" in caplog.text
    assert "Model pull progress: 75.0%" in caplog.text

@patch('lola.libs.ollama.runner.requests')
async def test_model_already_exists(mock_requests, runner):
    """Test behavior when model already exists."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "llama2", "size": 4000000000, "digest": "sha256:abc123"}]
    }
    mock_requests.get.return_value = mock_response
    
    with patch.object(runner, 'pull_model') as mock_pull:
        success = await runner.pull_model("llama2")
    
    # Should not call pull
    mock_pull.assert_not_called()
    assert success is True

@patch('lola.libs.ollama.runner.requests')
async def test_list_models(mock_requests, runner):
    """Test model listing functionality."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "llama2", "size": 4000000000, "digest": "sha256:abc123"},
            {"name": "mistral", "size": 7000000000, "digest": "sha256:def456"}
        ]
    }
    mock_requests.get.return_value = mock_response
    
    models = await runner.list_models()
    
    assert len(models) == 2
    assert models[0]["name"] == "llama2"
    assert models[0]["size"] == 4000000000
    assert "details" in models[0]

@patch('lola.libs.ollama.runner.requests')
async def test_run_completion_success(mock_requests, runner):
    """Test successful completion execution."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama2",
        "response": "Hello from Ollama!",
        "done": True,
        "context": [1, 2, 3]
    }
    mock_requests.post.return_value = mock_response
    
    # Ensure running first
    with patch.object(runner, 'ensure_ollama_running', return_value=True):
        result = await runner.run_completion("llama2", "Hello!")
    
    assert result["response"] == "Hello from Ollama!"
    assert result["done"] is True
    mock_requests.post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={"model": "llama2", "prompt": "Hello!", "stream": False},
        timeout=120
    )

@patch('lola.libs.ollama.runner.requests')
async def test_run_completion_stream(mock_requests, runner):
    """Test streaming completion."""
    mock_response = Mock()
    mock_response.status_code = 200
    
    # Mock streaming lines
    stream_lines = [
        '{"response": "Hello", "done": false}\n',
        '{"response": " from", "done": false}\n', 
        '{"response": " Ollama!", "done": true}\n'
    ]
    
    def iter_lines_mock():
        for line in stream_lines:
            yield line.encode()
    
    mock_response.iter_lines = Mock(side_effect=iter_lines_mock)
    mock_requests.post.return_value = mock_response
    
    with patch.object(runner, 'ensure_ollama_running', return_value=True):
        stream = await runner.run_completion("llama2", "Hello!", stream=True)
        
        # Collect stream
        collected = []
        async for line in stream:
            if line:
                collected.append(line.decode())
        
        full_response = "".join(collected)
        assert "Hello from Ollama!" in full_response
        assert len(collected) == 3

@patch('lola.libs.ollama.runner.requests')
async def test_run_completion_ollama_not_running(mock_requests, runner):
    """Test behavior when Ollama is not running."""
    with patch.object(runner, 'ensure_ollama_running', return_value=False):
        with pytest.raises(RuntimeError, match="Ollama not available"):
            await runner.run_completion("llama2", "test")

def test_resource_usage(runner):
    """Test resource monitoring functionality."""
    # Mock process for testing
    with patch.object(runner, '_process', Mock()):
        with patch('psutil.Process') as mock_process:
            mock_psutil = Mock()
            mock_psutil.cpu_percent.return_value = 25.5
            mock_psutil.memory_info.return_value = tp.SimpleNamespace(rss=104857600)  # 100MB
            mock_process.return_value = mock_psutil
            
            usage = runner.resource_usage
            
            assert usage["status"] == "running"
            assert usage["cpu_percent"] == 25.5
            assert usage["memory_mb"] == 100.0

def test_docker_cleanup(mocker):
    """Test Docker cleanup functionality."""
    mock_docker = mocker.patch('lola.libs.ollama.runner.docker')
    mock_client = Mock()
    mock_container = Mock()
    mock_container.status = "running"
    mock_client.containers.get.return_value = mock_container
    
    runner = LolaOllamaRunner()
    runner._docker_client = mock_client
    runner._process = True
    
    # Test cleanup
    runner._cleanup_docker()
    
    mock_container.stop.assert_called_once_with(timeout=30)
    mock_container.remove.assert_called_once()
    mock_client.containers.get.assert_called_once_with("lola-ollama", ignore_removed=True)

def test_singleton_pattern():
    """Test runner singleton works."""
    runner1 = get_ollama_runner()
    runner2 = get_ollama_runner()
    
    assert runner1 is runner2

# Integration test for model management
@pytest.mark.integration
@patch('lola.libs.ollama.runner.requests')
async def test_full_model_lifecycle(mock_requests, runner, temp_models_dir):
    """Integration test for complete model management flow."""
    # Configure for temp directory
    runner.models_dir = temp_models_dir
    
    # Mock API responses for full lifecycle
    mock_requests.get.side_effect = [
        # Initial tags (empty)
        Mock(status_code=200, json=lambda: {"models": []}),
        # After pull
        Mock(status_code=200, json=lambda: {
            "models": [{"name": "llama2", "size": 4000000000}]
        })
    ]
    
    mock_pull_response = Mock(status_code=200)
    def mock_iter_lines():
        yield b'{"status": "success"}\n'
    mock_pull_response.iter_lines = Mock(side_effect=mock_iter_lines)
    mock_requests.post.return_value = mock_pull_response
    
    # Test full lifecycle
    assert await runner.ensure_ollama_running(timeout=5)  # Should be mocked as running
    
    # Pull model
    success = await runner.pull_model("llama2")
    assert success is True
    
    # List models
    models = await runner.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "llama2"
    
    # Run completion
    completion = await runner.run_completion("llama2", "Hello")
    assert "response" in completion

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(runner):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_ollama_runner.py -v --cov=lola/libs/ollama --cov-report=html
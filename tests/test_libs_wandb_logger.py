# Standard imports
import pytest
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
from typing import Dict, Any, List
from pathlib import Path

# Local
from lola.libs.wandb.logger import (
    LolaWandBLogger, get_wandb_logger, ExperimentMode
)
from lola.core.agent import BaseAgent
from lola.utils.config import get_config

"""
Test file for W&B logger integration.
Purpose: Ensures experiment tracking, artifact management, and sweep 
         integration work correctly with various run modes.
Full Path: lola-os/tests/test_libs_wandb_logger.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration with W&B enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "wandb_enabled": True,
            "wandb_api_key": "test-api-key",
            "wandb_project": "test-project",
            "wandb_entity": "test-entity",
            "wandb_tags": ["test-tag"],
            "wandb_mode": "online",
            "wandb_offline_dir": str(tempfile.mkdtemp()),
            "wandb_save_code": True,
            "sentry_dsn": "test-dsn"
        }
        yield mock

@pytest.fixture
def logger_instance(mock_config):
    """Fixture for LolaWandBLogger instance."""
    return get_wandb_logger()

@pytest.fixture
def mock_run():
    """Mock W&B run instance."""
    mock_run = Mock()
    mock_run.id = "test-run-id"
    mock_run.config = {}
    mock_run._label_algo = None
    return mock_run

def test_logger_initialization(logger_instance):
    """Test logger initializes correctly."""
    assert logger_instance.enabled is True
    assert logger_instance.project_name == "test-project"
    assert logger_instance.entity == "test-entity"
    assert logger_instance.default_tags == ["lola-os", "v1.0", "test-tag"]
    assert logger_instance.mode == ExperimentMode.ONLINE

def test_logger_disabled(mock_config):
    """Test logger behavior when W&B is disabled."""
    with patch('lola.utils.config.get_config', return_value={"wandb_enabled": False}):
        logger = LolaWandBLogger()
        assert logger.enabled is False
        
        # Should return mock run
        with logger.start_run(name="test") as run:
            assert hasattr(run, 'log')
            assert hasattr(run, 'finish')

def test_start_run_context(mock_config, logger_instance, mock_run):
    """Test run context manager functionality."""
    with patch('wandb.init') as mock_init:
        mock_init.return_value = mock_run
        
        with logger_instance.start_run(
            name="test-run",
            config={"param": 42},
            tags=["experiment"],
            group="agent-tuning"
        ) as run:
            
            # Verify run was started
            mock_init.assert_called_once()
            call_args = mock_init.call_args[1]
            assert call_args["name"] == "test-run"
            assert call_args["config"] == {"param": 42}
            assert call_args["tags"] == ["lola-os", "v1.0", "test-tag", "experiment"]
            assert call_args["group"] == "agent-tuning"
            assert call_args["mode"] == "online"
            assert run == mock_run
        
        # Verify run was finished
        mock_run.finish.assert_called_once()

def test_run_metadata_enrichment(mock_config, logger_instance, mock_run):
    """Test LOLA-specific metadata is added to runs."""
    with patch('wandb.init') as mock_init:
        mock_init.return_value = mock_run
        
        with logger_instance.start_run(name="metadata-test"):
            pass
        
        # Verify LOLA metadata was set
        assert mock_run.config.get("lola_version") == "1.0.0"
        assert mock_run.config.get("lola_agent_version") == "1.0.0"

def test_log_metrics(logger_instance, mock_run):
    """Test metrics logging to active run."""
    with patch('wandb.init') as mock_init:
        mock_init.return_value = mock_run
        
        with logger_instance.start_run(name="metrics-test"):
            logger_instance.log_metrics({
                "accuracy": 0.95,
                "loss": 0.123,
                "f1_score": 0.92
            }, step=100)
        
        # Verify metrics were logged
        mock_run.log.assert_called_once_with(
            {"accuracy": 0.95, "loss": 0.123, "f1_score": 0.92},
            step=100
        )

def test_log_agent_performance(logger_instance, mock_run):
    """Test agent performance logging with context."""
    # Mock agent
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.__class__.__name__ = "TestAgent"
    mock_agent.model = "gpt-4"
    
    with patch('wandb.init') as mock_init:
        mock_init.return_value = mock_run
        
        with logger_instance.start_run(name="agent-test"):
            logger_instance.log_agent_performance(
                mock_agent,
                {"accuracy": 0.88, "response_time": 2.3, "token_usage": 150},
                step=50
            )
        
        # Verify enriched metrics
        logged_data = mock_run.log.call_args[0][0]
        assert logged_data["agent_type"] == "TestAgent"
        assert logged_data["model_name"] == "gpt-4"
        assert logged_data["tools_count"] == 0  # Default
        assert logged_data["accuracy"] == 0.88

@patch('lola.libs.prometheus.exporter.get_lola_prometheus')
def test_prometheus_integration(mock_prometheus, logger_instance, mock_run):
    """Test Prometheus integration in agent logging."""
    mock_exporter = Mock()
    mock_prometheus.return_value = mock_exporter
    
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.__class__.__name__ = "PerfAgent"
    
    with patch('wandb.init') as mock_init:
        mock_init.return_value = mock_run
        
        with logger_instance.start_run(name="perf-test"):
            logger_instance.log_agent_performance(
                mock_agent,
                {"duration": 1.5, "success_rate": 0.95, "token_usage": 200}
            )
        
        # Verify Prometheus call
        mock_exporter.record_agent_performance.assert_called_once_with(
            agent_type="PerfAgent",
            operation="evaluation",
            duration=1.5,
            success_rate=0.95,
            token_usage=200
        )

def test_log_model_artifact(tmpdir, logger_instance, mock_run):
    """Test model artifact logging."""
    # Create mock model directory
    model_dir = tmpdir.mkdir("test_model")
    model_file = model_dir.join("pytorch_model.bin")
    model_file.write("mock model weights", encoding="latin1")
    config_file = model_dir.join("config.json")
    config_file.write('{"vocab_size": 50000}', encoding="utf-8")
    
    with patch('wandb.init') as mock_init:
        mock_init.return_value = mock_run
        mock_artifact = Mock()
        mock_run.log_artifact.return_value = mock_artifact
        
        with logger_instance.start_run(name="artifact-test"):
            artifact = logger_instance.log_model_artifact(
                str(model_dir),
                name="test-model-v1",
                description="Test model artifact",
                metadata={"accuracy": 0.92, "size": "7B"}
            )
        
        # Verify artifact creation
        mock_run.log_artifact.assert_called_once()
        call_args = mock_run.log_artifact.call_args[0][0]
        assert call_args.name == "test-model-v1"
        assert call_args.type == "model"
        assert call_args.metadata == {"accuracy": 0.92, "size": "7B"}

def test_log_dataset_artifact(tmpdir, logger_instance, mock_run):
    """Test dataset artifact logging."""
    # Create sample dataset
    dataset_dir = tmpdir.mkdir("test_dataset")
    train_file = dataset_dir.join("train.jsonl")
    train_file.write(
        '{"text": "Sample training data"}\n{"text": "More training data"}',
        encoding="utf-8"
    )
    readme_file = dataset_dir.join("README.md")
    readme_file.write("# Test Dataset\nSample dataset for testing.", encoding="utf-8")
    
    with patch('wandb.init') as mock_init:
        mock_init.return_value = mock_run
        mock_artifact = Mock()
        mock_run.log_artifact.return_value = mock_artifact
        
        with logger_instance.start_run(name="dataset-test"):
            artifact = logger_instance.log_dataset_artifact(
                str(dataset_dir),
                name="test-dataset-v1",
                description="Training dataset with 2 samples"
            )
        
        # Verify dataset artifact
        call_args = mock_run.log_artifact.call_args[0][0]
        assert call_args.name == "test-dataset-v1"
        assert call_args.type == "dataset"
        assert "README.md" in [f.name for f in call_args.files]

def test_start_sweep(logger_instance):
    """Test sweep creation."""
    sweep_config = {
        "method": "grid",
        "parameters": {
            "learning_rate": {"values": [1e-4, 2e-4, 5e-4]},
            "batch_size": {"values": [8, 16, 32]}
        }
    }
    
    with patch('wandb.sweep') as mock_sweep:
        mock_sweep.return_value = "test-sweep-id"
        
        sweep_id = logger_instance.start_sweep(sweep_config)
        
        mock_sweep.assert_called_once_with(sweep_config, project="test-project")
        assert sweep_id == "test-sweep-id"

def test_log_sweep_agent(logger_instance):
    """Test agent configuration logging to sweep."""
    sweep_id = "test-sweep"
    agent_config = {
        "agent_type": "ReActAgent",
        "model": "gpt-4",
        "tools": ["web_search", "calculator"],
        "temperature": 0.7
    }
    
    with patch('wandb.Api') as mock_api:
        mock_sweep = Mock()
        mock_sweep.config = {"method": "grid"}
        mock_api.return_value.sweep.return_value = mock_sweep
        
        logger_instance.log_sweep_agent(sweep_id, agent_config)
        
        mock_sweep.update_config.assert_called_once()
        updated_config = mock_sweep.update_config.call_args[0][0]
        assert updated_config["agent_type"] == "ReActAgent"
        assert "temperature" in updated_config

def test_multiple_active_runs(logger_instance):
    """Test multiple concurrent runs."""
    run_configs = [
        {"name": "run1", "config": {"param1": 1}},
        {"name": "run2", "config": {"param2": 2}}
    ]
    
    runs = []
    with patch('wandb.init') as mock_init:
        for i, config in enumerate(run_configs):
            mock_run = Mock()
            mock_run.id = f"run-{i}"
            mock_init.side_effect = lambda **kwargs: mock_run
            
            with logger_instance.start_run(name=config["name"], config=config["config"]):
                runs.append(mock_run)
        
        # Verify separate runs
        assert len(logger_instance.active_runs) == 2
        assert set(logger_instance.active_runs.keys()) == {"run-0", "run-1"}

def test_run_finishing(logger_instance):
    """Test proper run cleanup."""
    with patch('wandb.init') as mock_init:
        mock_run = Mock()
        mock_init.return_value = mock_run
        
        run_id = None
        with logger_instance.start_run(name="cleanup-test") as run:
            run_id = run.id
        
        # Verify run was finished and removed
        mock_run.finish.assert_called_once()
        assert run_id not in logger_instance.active_runs

def test_offline_mode(tmpdir, logger_instance):
    """Test offline mode functionality."""
    offline_dir = str(tmpdir)
    
    # Configure for offline
    with patch('lola.utils.config.get_config', return_value={
        "wandb_enabled": True,
        "wandb_mode": "offline",
        "wandb_offline_dir": offline_dir
    }):
        logger = LolaWandBLogger()
        assert logger.mode == ExperimentMode.OFFLINE
        assert Path(offline_dir).exists()
        
        # Test offline run
        with patch('wandb.init') as mock_init:
            with logger.start_run(name="offline-test"):
                pass
        
        mock_init.assert_called_once_with(mode="offline", dir=offline_dir)

def test_artifact_waiting(logger_instance, mock_run):
    """Test artifact waiting behavior."""
    with patch('wandb.init') as mock_init:
        mock_init.return_value = mock_run
        mock_artifact = Mock()
        mock_run.log_artifact.return_value = mock_artifact
        
        with logger_instance.start_run(name="wait-test"):
            artifact = logger_instance.log_model_artifact("./test_model")
        
        # Verify wait was called
        mock_artifact.wait.assert_called_once()

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(logger_instance):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_wandb_logger.py -v --cov=lola/libs/wandb --cov-report=html
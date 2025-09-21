# Standard imports
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import yaml
from typing import Dict, Any
import time
import torch

# Local
from lola.libs.axolotl.trainer import (
    LolaAxolotlTrainer, FineTuneConfig, get_axolotl_trainer,
    TrainingStatus
)
from lola.utils.config import get_config

"""
Test file for Axolotl trainer integration.
Purpose: Ensures fine-tuning workflow, dataset preparation, config generation, 
         and job management work correctly with mock training processes.
Full Path: lola-os/tests/test_libs_axolotl_trainer.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration with Model Garden enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "model_garden_enabled": True,
            "model_garden_output_dir": str(tempfile.mkdtemp()),
            "model_garden_max_jobs": 1,
            "model_garden_auto_eval": True,
            "sentry_dsn": "test_dsn"
        }
        yield mock

@pytest.fixture
def trainer(mock_config):
    """Fixture for LolaAxolotlTrainer."""
    return get_axolotl_trainer()

@pytest.fixture
def sample_training_data(tmpdir):
    """Sample training data file."""
    data_file = tmpdir.join("training_data.jsonl")
    sample_data = [
        '{"instruction": "Explain LOLA OS", "input": "", "output": "LOLA OS is an AI agent framework."}',
        '{"instruction": "What is vector search?", "input": "", "output": "Vector search finds similar embeddings."}',
        '{"instruction": "Define fine-tuning", "input": "", "output": "Fine-tuning adapts pre-trained models."}'
    ]
    data_file.write_text("\n".join(sample_data), encoding="utf-8")
    return str(data_file)

@pytest.fixture
def sample_eval_data(tmpdir):
    """Sample evaluation data file."""
    eval_file = tmpdir.join("eval_data.jsonl")
    sample_data = [
        '{"instruction": "Summarize LOLA", "input": "", "output": "LOLA OS builds AI agents."}',
        '{"instruction": "What is RAG?", "input": "", "output": "RAG uses retrieval for generation."}'
    ]
    eval_file.write_text("\n".join(sample_data), encoding="utf-8")
    return str(eval_file)

def test_trainer_initialization(trainer, tmpdir):
    """Test trainer initializes correctly."""
    assert trainer.enabled is True
    assert Path(trainer.output_base_dir).exists()
    assert len(trainer.active_jobs) == 0
    assert len(trainer.completed_jobs) == 0

def test_trainer_disabled(mock_config):
    """Test trainer behavior when Model Garden is disabled."""
    with patch('lola.utils.config.get_config', return_value={"model_garden_enabled": False}):
        trainer = LolaAxolotlTrainer()
        assert trainer.enabled is False
        
        with pytest.raises(RuntimeError, match="Model Garden not enabled"):
            trainer.fine_tune(FineTuneConfig())

def test_fine_tune_config_dataclass():
    """Test FineTuneConfig dataclass."""
    config = FineTuneConfig(
        base_model="gpt2",
        num_epochs=2,
        learning_rate=1e-4,
        output_dir="./test_output"
    )
    
    assert config.base_model == "gpt2"
    assert config.num_epochs == 2
    assert config.learning_rate == 1e-4
    assert Path(config.output_dir).name == "test_output"

@patch('lola.libs.axolotl.trainer.datasets')
@patch('lola.libs.axolotl.trainer.asyncio.create_subprocess_exec')
async def test_dataset_preparation(mock_subprocess, mock_datasets, 
                                 trainer, sample_training_data, tmpdir):
    """Test training dataset preparation."""
    # Mock dataset loading
    mock_dataset = Mock()
    mock_dataset.map.return_value = Mock()
    mock_dataset.to_json.return_value = None
    mock_datasets.load_dataset.return_value = mock_dataset
    
    config = FineTuneConfig(output_dir=str(tmpdir))
    
    await trainer._prepare_dataset(sample_training_data, config, "test-job")
    
    # Verify dataset was loaded and formatted
    mock_datasets.load_dataset.assert_called_once_with(
        'json', data_files=sample_training_data, split='train'
    )
    
    # Verify formatting function was applied
    format_call = mock_dataset.map.call_args[0][1]
    assert callable(format_call)
    
    # Verify output path
    expected_path = Path(tmpdir) / "test-job_training_data.jsonl"
    mock_dataset.to_json.assert_called_once_with(
        expected_path, orient='records', lines=True
    )
    
    # Verify config was updated
    assert config.train_dataset == str(expected_path)

@patch('lola.libs.axolotl.trainer.yaml')
@patch('lola.libs.axolotl.trainer.Path.mkdir')
def test_config_generation(mock_mkdir, mock_yaml, trainer, tmpdir):
    """Test Axolotl configuration generation."""
    config = FineTuneConfig(
        base_model="test-model",
        output_dir=str(tmpdir),
        num_epochs=1,
        learning_rate=1e-5
    )
    
    run_name = "test-run"
    job_id = "test-job"
    
    axolotl_config = trainer._generate_axolotl_config(config, run_name, job_id)
    
    # Verify config structure
    assert "base_model" in axolotl_config
    assert axolotl_config["base_model"] == "test-model"
    assert "num_epochs" in axolotl_config
    assert axolotl_config["num_epochs"] == 1
    assert "wandb_project" in axolotl_config
    assert axolotl_config["wandb_project"] == "lola-model-garden"
    
    # Verify config was saved
    expected_path = Path(tmpdir) / f"{job_id}_{run_name}" / "axolotl_config.yml"
    mock_yaml.dump.assert_called_once()
    mock_mkdir.assert_called_once()

@patch('lola.libs.axolotl.trainer.asyncio.create_subprocess_exec')
@patch('lola.libs.axolotl.trainer.wandb')
async def test_training_job_launch(mock_wandb, mock_subprocess, 
                                  trainer, tmpdir):
    """Test training job launching and monitoring."""
    # Mock subprocess
    mock_process = AsyncMock()
    mock_process.wait.return_value = 0  # Success
    mock_process.stdout = AsyncMock()  # Mock streaming
    mock_subprocess.return_value = mock_process
    
    # Mock W&B run
    mock_run = Mock()
    mock_wandb_run = {"run": mock_run}
    mock_wandb.start_run.return_value = mock_wandb_run
    
    config = FineTuneConfig(output_dir=str(tmpdir))
    axolotl_config = {"base_model": "test-model"}
    
    status = await trainer._launch_training_job(
        axolotl_config, str(tmpdir), "test-job", mock_wandb_run
    )
    
    # Verify job was added to active jobs
    assert "test-job" in trainer.active_jobs
    
    # Verify subprocess was called
    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[0][0]
    assert "axolotl.cli.train" in args
    
    # Verify successful completion
    assert status == TrainingStatus.COMPLETED.value
    
    # Verify job moved to completed
    assert "test-job" not in trainer.active_jobs
    assert "test-job" in trainer.completed_jobs
    
    # Verify W&B logging
    mock_run.log.assert_called()

@patch('lola.libs.axolotl.trainer.asyncio.create_subprocess_exec')
async def test_training_job_failure(mock_subprocess, trainer, tmpdir):
    """Test training job failure handling."""
    # Mock failed process
    mock_process = AsyncMock()
    mock_process.wait.return_value = 1  # Failure
    mock_subprocess.return_value = mock_process
    
    config = FineTuneConfig(output_dir=str(tmpdir))
    axolotl_config = {"base_model": "test-model"}
    
    status = await trainer._launch_training_job(
        axolotl_config, str(tmpdir), "test-job", Mock()
    )
    
    # Verify failure status
    assert status == TrainingStatus.FAILED.value
    assert "test-job" in trainer.completed_jobs
    assert trainer.completed_jobs["test-job"]["status"] == TrainingStatus.FAILED

@patch('lola.libs.axolotl.trainer.datasets')
@patch('lola.libs.axolotl.trainer.torch')
async def test_model_evaluation(mock_torch, mock_datasets, 
                               trainer, sample_eval_data, tmpdir):
    """Test model evaluation functionality."""
    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.select.return_value = Mock()
    mock_datasets.load_dataset.return_value = mock_dataset
    
    # Mock model and tokenizer
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_model.eval.return_value = None
    mock_model.generate.return_value = torch.tensor([[101, 102, 103]])
    mock_tokenizer.decode.return_value = "Generated text"
    mock_tokenizer.eos_token_id = 0
    
    # Mock torch operations
    mock_outputs = Mock()
    mock_outputs.loss = Mock(item=1.5)
    mock_model.return_value = mock_outputs
    
    with patch('lola.libs.axolotl.trainer.AutoTokenizer') as mock_tokenizer_class:
        with patch('lola.libs.axolotl.trainer.AutoModelForCausalLM') as mock_model_class:
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model
            
            config = FineTuneConfig(output_dir=str(tmpdir))
            metrics = await trainer.evaluate_model("./test_model", sample_eval_data, "test-job")
    
    # Verify evaluation ran
    assert metrics["status"] != "skipped"
    assert "perplexity" in metrics
    assert "generation_samples" in metrics
    assert isinstance(metrics["generation_samples"], list)

def test_job_status_tracking(trainer):
    """Test job status management."""
    # Create mock active job
    trainer.active_jobs["active-job"] = {
        "status": TrainingStatus.RUNNING,
        "start_time": time.time()
    }
    
    # Create mock completed job
    trainer.completed_jobs["completed-job"] = {
        "status": TrainingStatus.COMPLETED,
        "duration": 300.0
    }
    
    # Test status retrieval
    active_status = trainer.get_job_status("active-job")
    assert active_status["status"] == TrainingStatus.RUNNING.value
    assert active_status["progress"] == "running"
    
    completed_status = trainer.get_job_status("completed-job")
    assert completed_status["status"] == TrainingStatus.COMPLETED.value
    
    # Test unknown job
    assert trainer.get_job_status("unknown-job") is None

def test_config_template_rendering(trainer, tmpdir):
    """Test Axolotl config template rendering."""
    config = FineTuneConfig(
        base_model="test-llama",
        micro_batch_size=2,
        output_dir=str(tmpdir)
    )
    
    # Test template rendering
    axolotl_config = trainer._generate_axolotl_config(config, "test-run", "test-job")
    
    # Verify key parameters are set
    assert axolotl_config["base_model"] == "test-llama"
    assert axolotl_config["micro_batch_size"] == 2
    assert "wandb_project" in axolotl_config

# Integration test for full fine-tuning workflow
@pytest.mark.integration
async def test_full_finetuning_workflow(tmpdir, trainer, sample_training_data, sample_eval_data):
    """Integration test for complete fine-tuning workflow."""
    config = FineTuneConfig(
        base_model="gpt2",  # Small model for testing
        num_epochs=1,
        output_dir=str(tmpdir),
        micro_batch_size=1
    )
    
    # Mock the training process to complete quickly
    with patch('lola.libs.axolotl.trainer.asyncio.create_subprocess_exec') as mock_subprocess:
        mock_process = AsyncMock()
        mock_process.wait.return_value = 0
        mock_subprocess.return_value = mock_process
        
        job_id = await trainer.fine_tune(
            config,
            dataset_path=sample_training_data,
            eval_dataset_path=sample_eval_data,
            run_name="integration-test"
        )
    
    # Verify job completed
    assert job_id is not None
    assert job_id in trainer.completed_jobs
    assert trainer.completed_jobs[job_id]["status"] == TrainingStatus.COMPLETED

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(trainer):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_axolotl_trainer.py -v --cov=lola/libs/axolotl --cov-report=html
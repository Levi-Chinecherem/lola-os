
# Standard imports
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from typing import Dict, Any, List
import time
import shutil

# Local
from lola.libs.huggingface.hub import (
    LolaHuggingFaceHub, get_huggingface_hub, ModelType
)
from lola.utils.config import get_config

"""
Test file for Hugging Face Hub integration.
Purpose: Ensures model and dataset publishing, downloading, and card 
         generation work correctly with mocked API calls.
Full Path: lola-os/tests/test_libs_huggingface_hub.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration with HF Hub enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "hf_hub_enabled": True,
            "hf_token": "test-hf-token",
            "hf_organization": "lola-org",
            "hf_default_visibility": "public",
            "hf_auto_card": True,
            "sentry_dsn": "test-dsn"
        }
        yield mock

@pytest.fixture
def hf_hub(mock_config):
    """Fixture for LolaHuggingFaceHub."""
    return get_huggingface_hub()

@pytest.fixture
def temp_model_dir(tmpdir):
    """Temporary model directory with sample files."""
    model_dir = tmpdir.mkdir("test_model")
    
    # Create model files
    config_file = model_dir.join("config.json")
    config_file.write('{"model_type": "test", "vocab_size": 1000}', encoding="utf-8")
    
    pytorch_file = model_dir.join("pytorch_model.bin")
    pytorch_file.write("mock model weights", encoding="latin1")
    
    tokenizer_file = model_dir.join("tokenizer.json")
    tokenizer_file.write('{"vocab": {"hello": 100}}', encoding="utf-8")
    
    return str(model_dir)

@pytest.fixture
def temp_dataset_dir(tmpdir):
    """Temporary dataset directory."""
    dataset_dir = tmpdir.mkdir("test_dataset")
    
    train_file = dataset_dir.join("train.jsonl")
    train_file.write(
        '{"text": "Sample 1"}\n{"text": "Sample 2"}\n{"text": "Sample 3"}',
        encoding="utf-8"
    )
    
    readme_file = dataset_dir.join("README.md")
    readme_file.write("# Test Dataset\nSample data.", encoding="utf-8")
    
    return str(dataset_dir)

def test_hub_initialization(hf_hub):
    """Test hub initializes correctly."""
    assert hf_hub.enabled is True
    assert hf_hub.api is not None
    assert hf_hub.token == "test-hf-token"
    assert hf_hub.organization == "lola-org"

def test_hub_disabled(mock_config):
    """Test hub behavior when disabled."""
    with patch('lola.utils.config.get_config', return_value={"hf_hub_enabled": False}):
        hub = LolaHuggingFaceHub()
        assert hub.enabled is False
        
        with pytest.raises(RuntimeError, match="Hugging Face Hub integration disabled"):
            hub.publish_model("./test")

@patch('huggingface_hub.HfApi.create_repo')
@patch('huggingface_hub.HfApi.upload_folder')
def test_model_publishing(mock_upload, mock_create, hf_hub, temp_model_dir):
    """Test model publishing workflow."""
    # Mock API responses
    mock_repo_info = {"id": "lola-org/test-model-123", "private": False}
    mock_create.return_value = mock_repo_info
    
    mock_upload_info = {
        "path_or_fileobj": "config.json",
        "repo_type": "model",
        "repo_id": "lola-org/test-model-123"
    }
    mock_upload.return_value = mock_upload_info
    
    # Test publishing
    repo_info = hf_hub.publish_model(
        model_path=temp_model_dir,
        repo_id="lola-org/test-model-123",
        model_type=ModelType.FINETUNED,
        metadata={"accuracy": 0.95},
        tags=["lola", "fine-tuned"]
    )
    
    # Verify repository creation
    mock_create.assert_called_once_with(
        repo_id="lola-org/test-model-123",
        repo_type="model",
        private=False,
        exist_ok=True
    )
    
    # Verify upload
    mock_upload.assert_called_once()
    call_args = mock_upload.call_args[1]
    assert call_args["folder_path"] == temp_model_dir
    assert call_args["repo_id"] == "lola-org/test-model-123"
    assert call_args["repo_type"] == "model"
    assert call_args["commit_message"] == "Upload fine-tuned model"
    assert call_args["tags"] == ["lola-os", "v1.0", "model-type:fine-tuned", "lola", "fine-tuned"]
    
    # Verify return value
    assert repo_info["repo_id"] == "lola-org/test-model-123"
    assert repo_info["model_type"] == "fine-tuned"
    assert "model_size_mb" in repo_info

def test_model_publishing_auto_repo_id(hf_hub, temp_model_dir):
    """Test auto-generation of repository ID."""
    with patch('huggingface_hub.HfApi.create_repo') as mock_create:
        with patch('huggingface_hub.HfApi.upload_folder') as mock_upload:
            mock_create.return_value = {"id": "lola-org/auto-model-123"}
            mock_upload.return_value = {"repo_id": "lola-org/auto-model-123"}
            
            repo_info = hf_hub.publish_model(
                model_path=temp_model_dir,
                model_type=ModelType.BASE
            )
    
    # Verify auto-generated repo_id
    assert repo_info["repo_id"].startswith("lola-org/lola-test_model-")
    mock_create.assert_called_once_with(
        repo_id=repo_info["repo_id"],
        repo_type="model",
        private=False,
        exist_ok=True
    )

def test_model_card_generation(hf_hub, tmpdir):
    """Test automatic model card generation."""
    model_path = str(tmpdir.mkdir("card_test"))
    
    card_content = hf_hub.create_model_card(
        model_type=ModelType.FINETUNED,
        base_model="gpt2",
        fine_tune_config={"epochs": 3, "lr": 1e-4},
        performance_metrics={"accuracy": 0.92, "perplexity": 15.3},
        usage_examples=[
            "```python\nagent = ReActAgent(model='lola-gpt2-finetuned')\n```"
        ],
        limitations="Limited to 512 token context.",
        ethical_considerations="Use responsibly for agent applications."
    )
    
    # Verify card structure and content
    assert "# LOLA" in card_content
    assert "Finetuned" in card_content
    assert "gpt2" in card_content
    assert '"epochs": 3' in card_content
    assert '"accuracy": 0.92' in card_content
    assert "ReActAgent" in card_content
    assert "Limited to 512" in card_content
    assert "Use responsibly" in card_content
    assert "auto-generated by LOLA OS" in card_content

def test_model_card_fallback(hf_hub):
    """Test model card fallback when template generation fails."""
    # Mock template loading failure
    with patch('builtins.open', side_effect=Exception("Template load failed")):
        card_content = hf_hub.create_model_card(
            model_type=ModelType.BASE,
            base_model="fallback-model"
        )
    
    # Verify fallback card
    assert "# LOLA Model" in card_content
    assert "Type: base_model" in card_content
    assert "Generated: " in card_content
    assert "auto-generated by LOLA OS" in card_content

def test_dataset_publishing(hf_hub, temp_dataset_dir):
    """Test dataset publishing workflow."""
    with patch('huggingface_hub.HfApi.create_repo') as mock_create:
        with patch('huggingface_hub.HfApi.upload_folder') as mock_upload:
            # Mock responses
            mock_create.return_value = {"id": "lola-org/test-dataset-123"}
            mock_upload.return_value = {"repo_id": "lola-org/test-dataset-123"}
            
            dataset_info = hf_hub.publish_dataset(
                dataset_path=temp_dataset_dir,
                repo_id="lola-org/test-dataset-123",
                dataset_name="training_data",
                description="Sample training dataset",
                tags=["lola", "training"]
            )
            
            # Verify repository creation
            mock_create.assert_called_once_with(
                repo_id="lola-org/test-dataset-123",
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            
            # Verify upload
            mock_upload.assert_called_once()
            call_args = mock_upload.call_args[1]
            assert call_args["folder_path"].endswith(os.path.dirname(temp_dataset_dir))
            assert call_args["path_in_repo"] == "training_data"
            assert call_args["repo_id"] == "lola-org/test-dataset-123"
            assert call_args["repo_type"] == "dataset"
            assert call_args["commit_message"] == "Upload training_data dataset"
            assert call_args["tags"] == ["lola-os", "v1.0", "dataset", "lola", "training"]
            
            # Verify dataset info
            assert dataset_info["repo_id"] == "lola-org/test-dataset-123"
            assert dataset_info["dataset_name"] == "training_data"
            assert dataset_info["num_files"] == 2  # train.jsonl + README.md
            assert dataset_info["total_size_mb"] > 0

def test_dataset_publishing_auto_repo_id(hf_hub, temp_dataset_dir):
    """Test auto-generation of dataset repository ID."""
    with patch('huggingface_hub.HfApi.create_repo') as mock_create:
        with patch('huggingface_hub.HfApi.upload_folder') as mock_upload:
            mock_create.return_value = {"id": "lola-org/auto-dataset-123"}
            mock_upload.return_value = {"repo_id": "lola-org/auto-dataset-123"}
            
            dataset_info = hf_hub.publish_dataset(
                dataset_path=temp_dataset_dir,
                dataset_name="auto_dataset"
            )
    
    # Verify auto-generated repo_id
    assert dataset_info["repo_id"].startswith("lola-org/lola-dataset-auto_dataset-")
    mock_create.assert_called_once_with(
        repo_id=dataset_info["repo_id"],
        repo_type="dataset",
        private=False,
        exist_ok=True
    )

def test_model_download(hf_hub, tmpdir):
    """Test model downloading functionality."""
    download_dir = str(tmpdir.mkdir("download_test"))
    
    with patch('huggingface_hub.snapshot_download') as mock_download:
        mock_download.return_value = download_dir
        
        downloaded_path = hf_hub.download_model(
            repo_id="gpt2",
            local_dir=download_dir,
            revision="main"
        )
        
        # Verify download call
        mock_download.assert_called_once_with(
            repo_id="gpt2",
            local_dir=download_dir,
            revision="main",
            cache_dir=None,
            local_dir_use_symlinks=False
        )
        
        assert downloaded_path == Path(download_dir)
        assert Path(download_dir).exists()

def test_dataset_download(hf_hub, tmpdir):
    """Test dataset downloading functionality."""
    download_dir = str(tmpdir.mkdir("dataset_download"))
    
    with patch('datasets.load_dataset') as mock_load:
        mock_dataset_dict = MagicMock()
        mock_train_split = Mock()
        mock_test_split = Mock()
        type(mock_dataset_dict).items = Mock(return_value=[
            ("train", mock_train_split),
            ("test", mock_test_split)
        ])
        
        mock_load.return_value = mock_dataset_dict
        
        downloaded_path = hf_hub.download_dataset(
            repo_id="test-dataset",
            local_dir=download_dir,
            revision="v1.0"
        )
        
        # Verify dataset load call
        mock_load.assert_called_once_with(
            "test-dataset",
            revision="v1.0",
            cache_dir=None,
            download_mode="reuse_dataset_if_exists"
        )
        
        # Verify directory structure created
        train_path = Path(download_dir) / "train" / "data.jsonl"
        test_path = Path(download_dir) / "test" / "data.jsonl"
        assert train_path.parent.exists()
        assert test_path.parent.exists()

def test_dataset_download_no_local_dir(hf_hub):
    """Test dataset download without local directory (uses cache)."""
    with patch('datasets.load_dataset') as mock_load:
        mock_load.return_value = Mock()
        
        with patch('lola.utils.config.get_config', return_value={"hf_hub_enabled": True}):
            cache_dir = tempfile.mkdtemp()
            try:
                downloaded_path = hf_hub.download_dataset(
                    repo_id="cache-test-dataset",
                    cache_dir=cache_dir
                )
                
                # Should use cache directory structure
                assert "cache" in str(downloaded_path)
                assert "datasets" in str(downloaded_path)
            finally:
                os.rmdir(cache_dir)

def test_model_card_integration_with_wandb(hf_hub, tmpdir):
    """Test model card generation integrates with W&B artifact logging."""
    model_path = str(tmpdir.mkdir("wandb_model"))
    config_file = Path(model_path) / "config.json"
    config_file.write_text('{"test": "data"}', encoding="utf-8")
    
    with patch('wandb.init') as mock_init:
        mock_run = Mock()
        mock_init.return_value = mock_run
        
        with patch('lola.libs.wandb.logger.get_wandb_logger') as mock_wandb_get:
            mock_wandb_logger = Mock()
            mock_wandb_get.return_value = mock_wandb_logger
            
            with patch.object(mock_wandb_logger, 'start_run') as mock_start_run:
                mock_context = Mock()
                mock_start_run.return_value.__enter__.return_value = mock_context
                
                with hf_hub.start_run(name="integration-test"):
                    # This should trigger W&B integration in publish_model
                    with patch.object(hf_hub, 'publish_model') as mock_publish:
                        mock_publish.return_value = {"repo_id": "test-repo"}
                        repo_info = hf_hub.publish_model(model_path)
                
                # Verify W&B context manager was used
                mock_start_run.assert_called_once()
                mock_context.__exit__.assert_called_once()

def test_error_handling_model_publish(hf_hub, tmpdir):
    """Test error handling during model publishing."""
    non_existent_path = str(tmpdir / "nonexistent_model")
    
    with pytest.raises(FileNotFoundError, match="Model path not found"):
        hf_hub.publish_model(non_existent_path)

def test_error_handling_dataset_publish(hf_hub, tmpdir):
    """Test error handling during dataset publishing."""
    non_existent_path = str(tmpdir / "nonexistent_dataset")
    
    with pytest.raises(FileNotFoundError, match="Dataset path not found"):
        hf_hub.publish_dataset(non_existent_path)

@patch('huggingface_hub.snapshot_download')
def test_model_download_error_handling(mock_download, hf_hub):
    """Test model download error handling."""
    mock_download.side_effect = Exception("Download failed")
    
    with pytest.raises(Exception, match="model download"):
        hf_hub.download_model("error-model")
    
    # Verify Prometheus error recording
    with patch('lola.libs.prometheus.exporter.get_lola_prometheus') as mock_prometheus:
        mock_exporter = Mock()
        mock_prometheus.return_value = mock_exporter
        
        with pytest.raises(Exception):
            hf_hub.download_model("error-model")
        
        mock_exporter.record_model_download.assert_called_once_with(
            model_repo="error-model",
            revision="main",
            size_mb=0,
            success=False
        )

# Performance test for large model upload (mocked)
@pytest.mark.performance
def test_large_model_upload_performance(hf_hub):
    """Test performance of uploading large model directory (mocked)."""
    large_dir = Path(tempfile.mkdtemp()) / "large_model"
    large_dir.mkdir(exist_ok=True)
    
    # Create multiple large files
    for i in range(10):
        file_path = large_dir / f"weights_{i:03d}.bin"
        file_path.write_bytes(os.urandom(1024 * 1024))  # 1MB files
    
    start_time = time.time()
    
    with patch('huggingface_hub.HfApi.upload_folder') as mock_upload:
        mock_upload.return_value = {"repo_id": "large-model-test"}
        
        repo_info = hf_hub.publish_model(str(large_dir))
    
    duration = time.time() - start_time
    total_size_mb = sum(f.stat().st_size for f in large_dir.rglob("*") if f.is_file()) / 1024 / 1024
    
    print(f"Uploaded {total_size_mb:.1f}MB model in {duration:.2f}s ({total_size_mb/duration:.1f}MB/s)")
    
    # Cleanup
    shutil.rmtree(large_dir.parent)

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(hf_hub):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_huggingface_hub.py -v --cov=lola/libs/huggingface --cov-report=html

# Standard imports
import typing as tp
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import os
import json
import hashlib
from contextlib import contextmanager
import time
from enum import Enum

# Third-party
try:
    from huggingface_hub import (
        HfApi, Repository, ModelCard, HfFolder, upload_folder,
        login, snapshot_download, delete_repo, create_repo
    )
    from datasets import Dataset, DatasetDict
    import wandb
except ImportError:
    raise ImportError("Hugging Face Hub dependencies missing. Run 'poetry add huggingface_hub datasets wandb'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.libs.wandb.logger import get_wandb_logger
from lola.libs.prometheus.exporter import get_lola_prometheus
from sentry_sdk import capture_exception

"""
File: Hugging Face Hub integration for LOLA OS model sharing.
Purpose: Provides seamless model and dataset publishing to Hugging Face Hub 
         with automatic versioning, metadata generation, and dataset card creation.
How: Wraps HuggingFace Hub Python client with LOLA-specific utilities for 
     model card generation, dataset conversion, and repository management; 
     integrates with W&B for experiment-to-model promotion.
Why: Enables community model sharing and discovery while maintaining 
     LOLA's open-source philosophy and developer sovereignty through 
     automated publishing workflows.
Full Path: lola-os/python/lola/libs/huggingface/hub.py
"""

class ModelType(Enum):
    """Supported model types for HF Hub."""
    BASE = "base_model"
    FINETUNED = "fine-tuned"
    QUANTIZED = "quantized"
    MERGED = "merged"

class LolaHuggingFaceHub:
    """LolaHuggingFaceHub: Model and dataset sharing integration.
    Does NOT require authentication for reading—write operations need token."""

    DEFAULT_REPO_TYPE = "model"
    SUPPORTED_DATASET_FORMATS = [".jsonl", ".csv", ".parquet", ".json"]

    def __init__(self):
        """
        Initializes HF Hub integration with LOLA configuration.
        Does Not: Authenticate—lazy auth on write operations.
        """
        config = get_config()
        self.enabled = config.get("hf_hub_enabled", True)
        self.token = config.get("hf_token")
        self.organization = config.get("hf_organization", None)
        self.default_visibility = config.get("hf_default_visibility", "public")
        self.auto_create_card = config.get("hf_auto_card", True)
        self.templates_dir = config.get("hf_templates_dir", "./hf_templates")
        
        # Ensure templates directory
        Path(self.templates_dir).mkdir(exist_ok=True)
        
        # Integrations
        self.wandb_logger = get_wandb_logger()
        self.prometheus = get_lola_prometheus()
        self.sentry_dsn = config.get("sentry_dsn")
        
        # API client
        self.api = HfApi()
        
        if self.enabled and self.token:
            # Authenticate for write operations
            login(token=self.token)
            logger.info("Hugging Face Hub authenticated")
        else:
            logger.info("Hugging Face Hub read-only mode (no token)")

    def publish_model(
        self,
        model_path: str,
        repo_id: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        model_card_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        create_repo: bool = True,
        commit_message: Optional[str] = None,
        tags: Optional[List[str]] = None,
        private: bool = False
    ) -> Dict[str, Any]:
        """
        Publishes model to Hugging Face Hub.
        Args:
            model_path: Local path to model directory.
            repo_id: Repository ID (auto-generated if None).
            model_type: Model type for metadata.
            model_card_content: Custom model card content.
            metadata: Additional metadata.
            create_repo: Create repository if it doesn't exist.
            commit_message: Custom commit message.
            tags: Repository tags.
            private: Make repository private.
        Returns:
            Repository information dictionary.
        """
        if not self.enabled:
            raise RuntimeError("Hugging Face Hub integration disabled")

        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            # Generate repo_id if not provided
            if not repo_id:
                repo_name = f"lola-{model_path.name}-{int(time.time())}"
                repo_id = f"{self.organization}/{repo_name}" if self.organization else repo_name

            # Create repository if needed
            if create_repo:
                try:
                    self.api.create_repo(
                        repo_id=repo_id,
                        repo_type="model",
                        private=private,
                        exist_ok=True
                    )
                except Exception as create_exc:
                    if "already exists" not in str(create_exc):
                        raise

            # Generate model card
            if self.auto_create_card or not model_card_content:
                model_card_content = self._generate_model_card(
                    model_type or ModelType.FINETUNED,
                    model_path,
                    metadata or {}
                )

            # Create model card file
            card_path = model_path / "README.md"
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)

            # Prepare tags
            all_tags = ["lola-os", "v1.0", f"model-type:{model_type.value}"] + (tags or [])
            
            # Upload model
            upload_info = self.api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message or f"Upload {model_type.value} model",
                tags=all_tags,
                create_repo=False  # Already created
            )

            # Log to W&B if active
            if self.wandb_logger.enabled:
                with self.wandb_logger.start_run() as run:
                    self.wandb_logger.log_model_artifact(
                        str(model_path),
                        run_id=run.id,
                        name=f"{repo_id.split('/')[-1]}-v1"
                    )

            # Record metrics
            model_size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1024 / 1024
            self.prometheus.record_model_publish(
                model_type=model_type.value if model_type else "unknown",
                repo_id=repo_id,
                size_mb=model_size_mb,
                success=True
            )

            repo_info = {
                "repo_id": repo_id,
                "model_type": model_type.value if model_type else "unknown",
                "upload_info": upload_info,
                "model_path": str(model_path),
                "card_path": str(card_path),
                "model_size_mb": model_size_mb,
                "timestamp": time.time()
            }

            logger.info(f"Model published to {repo_id} ({model_size_mb:.1f}MB)")
            return repo_info

        except Exception as exc:
            self._handle_error(exc, f"model publish {repo_id}")
            self.prometheus.record_model_publish(
                model_type=model_type.value if model_type else "unknown",
                repo_id=repo_id or "unknown",
                success=False
            )
            raise

    def publish_dataset(
        self,
        dataset_path: str,
        repo_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        create_repo: bool = True,
        tags: Optional[List[str]] = None,
        private: bool = False
    ) -> Dict[str, Any]:
        """
        Publishes dataset to Hugging Face Hub.
        Args:
            dataset_path: Path to dataset file/directory.
            repo_id: Repository ID.
            dataset_name: Dataset name in repo.
            description: Dataset description.
            metadata: Additional metadata.
            create_repo: Create repository if needed.
            tags: Repository tags.
            private: Make repository private.
        Returns:
            Dataset repository information.
        """
        if not self.enabled:
            raise RuntimeError("Hugging Face Hub integration disabled")

        try:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

            # Generate repo_id
            if not repo_id:
                name_part = dataset_name or dataset_path.name
                repo_name = f"lola-dataset-{name_part}-{int(time.time())}"
                repo_id = f"{self.organization}/{repo_name}" if self.organization else repo_name

            # Create repository
            if create_repo:
                self.api.create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=private,
                    exist_ok=True
                )

            # Generate dataset card
            if not description:
                description = self._generate_dataset_card(dataset_path, metadata or {})

            # Create dataset card
            card_path = dataset_path / "README.md" if dataset_path.is_dir() else dataset_path.parent / "README.md"
            card_path.parent.mkdir(exist_ok=True)
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(description)

            # Prepare tags
            all_tags = ["lola-os", "v1.0", "dataset"] + (tags or [])

            # Upload dataset
            upload_info = self.api.upload_folder(
                folder_path=str(dataset_path.parent if dataset_path.is_file() else dataset_path),
                path_in_repo=dataset_name or dataset_path.name if dataset_path.is_file() else "",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {dataset_name or dataset_path.name} dataset",
                tags=all_tags,
                create_repo=False
            )

            # Calculate dataset size
            total_size = sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file())
            dataset_size_mb = total_size / 1024 / 1024

            # Log dataset info
            dataset_info = {
                "repo_id": repo_id,
                "dataset_name": dataset_name or dataset_path.name,
                "num_files": len(list(dataset_path.rglob("*") if dataset_path.is_dir() else [dataset_path])),
                "total_size_mb": dataset_size_mb,
                "upload_info": upload_info,
                "timestamp": time.time()
            }

            # Integrate with W&B
            if self.wandb_logger.enabled:
                with self.wandb_logger.start_run() as run:
                    self.wandb_logger.log_dataset_artifact(
                        str(dataset_path),
                        run_id=run.id,
                        name=f"{repo_id.split('/')[-1]}-v1",
                        description=description
                    )

            self.prometheus.record_dataset_publish(
                dataset_type="training" if "train" in str(dataset_path).lower() else "evaluation",
                repo_id=repo_id,
                size_mb=dataset_size_mb,
                success=True
            )

            logger.info(f"Dataset published to {repo_id} ({dataset_size_mb:.1f}MB)")
            return dataset_info

        except Exception as exc:
            self._handle_error(exc, f"dataset publish {repo_id}")
            self.prometheus.record_dataset_publish(
                dataset_type="unknown",
                repo_id=repo_id or "unknown",
                success=False
            )
            raise

    def download_model(self, repo_id: str, local_dir: Optional[str] = None, 
                     revision: Optional[str] = None, 
                     cache_dir: Optional[str] = None) -> Path:
        """
        Downloads model from Hugging Face Hub.
        Args:
            repo_id: Repository ID.
            local_dir: Local directory to download to.
            revision: Specific revision/branch.
            cache_dir: Cache directory.
        Returns:
            Path to downloaded model directory.
        """
        if not self.enabled:
            raise RuntimeError("Hugging Face Hub integration disabled")

        try:
            download_path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                revision=revision,
                cache_dir=cache_dir,
                local_dir_use_symlinks=False
            )

            # Calculate size
            total_size = sum(f.stat().st_size for f in Path(download_path).rglob("*") if f.is_file())
            model_size_mb = total_size / 1024 / 1024

            self.prometheus.record_model_download(
                model_repo=repo_id,
                revision=revision or "main",
                size_mb=model_size_mb,
                success=True
            )

            logger.info(f"Model downloaded: {repo_id} -> {download_path} ({model_size_mb:.1f}MB)")
            return Path(download_path)

        except Exception as exc:
            self._handle_error(exc, f"model download {repo_id}")
            self.prometheus.record_model_download(
                model_repo=repo_id,
                revision=revision or "main",
                success=False
            )
            raise

    def download_dataset(self, repo_id: str, local_dir: Optional[str] = None,
                       revision: Optional[str] = None, 
                       cache_dir: Optional[str] = None) -> Path:
        """
        Downloads dataset from Hugging Face Hub.
        Args:
            repo_id: Dataset repository ID.
            local_dir: Local directory.
            revision: Revision/branch.
            cache_dir: Cache directory.
        Returns:
            Path to downloaded dataset.
        """
        if not self.enabled:
            raise RuntimeError("Hugging Face Hub integration disabled")

        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                download_mode="force_redownload" if local_dir else "reuse_dataset_if_exists"
            )

            # Save to local directory if specified
            if local_dir:
                download_path = Path(local_dir)
                download_path.mkdir(exist_ok=True)
                
                # Save splits
                for split_name, split_data in dataset.items():
                    split_path = download_path / split_name
                    split_path.mkdir(exist_ok=True)
                    
                    # Save as JSONL
                    split_data.to_json(split_path / "data.jsonl", orient='records', lines=True)
                
                download_path = download_path
            else:
                # Use cache directory
                cache_path = Path(cache_dir or HfFolder().cache_dir) / "datasets" 
                repo_path = cache_path / repo_id.replace("/", "_")
                download_path = repo_path

            # Estimate size (rough calculation)
            dataset_size_mb = 0
            for split_path in download_path.rglob("*.jsonl"):
                if split_path.is_file():
                    dataset_size_mb += split_path.stat().st_size / 1024 / 1024

            self.prometheus.record_dataset_download(
                dataset_repo=repo_id,
                revision=revision or "main",
                size_mb=dataset_size_mb,
                success=True
            )

            logger.info(f"Dataset downloaded: {repo_id} -> {download_path} (~{dataset_size_mb:.1f}MB)")
            return download_path

        except Exception as exc:
            self._handle_error(exc, f"dataset download {repo_id}")
            self.prometheus.record_dataset_download(
                dataset_repo=repo_id,
                revision=revision or "main",
                success=False
            )
            raise

    def create_model_card(
        self,
        model_type: ModelType,
        base_model: Optional[str] = None,
        fine_tune_config: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        usage_examples: Optional[List[str]] = None,
        limitations: Optional[str] = None,
        ethical_considerations: Optional[str] = None
    ) -> str:
        """
        Generates comprehensive model card content.
        Args:
            model_type: Type of model.
            base_model: Base model name.
            fine_tune_config: Fine-tuning configuration.
            performance_metrics: Evaluation results.
            usage_examples: Code examples.
            limitations: Known limitations.
            ethical_considerations: Ethical notes.
        Returns:
            Model card markdown content.
        """
        try:
            # Load template
            template_path = Path(self.templates_dir) / "model_card.md.j2"
            if not template_path.exists():
                template_content = self._get_default_model_card_template()
                template_path.parent.mkdir(exist_ok=True)
                with open(template_path, "w") as f:
                    f.write(template_content)

            with open(template_path, "r") as f:
                from jinja2 import Template
                template = Template(f.read())

            # Render template
            model_card = template.render({
                "model_type": model_type.value,
                "base_model": base_model or "Unknown",
                "lola_version": "1.0.0",
                "fine_tune_config": json.dumps(fine_tune_config or {}, indent=2),
                "performance_metrics": json.dumps(performance_metrics or {}, indent=2),
                "usage_examples": usage_examples or ["```python\n# Example usage\nagent = ReActAgent(model='your-model')\n```"],
                "limitations": limitations or "No specific limitations documented.",
                "ethical_considerations": ethical_considerations or "Follow responsible AI guidelines.",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC")
            })

            return model_card

        except Exception as exc:
            self._handle_error(exc, "model card generation")
            # Fallback to simple card
            return f"""# LOLA Model

**Type:** {model_type.value}
**Generated:** {time.strftime('%Y-%m-%d')}

No detailed card available.

---
*This model card was auto-generated by LOLA OS.*
"""

    def _get_default_model_card_template(self) -> str:
        """Returns default model card template."""
        return """---
language: en
license: apache-2.0
library_name: transformers
tags:
- lola-os
- {{ model_type }}
- pytorch
base_model: {{ base_model }}
---

# LOLA {{ model_type.title() }} Model

Generated by LOLA OS Model Garden on {{ timestamp }}.

## Model Details

**Base Model:** {{ base_model }}
**LOLA Version:** {{ lola_version }}
**Model Type:** {{ model_type }}

## Fine-tuning Configuration

```json
{{ fine_tune_config }}
```

## Performance Metrics

```json
{{ performance_metrics }}
```

## Usage

### Installation

```bash
pip install transformers torch
```

### Example Usage

{{ usage_examples | join('\\n\\n') }}

## LOLA Integration

This model is optimized for use with LOLA OS agents:

```python
from lola.agents import ReActAgent

agent = ReActAgent(
    model="{{ base_model }}",  # Use your fine-tuned model
    tools=[web_search_tool, calculator_tool]
)

response = agent.run("Your query here")
```

## Limitations

{{ limitations }}

## Ethical Considerations

{{ ethical_considerations }}

---
*This model card was auto-generated by LOLA OS.*
"""

    def _generate_dataset_card(self, dataset_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generates dataset card content.
        """
        try:
            # Count examples
            if dataset_path.suffix == '.jsonl':
                with open(dataset_path) as f:
                    num_examples = sum(1 for _ in f)
            else:
                num_examples = "Unknown"

            dataset_size_mb = dataset_path.stat().st_size / 1024 / 1024 if dataset_path.is_file() else 0

            return f"""---
language: en
license: apache-2.0
size_categories:
  - 10K<n<100K
task_categories:
  - text-classification
  - text-generation
tags:
- lola-os
- training-data
- synthetic

---

# LOLA Dataset

**Examples:** {num_examples}
**Format:** {dataset_path.suffix}
**Size:** {dataset_size_mb:.1f}MB
**Generated:** {time.strftime('%Y-%m-%d')}

## Dataset Details

Generated by LOLA OS for training AI agents.

### Metadata

```json
{json.dumps(metadata, indent=2)}
```

---
*This dataset card was auto-generated by LOLA OS.*
"""

        except Exception as exc:
            logger.warning(f"Dataset card generation failed: {str(exc)}")
            return f"# LOLA Dataset\n\nDataset from {dataset_path.name}\n\n{json.dumps(metadata, indent=2)}"

    def _handle_error(self, exc: Exception, context: str) -> None:
        """
        Error handling for HF Hub operations.
        """
        logger.error(f"HF Hub {context}: {str(exc)}")
        if self.sentry_dsn:
            capture_exception(exc)

        self.prometheus.record_hf_operation(
            operation=context,
            success=False
        )


# Global hub instance
_lola_hf_hub = None

def get_huggingface_hub() -> LolaHuggingFaceHub:
    """Singleton Hugging Face Hub instance."""
    global _lola_hf_hub
    if _lola_hf_hub is None:
        _lola_hf_hub = LolaHuggingFaceHub()
    return _lola_hf_hub

__all__ = [
    "ModelType",
    "LolaHuggingFaceHub",
    "get_huggingface_hub"
]




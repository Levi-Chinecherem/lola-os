# Standard imports
import typing as tp
from typing import Dict, Any, List, Optional, Union
import os
import json
from pathlib import Path
import time
from contextlib import contextmanager
import threading
import sys
from unittest.mock import Mock
from enum import Enum

# Third-party
try:
    import wandb
    from wandb.sdk.data_types.trace_tree import TraceTree, TraceEvent
    from wandb.sdk.wandb_run import Run
except ImportError:
    raise ImportError("W&B not installed. Run 'poetry add wandb'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.core.agent import BaseAgent
from lola.core.graph import StateGraph
from lola.libs.prometheus.exporter import get_lola_prometheus
from sentry_sdk import capture_exception

"""
File: Weights & Biases integration for LOLA OS experiment tracking.
Purpose: Provides comprehensive experiment tracking for model training, agent 
         performance evaluation, and A/B testing with automatic metric logging, 
         artifact management, and sweep integration.
How: Wraps W&B Python SDK with LOLA-specific integrations for agent runs, 
     training jobs, evaluation results, and model artifacts; supports both 
     local and cloud tracking modes.
Why: Enables systematic improvement of LOLA agents through data-driven 
     experimentation, hyperparameter tuning, and model version control while 
     maintaining the open-source philosophy with optional cloud dependency.
Full Path: lola-os/python/lola/libs/wandb/logger.py
"""

class ExperimentMode(Enum):
    """W&B experiment modes."""
    DISABLED = "disabled"
    ONLINE = "online" 
    OFFLINE = "offline"
    DRYRUN = "dryrun"

class LolaWandBLogger:
    """LolaWandBLogger: W&B integration for LOLA OS experiments.
    Does NOT initialize W&B globally—lazy initialization per run."""

    def __init__(self):
        """
        Initializes W&B logger with LOLA configuration.
        Does Not: Start runs—explicit run management required.
        """
        config = get_config()
        self.enabled = config.get("wandb_enabled", False)
        self.api_key = config.get("wandb_api_key")
        self.project_name = config.get("wandb_project", "lola-experiments")
        self.entity = config.get("wandb_entity", None)
        self.default_tags = config.get("wandb_tags", ["lola-os", "v1.0"])
        self.mode = ExperimentMode(config.get("wandb_mode", "online"))
        self.offline_dir = config.get("wandb_offline_dir", "./wandb_runs")
        self.save_code = config.get("wandb_save_code", True)
        
        # Ensure offline directory
        if self.mode == ExperimentMode.OFFLINE:
            Path(self.offline_dir).mkdir(exist_ok=True)
        
        # Observability integration
        self.prometheus = get_lola_prometheus()
        self.sentry_dsn = config.get("sentry_dsn")
        
        # Run management
        self.active_runs: Dict[str, Run] = {}
        self._lock = threading.RLock()
        
        if self.enabled and self.api_key:
            # Set API key for W&B
            os.environ["WANDB_API_KEY"] = self.api_key
            wandb.require("service")  # Ensure service mode
        
        logger.info(f"W&B logger initialized: {self.project_name} ({self.mode.value})")

    @contextmanager
    def start_run(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None,
        mode: Optional[ExperimentMode] = None
    ):
        """
        Context manager for W&B runs.
        Args:
            name: Run name.
            project: Project name (defaults to configured).
            config: Run configuration dictionary.
            tags: Additional tags.
            group: Experiment group for sweeps.
            notes: Run description.
            mode: Run mode (online/offline/dryrun).
        Yields:
            W&B run instance.
        """
        if not self.enabled:
            # Yield mock run for compatibility
            class MockRun:
                def __init__(self): self._mock_logs = []
                def log(self, data, **kwargs): self._mock_logs.append(data)
                def finish(self): pass
                def save(self, path): pass
                def log_artifact(self, artifact): pass
            yield MockRun()
            return

        mode = mode or self.mode
        
        run_name = name or f"lola-run-{int(time.time())}"
        project = project or self.project_name
        
        try:
            # Start W&B run
            run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=self.default_tags + (tags or []),
                group=group,
                notes=notes,
                mode=mode.value,
                dir=self.offline_dir if mode == ExperimentMode.OFFLINE else None,
                save_code=self.save_code,
                entity=self.entity
            )
            
            # Set LOLA-specific metadata
            run._label_algo = "lola-os"
            run.config.setdefault("lola_version", "1.0.0")
            run.config.setdefault("python_version", f"{sys.version_info.major}.{sys.version_info.minor}")
            
            with self._lock:
                self.active_runs[run.id] = run
            
            logger.info(f"W&B run started: {run_name} in {project} ({mode.value})")
            
            yield run
            
        except Exception as exc:
            self._handle_error(exc, f"run start {run_name}")
            raise
        finally:
            if 'run' in locals():
                run.finish()
                with self._lock:
                    self.active_runs.pop(run.id, None)
                logger.debug(f"W&B run {run_name} finished")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, 
                   run_id: Optional[str] = None) -> None:
        """
        Logs metrics to active or specified run.
        Args:
            metrics: Dictionary of metric name to value.
            step: Optional training step.
            run_id: Specific run ID (uses active run if None).
        """
        if not self.enabled:
            return

        try:
            if run_id and run_id in self.active_runs:
                run = self.active_runs[run_id]
            elif self.active_runs:
                run = list(self.active_runs.values())[0]
            else:
                logger.warning("No active W&B run for logging")
                return
            
            run.log(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics to W&B run {run.id}")
            
        except Exception as exc:
            self._handle_error(exc, "metrics logging")

    def log_agent_performance(self, agent: BaseAgent, metrics: Dict[str, Any], 
                            step: Optional[int] = None) -> None:
        """
        Logs agent performance metrics with LOLA-specific context.
        Args:
            agent: LOLA BaseAgent instance.
            metrics: Performance metrics.
            step: Optional step number.
        """
        if not self.enabled:
            return

        try:
            # Enrich with agent context
            agent_metrics = metrics.copy()
            agent_metrics.update({
                "agent_type": agent.__class__.__name__,
                "model_name": getattr(agent, 'model', 'unknown'),
                "tools_count": len(getattr(agent, 'tools', [])),
                "lola_agent_version": "1.0.0"
            })
            
            self.log_metrics(agent_metrics, step)
            
            # Log to Prometheus for real-time monitoring
            self.prometheus.record_agent_performance(
                agent_type=agent_metrics["agent_type"],
                operation="evaluation",
                duration=metrics.get("duration", 0),
                success_rate=metrics.get("success_rate", 0),
                token_usage=metrics.get("token_usage", 0)
            )
            
        except Exception as exc:
            self._handle_error(exc, f"agent performance {agent.__class__.__name__}")

    def log_training_metrics(self, metrics: Dict[str, Any], step: int, 
                           epoch: Optional[int] = None) -> None:
        """
        Logs training metrics with step and epoch information.
        Args:
            metrics: Training metrics (loss, learning_rate, etc.).
            step: Training step.
            epoch: Current epoch.
        """
        if not self.enabled:
            return

        try:
            training_metrics = metrics.copy()
            training_metrics.update({
                "training_step": step,
                "epoch": epoch,
                "lola_training_batch_size": metrics.get("batch_size", 1)
            })
            
            self.log_metrics(training_metrics, step)
            
            # Log to Prometheus
            self.prometheus.record_training_metrics(
                model=training_metrics.get("model_name", "unknown"),
                loss=training_metrics.get("train_loss", 0),
                learning_rate=training_metrics.get("learning_rate", 0),
                step=step
            )
            
        except Exception as exc:
            self._handle_error(exc, "training metrics")

    def log_model_artifact(self, model_path: str, run_id: Optional[str] = None, 
                         name: Optional[str] = None, description: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> wandb.Artifact:
        """
        Logs model as W&B artifact.
        Args:
            model_path: Path to model directory/files.
            run_id: Specific run ID.
            name: Artifact name.
            description: Artifact description.
            metadata: Additional metadata.
        Returns:
            W&B Artifact instance.
        """
        if not self.enabled:
            return Mock()

        try:
            if run_id and run_id in self.active_runs:
                run = self.active_runs[run_id]
            elif self.active_runs:
                run = list(self.active_runs.values())[0]
            else:
                raise RuntimeError("No active W&B run for artifact logging")

            # Create artifact
            artifact_name = name or f"lola-model-{int(time.time())}"
            artifact = wandb.Artifact(artifact_name, type="model")
            
            # Add model files
            model_path = Path(model_path)
            if model_path.is_dir():
                for file_path in model_path.rglob("*"):
                    if file_path.is_file():
                        with open(file_path, "rb") as f:
                            artifact.add_file(file_path.name, f)
            else:
                with open(model_path, "rb") as f:
                    artifact.add_file(model_path.name, f)
            
            # Add metadata
            if metadata:
                artifact.metadata = metadata
            
            # Log description
            if description:
                artifact.add_code(f"model_description.md", description)
            
            # Log to run
            run.log_artifact(artifact)
            artifact.wait()
            
            logger.info(f"Model artifact logged: {artifact_name} (version: {artifact.version})")
            return artifact
            
        except Exception as exc:
            self._handle_error(exc, "model artifact")
            raise

    def log_dataset_artifact(self, dataset_path: str, run_id: Optional[str] = None,
                           name: Optional[str] = None, description: Optional[str] = None) -> wandb.Artifact:
        """
        Logs dataset as W&B artifact.
        Args:
            dataset_path: Path to dataset file/directory.
            run_id: Specific run ID.
            name: Artifact name.
            description: Dataset description.
        Returns:
            W&B Artifact instance.
        """
        if not self.enabled:
            return Mock()

        try:
            if run_id and run_id in self.active_runs:
                run = self.active_runs[run_id]
            elif self.active_runs:
                run = list(self.active_runs.values())[0]
            else:
                raise RuntimeError("No active W&B run for dataset logging")

            # Create dataset artifact
            artifact_name = name or f"lola-dataset-{int(time.time())}"
            artifact = wandb.Artifact(artifact_name, type="dataset")
            
            dataset_path = Path(dataset_path)
            if dataset_path.is_dir():
                # Add all files in directory
                for file_path in dataset_path.rglob("*"):
                    if file_path.is_file():
                        with open(file_path, "rb") as f:
                            # Preserve directory structure
                            rel_path = file_path.relative_to(dataset_path)
                            artifact.add_file(str(rel_path), f)
            else:
                # Single file
                with open(dataset_path, "rb") as f:
                    artifact.add_file(dataset_path.name, f)
            
            # Add description
            if description:
                artifact.add_code("README.md", description)
            
            # Log dataset info
            if dataset_path.suffix == '.jsonl':
                try:
                    with open(dataset_path) as f:
                        lines = f.readlines()
                        artifact.metadata = {
                            "num_examples": len(lines),
                            "format": "jsonl",
                            "file_size_bytes": os.path.getsize(dataset_path)
                        }
                except Exception:
                    pass
            
            run.log_artifact(artifact)
            artifact.wait()
            
            logger.info(f"Dataset artifact logged: {artifact_name}")
            return artifact
            
        except Exception as exc:
            self._handle_error(exc, "dataset artifact")
            raise

    def start_sweep(self, sweep_config: Dict[str, Any], project: Optional[str] = None) -> str:
        """
        Starts W&B sweep for hyperparameter optimization.
        Args:
            sweep_config: Sweep configuration dictionary.
            project: Optional project name.
        Returns:
            Sweep ID.
        """
        if not self.enabled:
            return "disabled"

        try:
            project = project or self.project_name
            sweep_id = wandb.sweep(sweep_config, project=project)
            
            logger.info(f"W&B sweep started: {sweep_id} in project {project}")
            return sweep_id
            
        except Exception as exc:
            self._handle_error(exc, "sweep start")
            raise

    def log_sweep_agent(self, sweep_id: str, agent_config: Dict[str, Any]) -> None:
        """
        Logs agent configuration to sweep.
        Args:
            sweep_id: Sweep identifier.
            agent_config: Agent configuration.
        """
        if not self.enabled:
            return

        try:
            api = wandb.Api()
            sweep = api.sweep(sweep_id)
            
            # Log as sweep config
            sweep_config = sweep.config
            sweep_config.update(agent_config)
            sweep.update_config(sweep_config)
            
            logger.debug(f"Agent config logged to sweep {sweep_id}")
            
        except Exception as exc:
            self._handle_error(exc, "sweep agent logging")

    def _handle_error(self, exc: Exception, context: str) -> None:
        """
        Error handling for W&B operations.
        """
        logger.error(f"W&B {context}: {str(exc)}")
        if self.sentry_dsn:
            capture_exception(exc)

    def get_active_run_id(self) -> Optional[str]:
        """
        Gets ID of currently active run.
        Returns:
            Run ID or None if no active run.
        """
        with self._lock:
            if self.active_runs:
                return list(self.active_runs.keys())[0]
            return None

    def finish_all_runs(self) -> None:
        """Finishes all active runs."""
        with self._lock:
            for run_id, run in list(self.active_runs.items()):
                run.finish()
                del self.active_runs[run_id]
                logger.info(f"Finished W&B run: {run_id}")


# Global logger instance
_lola_wandb_logger = None

def get_wandb_logger() -> LolaWandBLogger:
    """Singleton W&B logger instance."""
    global _lola_wandb_logger
    if _lola_wandb_logger is None:
        _lola_wandb_logger = LolaWandBLogger()
    return _lola_wandb_logger

__all__ = [
    "ExperimentMode",
    "LolaWandBLogger",
    "get_wandb_logger"
]
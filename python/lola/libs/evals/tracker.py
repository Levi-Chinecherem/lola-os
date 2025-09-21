# Standard imports
import typing as tp
import logging

# Third-party
import wandb
import mlflow
from tenacity import retry, stop_after_attempt, wait_exponential

# Local
from lola.utils import sentry

"""
File: Implements evaluation trackers for LOLA OS TMVP 1 Phase 5.

Purpose: Provides flexible backends for logging evaluation metrics.
How: Abstracts W&B and MLflow tracking.
Why: Ensures adaptable evaluation tracking, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/libs/evals/tracker.py
"""

logger = logging.getLogger(__name__)

class EvalTracker:
    """EvalTracker: Abstract base for evaluation metric tracking."""

    @abstractmethod
    def log_metrics(self, agent_name: str, metrics: tp.Dict[str, float]) -> None:
        """
        Logs metrics to tracking backend.

        Args:
            agent_name: Name of the agent.
            metrics: Dict of metric names to values.

        Does Not: Persist metrics—use StateManager.
        """
        pass

class WandbTracker(EvalTracker):
    """WandbTracker: Logs metrics to Weights & Biases."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with W&B project details.

        Args:
            config: Dict with 'project' (W&B project name) and optional 'api_key'.

        Does Not: Validate project—assume pre-configured.
        """
        try:
            self.project = config.get("project", "lola-evals")
            api_key = config.get("api_key")
            if api_key:
                wandb.login(key=api_key)
            wandb.init(project=self.project, reinit=True)
            logger.info("Initialized WandbTracker")
        except Exception as e:
            logger.error(f"WandbTracker initialization failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def log_metrics(self, agent_name: str, metrics: tp.Dict[str, float]) -> None:
        """
        Logs metrics to W&B.

        Args:
            agent_name: Name of the agent.
            metrics: Dict of metric names to values.
        """
        try:
            wandb.log({f"{agent_name}/{k}": v for k, v in metrics.items()})
            logger.debug(f"Logged metrics to W&B for {agent_name}")
        except Exception as e:
            logger.error(f"W&B logging failed: {e}")
            sentry.capture_exception(e)
            raise

class MLflowTracker(EvalTracker):
    """MLflowTracker: Logs metrics to MLflow."""

    def __init__(self, config: tp.Dict[str, tp.Any]):
        """
        Initialize with MLflow experiment details.

        Args:
            config: Dict with 'experiment_name' (MLflow experiment).

        Does Not: Validate experiment—assume pre-configured.
        """
        try:
            self.experiment_name = config.get("experiment_name", "lola-evals")
            mlflow.set_experiment(self.experiment_name)
            logger.info("Initialized MLflowTracker")
        except Exception as e:
            logger.error(f"MLflowTracker initialization failed: {e}")
            sentry.capture_exception(e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def log_metrics(self, agent_name: str, metrics: tp.Dict[str, float]) -> None:
        """
        Logs metrics to MLflow.

        Args:
            agent_name: Name of the agent.
            metrics: Dict of metric names to values.
        """
        try:
            with mlflow.start_run(run_name=agent_name):
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
            logger.debug(f"Logged metrics to MLflow for {agent_name}")
        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
            sentry.capture_exception(e)
            raise

__all__ = ["EvalTracker", "WandbTracker", "MLflowTracker"]
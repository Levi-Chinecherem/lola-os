# Standard imports
import typing as tp
from typing import Dict, Any, List, Optional, Callable
from celery import Celery, Task
from celery.utils.log import get_task_logger
from celery.signals import worker_process_init, task_prerun, task_postrun
import json
from pathlib import Path
import time
from contextlib import contextmanager
import os

# Third-party
try:
    from celery.result import AsyncResult
    from celery.exceptions import WorkerLostError, SoftTimeLimitExceeded
    import kombu
except ImportError:
    raise ImportError("Celery dependencies missing. Run 'poetry add celery redis'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.libs.prometheus.exporter import get_lola_prometheus
from lola.libs.sentry.sdk import get_lola_sentry
from lola.core.agent import BaseAgent
from lola.core.graph import StateGraph
from sentry_sdk import capture_exception, start_transaction

"""
File: Celery distributed task queue integration for LOLA OS.
Purpose: Provides asynchronous task execution for long-running agent operations, 
         model training jobs, data processing pipelines, and distributed multi-agent 
         orchestration with comprehensive monitoring and error handling.
How: Configures Celery with Redis broker; defines task patterns for LOLA components 
     (agents, graphs, RAG indexing); integrates with Prometheus/Sentry for observability; 
     supports task retries, time limits, and result tracking.
Why: Enables scalable, production-grade distributed execution for LOLA agents 
     while maintaining the framework's developer sovereignty through 
     configuration-driven task management.
Full Path: lola-os/python/lola/libs/celery/tasks.py
"""

# Global Celery app instance
_lola_celery_app = None

def get_celery_app() -> Celery:
    """
    Returns the global LOLA Celery application instance.
    Returns:
        Configured Celery app.
    """
    global _lola_celery_app
    if _lola_celery_app is None:
        _lola_celery_app = _create_celery_app()
    return _lola_celery_app

def _create_celery_app() -> Celery:
    """Creates and configures LOLA Celery application."""
    config = get_config()
    
    # Celery configuration
    celery_config = config.get("celery", {})
    broker_url = celery_config.get("broker_url", "redis://localhost:6379/0")
    result_backend = celery_config.get("result_backend", broker_url)
    include = celery_config.get("include", ["lola.libs.celery.tasks"])
    
    # Create app
    app = Celery(
        "lola_os",
        broker=broker_url,
        backend=result_backend,
        include=include
    )
    
    # Load configuration
    app.conf.update(
        # Task queue configuration
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        
        # Worker configuration
        worker_prefetch_multiplier=1,  # Conservative prefetching
        worker_max_tasks_per_child=config.get("celery_max_tasks_per_child", 1000),
        worker_concurrency=config.get("celery_worker_concurrency", 4),
        
        # Task time limits
        task_time_limit=config.get("celery_task_time_limit", 3600),  # 1 hour
        task_soft_time_limit=config.get("celery_task_soft_time_limit", 3000),  # 50 min
        
        # Retry configuration
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        
        # Security
        task_send_sent_event=True,
        task_ignore_result=False,
        
        # LOLA-specific
        lola_namespace=config.get("lola_namespace", "lola-os"),
        lola_version=config.get("lola_version", "1.0.0"),
        lola_environment=config.get("environment", "development")
    )
    
    # Add LOLA-specific signals
    worker_process_init.connect(_lola_worker_init)
    task_prerun.connect(_lola_task_prerun)
    task_postrun.connect(_lola_task_postrun)
    
    logger.info(f"Celery app created: broker={broker_url}, backend={result_backend}")
    return app

def _lola_worker_init(sender=None, **kwargs):
    """Initialize worker process with LOLA context."""
    try:
        # Initialize LOLA components in worker
        from lola.utils.config import load_config
        from lola.libs.prometheus.exporter import get_lola_prometheus
        from lola.libs.sentry.sdk import get_lola_sentry
        
        load_config()  # Reload config in worker
        get_lola_prometheus()  # Initialize metrics
        get_lola_sentry()  # Initialize error tracking
        
        logger.info("LOLA worker process initialized")
        
    except Exception as exc:
        logger.error(f"Worker initialization failed: {str(exc)}")
        if get_config().get("sentry_dsn"):
            capture_exception(exc)

def _lola_task_prerun(sender=None, task=None, **kwargs):
    """Pre-run task hook for observability."""
    if task:
        try:
            # Start Sentry transaction
            sentry = get_lola_sentry()
            if sentry.is_enabled():
                sentry.start_agent_transaction(task, task.name)
            
            # Start Prometheus timing
            prometheus = get_lola_prometheus()
            if prometheus.enabled:
                prometheus.start_task_timing(task.name)
                
        except Exception as exc:
            logger.debug(f"Task prerun setup failed: {str(exc)}")

def _lola_task_postrun(sender=None, task=None, retval=None, state=None, **kwargs):
    """Post-run task hook for observability."""
    if task:
        try:
            # Record metrics
            prometheus = get_lola_prometheus()
            if prometheus.enabled:
                prometheus.record_task_completion(
                    task_name=task.name,
                    duration=prometheus.get_task_duration(task.name),
                    success=state == "SUCCESS",
                    result_size=len(json.dumps(retval)) if retval else 0
                )
            
            # Finish Sentry transaction
            sentry = get_lola_sentry()
            if sentry.is_enabled():
                sentry.finish_task_transaction(task.name, state == "SUCCESS")
                
        except Exception as exc:
            logger.debug(f"Task postrun cleanup failed: {str(exc)}")

class LolaTask(Task):
    """Base task class with LOLA-specific enhancements."""
    
    abstract = True
    
    def __init__(self):
        super().__init__()
        self._sentry_transaction = None
        self._prometheus_timer = None
    
    def on_failure(self, exc, traceback, einfo=None, **kwargs):
        """Enhanced failure handling."""
        super().on_failure(exc, traceback, einfo, **kwargs)
        
        try:
            # Enhanced error reporting
            error_context = {
                "task_name": self.name,
                "task_id": self.request.id,
                "args": self.request.args,
                "kwargs": self.request.kwargs,
                "worker_hostname": self.request.hostname
            }
            
            # Sentry
            sentry = get_lola_sentry()
            if sentry.is_enabled():
                sentry.capture_exception(exc, context=error_context)
            
            # Prometheus
            prometheus = get_lola_prometheus()
            if prometheus.enabled:
                prometheus.record_task_error(
                    task_name=self.name,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:200]
                )
                
        except Exception as log_exc:
            logger.error(f"Enhanced error logging failed: {str(log_exc)}")

# Agent execution tasks
@celery_app.task(bind=True, base=LolaTask, time_limit=3600, soft_time_limit=3000)
def execute_agent_task(self, agent_config: Dict[str, Any], input_data: Dict[str, Any], 
                      task_id: str) -> Dict[str, Any]:
    """
    Executes LOLA agent asynchronously.
    Args:
        agent_config: Agent configuration dictionary.
        input_data: Input data for agent.
        task_id: Unique task identifier.
    Returns:
        Agent execution results.
    """
    try:
        # Deserialize agent
        agent_type = agent_config.pop("agent_type")
        agent_module = importlib.import_module(f"lola.agents.{agent_type.lower()}")
        agent_class = getattr(agent_module, agent_type)
        
        agent = agent_class(**agent_config)
        
        # Execute agent
        result = agent.run(input_data.get("query", ""))
        
        # Prepare result
        task_result = {
            "task_id": task_id,
            "status": "completed",
            "result": result.dict() if hasattr(result, 'dict') else result,
            "execution_time": time.time() - self.request.start,
            "agent_type": agent_type,
            "model_used": agent.model
        }
        
        logger.info(f"Agent task {task_id} completed: {agent_type}")
        return task_result
        
    except Exception as exc:
        logger.error(f"Agent task {task_id} failed: {str(exc)}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)

@celery_app.task(bind=True, base=LolaTask, time_limit=7200, soft_time_limit=6000)
def execute_graph_task(self, graph_config: Dict[str, Any], input_state: Dict[str, Any], 
                      task_id: str) -> Dict[str, Any]:
    """
    Executes LOLA graph workflow asynchronously.
    Args:
        graph_config: Graph configuration.
        input_state: Initial state.
        task_id: Task identifier.
    Returns:
        Final graph state.
    """
    try:
        # Reconstruct graph
        from lola.core.graph import StateGraph
        
        graph = StateGraph.from_config(graph_config)
        initial_state = type('State', (), input_state)()  # Simple object from dict
        
        # Execute graph
        final_state = graph.execute(initial_state)
        
        task_result = {
            "task_id": task_id,
            "status": "completed",
            "final_state": final_state.__dict__,
            "nodes_executed": len(graph.execution_history) if hasattr(graph, 'execution_history') else 0,
            "execution_time": time.time() - self.request.start
        }
        
        logger.info(f"Graph task {task_id} completed: {len(task_result['nodes_executed'])} nodes")
        return task_result
        
    except Exception as exc:
        logger.error(f"Graph task {task_id} failed: {str(exc)}")
        raise self.retry(exc=exc, countdown=120, max_retries=2)

# Model training tasks
@celery_app.task(bind=True, base=LolaTask, time_limit=86400, soft_time_limit=72000)
def fine_tune_model_task(self, training_config: Dict[str, Any], dataset_path: str, 
                        job_id: str) -> Dict[str, Any]:
    """
    Fine-tunes model using Axolotl asynchronously.
    Args:
        training_config: Fine-tuning configuration.
        dataset_path: Path to training dataset.
        job_id: Job identifier.
    Returns:
        Training job results.
    """
    from lola.libs.axolotl.trainer import get_axolotl_trainer
    
    try:
        trainer = get_axolotl_trainer()
        config = FineTuneConfig(**training_config)
        
        # Start fine-tuning
        fine_tune_id = trainer.fine_tune(
            config,
            dataset_path=dataset_path,
            run_name=f"celery-job-{job_id}"
        )
        
        # Wait for completion (simplified - in production would use callbacks)
        while trainer.get_job_status(fine_tune_id)["status"] == "running":
            await asyncio.sleep(30)
        
        job_status = trainer.get_job_status(fine_tune_id)
        
        result = {
            "job_id": job_id,
            "status": "completed" if job_status["status"] == "completed" else "failed",
            "fine_tune_id": fine_tune_id,
            "model_path": job_status.get("output_dir"),
            "metrics": job_status.get("metrics", {}),
            "execution_time": time.time() - self.request.start
        }
        
        logger.info(f"Model fine-tuning task {job_id} completed")
        return result
        
    except Exception as exc:
        logger.error(f"Model fine-tuning task {job_id} failed: {str(exc)}")
        raise self.retry(exc=exc, countdown=300, max_retries=1)

# Data processing tasks
@celery_app.task(bind=True, base=LolaTask, time_limit=1800, soft_time_limit=1500)
def process_rag_documents_task(self, documents: List[Dict[str, Any]], 
                              vector_db_config: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Processes and indexes documents for RAG asynchronously.
    Args:
        documents: List of document dictionaries.
        vector_db_config: VectorDB configuration.
        task_id: Task identifier.
    Returns:
        Indexing results.
    """
    from lola.libs.vector_dbs.adapter import get_vector_db_adapter
    from lola.rag.multimodal import MultiModalRetriever
    
    try:
        # Initialize VectorDB
        vector_db = get_vector_db_adapter(vector_db_config)
        vector_db.connect()
        
        # Extract embeddings (mock for now - integrate with actual embedding service)
        embeddings = []
        processed_docs = []
        
        for doc in documents:
            # Generate embedding (in production, use embedding service)
            embedding = [0.1] * vector_db._embedding_dim  # Mock embedding
            embeddings.append(embedding)
            
            processed_doc = {
                "id": doc.get("id", str(uuid.uuid4())),
                "text": doc["text"][:1000],  # Truncate for storage
                "metadata": doc.get("metadata", {}),
                "embedding": embedding
            }
            processed_docs.append(processed_doc)
        
        # Index documents
        texts = [doc["text"] for doc in processed_docs]
        metadatas = [doc["metadata"] for doc in processed_docs]
        ids = [doc["id"] for doc in processed_docs]
        
        vector_db.index(embeddings, texts, metadatas, ids)
        
        # Record metrics
        self.prometheus.record_rag_indexing(
            documents_count=len(documents),
            indexed_count=len(ids),
            vector_db_type=vector_db_config["type"]
        )
        
        result = {
            "task_id": task_id,
            "status": "completed",
            "processed_count": len(processed_docs),
            "indexed_count": len(ids),
            "vector_db_type": vector_db_config["type"],
            "execution_time": time.time() - self.request.start
        }
        
        logger.info(f"RAG indexing task {task_id} completed: {len(ids)} documents")
        return result
        
    except Exception as exc:
        logger.error(f"RAG indexing task {task_id} failed: {str(exc)}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)

# Multi-agent coordination tasks
@celery_app.task(bind=True, base=LolaTask, time_limit=3600, soft_time_limit=3000)
def coordinate_agent_swarm_task(self, swarm_config: Dict[str, Any], 
                               task_assignment: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Coordinates multi-agent swarm execution.
    Args:
        swarm_config: Swarm configuration.
        task_assignment: Task assignments to agents.
        task_id: Coordination task ID.
    Returns:
        Swarm execution results.
    """
    from lola.orchestration.swarm import AgentSwarmOrchestrator
    
    try:
        # Initialize swarm orchestrator
        orchestrator = AgentSwarmOrchestrator.from_config(swarm_config)
        
        # Execute swarm
        results = orchestrator.execute_swarm(task_assignment)
        
        # Aggregate results
        swarm_result = {
            "task_id": task_id,
            "status": "completed",
            "agents_executed": len(results),
            "coordination_time": time.time() - self.request.start,
            "individual_results": results,
            "swarm_summary": {
                "success_rate": sum(1 for r in results if r.get("success", False)) / len(results),
                "avg_duration": sum(r.get("duration", 0) for r in results) / len(results),
                "total_tokens": sum(r.get("tokens", 0) for r in results)
            }
        }
        
        logger.info(f"Swarm coordination task {task_id} completed: {len(results)} agents")
        return swarm_result
        
    except Exception as exc:
        logger.error(f"Swarm coordination task {task_id} failed: {str(exc)}")
        raise self.retry(exc=exc, countdown=120, max_retries=2)

# Utility functions
def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Gets task status by ID.
    Args:
        task_id: Celery task ID.
    Returns:
        Task status information.
    """
    app = get_celery_app()
    result = AsyncResult(task_id, app=app)
    
    return {
        "task_id": task_id,
        "status": result.status,
        "successful": result.successful(),
        "failed": result.failed(),
        "result": result.result if result.ready() else None,
        "traceback": result.traceback if result.failed() else None,
        "info": result.info() if result.ready() else "Task not ready"
    }

def schedule_agent_task(agent_config: Dict[str, Any], input_data: Dict[str, Any], 
                       queue: Optional[str] = None, countdown: Optional[int] = None,
                       **kwargs) -> str:
    """
    Schedules agent execution task.
    Args:
        agent_config: Agent configuration.
        input_data: Input data.
        queue: Optional task queue.
        countdown: Delay in seconds.
        **kwargs: Additional task parameters.
    Returns:
        Task ID.
    """
    app = get_celery_app()
    task = execute_agent_task.apply_async(
        args=(agent_config, input_data, str(uuid.uuid4())),
        queue=queue,
        countdown=countdown,
        **kwargs
    )
    return task.id

def schedule_graph_task(graph_config: Dict[str, Any], input_state: Dict[str, Any], 
                       queue: Optional[str] = None, **kwargs) -> str:
    """
    Schedules graph execution task.
    """
    app = get_celery_app()
    task = execute_graph_task.apply_async(
        args=(graph_config, input_state, str(uuid.uuid4())),
        queue=queue,
        **kwargs
    )
    return task.id

# Task result tracking
class TaskResultTracker:
    """Tracks and manages task results."""
    
    def __init__(self, app: Celery):
        self.app = app
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def track_task(self, task_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """
        Tracks task completion with timeout.
        Args:
            task_id: Task ID to track.
            timeout: Maximum wait time.
        Returns:
            Task result or timeout error.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = AsyncResult(task_id, app=self.app)
            
            if result.ready():
                status_info = {
                    "task_id": task_id,
                    "status": result.status,
                    "result": result.result,
                    "successful": result.successful(),
                    "execution_time": time.time() - start_time
                }
                
                if result.successful():
                    self.completed_tasks[task_id] = status_info
                else:
                    status_info["error"] = str(result.result)
                
                return status_info
            
            await asyncio.sleep(1)
        
        # Timeout
        return {
            "task_id": task_id,
            "status": "timeout",
            "error": f"Task timed out after {timeout}s",
            "execution_time": timeout
        }

# Global tracker
_lola_task_tracker = None

def get_task_tracker() -> TaskResultTracker:
    """Returns task result tracker."""
    global _lola_task_tracker
    if _lola_task_tracker is None:
        _lola_task_tracker = TaskResultTracker(get_celery_app())
    return _lola_task_tracker

__all__ = [
    "get_celery_app",
    "LolaTask",
    "execute_agent_task",
    "execute_graph_task",
    "fine_tune_model_task", 
    "process_rag_documents_task",
    "coordinate_agent_swarm_task",
    "get_task_status",
    "schedule_agent_task",
    "schedule_graph_task",
    "get_task_tracker"
]
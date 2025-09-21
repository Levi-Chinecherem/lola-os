# Standard imports
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
from typing import Dict, Any, List
import json

# Local
from lola.libs.celery.tasks import (
    get_celery_app, execute_agent_task, execute_graph_task,
    fine_tune_model_task, process_rag_documents_task,
    get_task_status, schedule_agent_task, get_task_tracker
)
from lola.utils.config import get_config
from lola.core.agent import BaseAgent

"""
Test file for Celery task integration.
Purpose: Ensures all LOLA task patterns work correctly with mocked 
         Celery execution and proper error handling/retry logic.
Full Path: lola-os/tests/test_libs_celery_tasks.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration with Celery enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "celery": {
                "broker_url": "memory://",
                "result_backend": "cache+memory://",
                "include": ["lola.libs.celery.tasks"],
                "max_tasks_per_child": 1000,
                "worker_concurrency": 4,
                "task_time_limit": 3600,
                "task_soft_time_limit": 3000
            },
            "lola_namespace": "test",
            "lola_version": "1.0.0",
            "environment": "test",
            "sentry_dsn": "test-dsn"
        }
        yield mock

@pytest.fixture
def celery_app(mock_config):
    """Fixture for Celery application."""
    app = get_celery_app()
    yield app
    app.close()

@pytest.fixture
def mock_agent():
    """Mock LOLA agent for testing."""
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.__class__.__name__ = "TestAgent"
    mock_agent.model = "gpt-4"
    mock_agent.run.return_value = {"result": "test", "tokens": 50}
    return mock_agent

def test_celery_app_creation(celery_app):
    """Test Celery application creation and configuration."""
    assert celery_app.name == "lola_os"
    assert celery_app.conf["task_serializer"] == "json"
    assert celery_app.conf["worker_prefetch_multiplier"] == 1
    assert celery_app.conf["task_time_limit"] == 3600
    assert celery_app.conf["lola_namespace"] == "test"

def test_celery_app_configuration_from_env(celery_app, monkeypatch):
    """Test configuration loading from environment variables."""
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://test-redis:6379/1")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "rpc://")
    
    # Reload app to pick up env vars
    from lola.libs.celery.tasks import get_celery_app
    app = get_celery_app()
    
    assert app.conf["broker_url"] == "redis://test-redis:6379/1"
    assert app.conf["result_backend"] == "rpc://"

@patch('lola.core.agent.BaseAgent')
def test_execute_agent_task_success(mock_base_agent, celery_app, mock_agent):
    """Test successful agent task execution."""
    # Configure mock
    mock_base_agent.from_config.return_value = mock_agent
    mock_agent.run.return_value = {"answer": "42", "confidence": 0.95}
    
    # Create task config
    agent_config = {
        "agent_type": "TestAgent",
        "model": "gpt-4",
        "tools": ["calculator"],
        "temperature": 0.7
    }
    input_data = {"query": "What is 6*7?"}
    task_id = "test-task-123"
    
    # Execute task (synchronously for test)
    @celery_app.task(bind=True, base=LolaTask)
    def test_task(self):
        return execute_agent_task(agent_config, input_data, task_id)
    
    result = test_task.delay().get(timeout=5)
    
    # Verify success
    assert result["status"] == "completed"
    assert result["task_id"] == task_id
    assert result["result"]["answer"] == "42"
    assert "execution_time" in result
    assert result["agent_type"] == "TestAgent"
    
    # Verify agent was called
    mock_agent.run.assert_called_once_with("What is 6*7?")

@patch('lola.core.agent.BaseAgent')
def test_execute_agent_task_failure(mock_base_agent, celery_app):
    """Test agent task failure and retry logic."""
    # Mock failure
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.run.side_effect = ValueError("Agent computation failed")
    mock_base_agent.from_config.return_value = mock_agent
    
    agent_config = {"agent_type": "TestAgent", "model": "gpt-4"}
    input_data = {"query": "test"}
    task_id = "failure-task"
    
    @celery_app.task(bind=True, base=LolaTask, max_retries=1)
    def test_task(self):
        return execute_agent_task(agent_config, input_data, task_id)
    
    # First execution should fail and retry
    with pytest.raises(ValueError):
        test_task.delay().get(timeout=10)
    
    # Verify retry was attempted
    assert mock_agent.run.call_count == 2  # Original + retry

def test_execute_graph_task(celery_app):
    """Test graph execution task."""
    # Mock graph execution
    with patch('lola.core.graph.StateGraph') as mock_graph:
        mock_graph_instance = Mock()
        mock_graph_instance.from_config.return_value = mock_graph_instance
        mock_graph_instance.execute.return_value = {"final_state": "completed"}
        mock_graph_instance.execution_history = [{"node": "test_node"}]
        
        graph_config = {"nodes": [{"name": "test_node", "type": "llm"}]}
        input_state = {"input": "test"}
        task_id = "graph-task-123"
        
        @celery_app.task(bind=True, base=LolaTask)
        def test_graph_task(self):
            return execute_graph_task(graph_config, input_state, task_id)
        
        result = test_graph_task.delay().get(timeout=5)
        
        # Verify graph execution
        assert result["status"] == "completed"
        assert result["nodes_executed"] == 1
        assert result["final_state"]["final_state"] == "completed"

@patch('lola.libs.vector_dbs.adapter.get_vector_db_adapter')
def test_process_rag_documents_task(mock_vector_db, celery_app):
    """Test RAG document processing task."""
    # Mock VectorDB
    mock_db = Mock()
    mock_db._embedding_dim = 1536
    mock_db.index.return_value = None
    mock_db.get_stats.return_value = {"count": 0}
    mock_vector_db.return_value = mock_db
    
    # Mock Prometheus
    with patch('lola.libs.prometheus.exporter.get_lola_prometheus') as mock_prometheus:
        mock_exporter = Mock()
        mock_prometheus.return_value = mock_exporter
        
        documents = [
            {"id": "doc1", "text": "Sample document 1", "metadata": {"category": "tech"}},
            {"id": "doc2", "text": "Sample document 2", "metadata": {"category": "science"}}
        ]
        vector_db_config = {"type": "memory"}
        task_id = "rag-task-123"
        
        @celery_app.task(bind=True, base=LolaTask)
        def test_rag_task(self):
            return process_rag_documents_task(documents, vector_db_config, task_id)
        
        result = test_rag_task.delay().get(timeout=5)
        
        # Verify processing
        assert result["status"] == "completed"
        assert result["processed_count"] == 2
        assert result["vector_db_type"] == "memory"
        
        # Verify VectorDB was called
        mock_db.index.assert_called_once()
        args = mock_db.index.call_args[0]
        assert len(args[0]) == 2  # 2 embeddings
        assert len(args[1]) == 2  # 2 texts
        assert len(args[2]) == 2  # 2 metadatas
        
        # Verify Prometheus
        mock_exporter.record_rag_indexing.assert_called_once_with(
            documents_count=2,
            indexed_count=2,
            vector_db_type="memory"
        )

def test_task_status(celery_app):
    """Test task status retrieval."""
    # Create successful task
    @celery_app.task(bind=True)
    def success_task():
        return {"result": "success"}
    
    task = success_task.delay()
    task_result = task.get(timeout=5)
    
    status = get_task_status(task.id)
    
    assert status["task_id"] == task.id
    assert status["status"] == "SUCCESS"
    assert status["successful"] is True
    assert status["result"] == task_result

def test_task_status_failure(celery_app):
    """Test failed task status."""
    @celery_app.task(bind=True)
    def failure_task():
        raise ValueError("Test failure")
    
    task = failure_task.delay()
    
    with pytest.raises(ValueError):
        task.get(timeout=5)
    
    status = get_task_status(task.id)
    
    assert status["status"] == "FAILURE"
    assert status["successful"] is False
    assert "ValueError" in str(status["result"])

def test_schedule_agent_task(celery_app):
    """Test agent task scheduling."""
    agent_config = {"agent_type": "TestAgent", "model": "gpt-4"}
    input_data = {"query": "test query"}
    
    task_id = schedule_agent_task(agent_config, input_data, queue="agent_queue")
    
    assert task_id is not None
    assert isinstance(task_id, str)
    assert len(task_id) > 10  # UUID length
    
    # Verify task was queued
    result = AsyncResult(task_id, app=celery_app)
    assert result.status == "PENDING"

def test_task_tracker(celery_app):
    """Test task result tracking."""
    tracker = get_task_tracker()
    
    @celery_app.task(bind=True)
    def test_tracked_task():
        return {"tracked": True}
    
    task = test_tracked_task.delay()
    result = asyncio.run(tracker.track_task(task.id, timeout=10))
    
    assert result["status"] == "SUCCESS"
    assert result["successful"] is True
    assert result["result"]["tracked"] is True

@patch('lola.libs.celery.tasks.LolaTask.on_failure')
def test_task_failure_handling(mock_on_failure, celery_app):
    """Test enhanced task failure handling."""
    @celery_app.task(bind=True, base=LolaTask)
    def failing_task():
        raise ValueError("Intentional failure")
    
    task = failing_task.delay()
    
    with pytest.raises(ValueError):
        task.get(timeout=5)
    
    # Verify enhanced failure handling
    mock_on_failure.assert_called_once()
    exc, traceback, einfo = mock_on_failure.call_args[0][:3]
    assert isinstance(exc, ValueError)
    assert "Intentional failure" in str(exc)

# Integration test for agent task workflow
def test_complete_agent_workflow(celery_app):
    """Integration test for complete agent task workflow."""
    from lola.core.agent import BaseAgent
    
    class TestWorkflowAgent(BaseAgent):
        def run(self, query: str):
            return {"answer": f"Processed: {query}", "tokens": 10}
    
    # Create workflow
    agent = TestWorkflowAgent(model="test-model")
    agent_config = {
        "agent_type": "TestWorkflowAgent",
        "model": "test-model",
        "tools": []
    }
    
    input_data = {"query": "workflow test"}
    task_id = schedule_agent_task(agent_config, input_data)
    
    # Track completion
    tracker = get_task_tracker()
    result = asyncio.run(tracker.track_task(task_id, timeout=30))
    
    assert result["successful"] is True
    assert "answer" in result["result"]
    assert result["result"]["answer"] == "Processed: workflow test"

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(celery_app):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_celery_tasks.py -v --cov=lola/libs/celery --cov-report=html
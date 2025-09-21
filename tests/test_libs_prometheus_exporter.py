# Standard imports
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from contextlib import contextmanager
import time
from typing import Any

# Local
from lola.libs.prometheus.exporter import (
    LolaPrometheusExporter, get_lola_prometheus, AgentRunContext
)
from lola.core.agent import BaseAgent
from lola.utils.config import get_config

"""
Test file for Prometheus exporter integration.
Purpose: Ensures all metric families are created correctly and agent/LLM/EVM 
         metrics are recorded properly.
Full Path: lola-os/tests/test_libs_prometheus_exporter.py
"""

@pytest.fixture
def mock_config():
    """Mock configuration with Prometheus enabled."""
    with patch('lola.utils.config.get_config') as mock:
        mock.return_value = {
            "prometheus_enabled": True,
            "prometheus_namespace": "test",
            "prometheus_multiprocess": False
        }
        yield mock

@pytest.fixture
def mock_registry():
    """Mock Prometheus registry."""
    with patch('prometheus_client.REGISTRY') as mock_reg:
        yield mock_reg

@pytest.fixture
def exporter(mock_config, mock_registry):
    """Fixture for LolaPrometheusExporter."""
    return LolaPrometheusExporter()

@pytest.fixture
def mock_agent():
    """Mock BaseAgent for testing."""
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.__class__.__name__ = "TestAgent"
    mock_agent.model = "gpt-4"
    return mock_agent

def test_exporter_initialization(exporter):
    """Test exporter initializes with correct metrics."""
    assert exporter.enabled is True
    assert len(exporter._metrics) == 13  # Should match METRIC_FAMILIES
    assert "lola_agent_runs_total" in exporter._metrics
    assert "lola_llm_tokens_total" in exporter._metrics

def test_exporter_disabled(mock_config, mock_registry):
    """Test exporter behavior when disabled."""
    with patch('lola.utils.config.get_config', return_value={"prometheus_enabled": False}):
        exporter = LolaPrometheusExporter()
        assert exporter.enabled is False
        assert len(exporter._metrics) == 0

def test_metric_families_created(exporter):
    """Test all metric families are created correctly."""
    expected_families = [
        "lola_agent_runs_total",
        "lola_agent_run_duration_seconds", 
        "lola_agent_errors_total",
        "lola_llm_calls_total",
        "lola_llm_call_duration_seconds",
        "lola_llm_tokens_total",
        "lola_llm_cost_usd",
        "lola_evm_calls_total",
        "lola_evm_call_duration_seconds",
        "lola_evm_gas_used_total",
        "lola_evm_errors_total",
        "lola_graph_executions_total",
        "lola_graph_node_executions_total",
        "lola_graph_node_duration_seconds",
        "lola_rag_queries_total",
        "lola_rag_retrieval_latency_seconds",
        "lola_rag_hits_total",
        "lola_rag_hit_rate",
        "lola_tool_calls_total",
        "lola_tool_call_duration_seconds",
        "lola_tool_errors_total",
        "lola_process_cpu_percent",
        "lola_process_memory_mb",
        "lola_system_load"
    ]
    
    # Check all expected metrics exist
    for family in expected_families:
        assert family in exporter._metrics, f"Missing metric: {family}"
        assert callable(exporter._metrics[family].inc), f"Metric not callable: {family}"

def test_agent_run_context_success(mock_agent, exporter):
    """Test successful agent run metrics."""
    operation = "run"
    
    with patch.object(exporter._metrics["lola_agent_runs_total"], "labels") as mock_labels:
        with patch.object(exporter._metrics["lola_agent_run_duration_seconds"], "labels") as mock_histogram:
            mock_timer = Mock()
            mock_histogram.return_value.time.return_value = mock_timer
            
            with exporter.start_agent_run(mock_agent, operation):
                time.sleep(0.01)  # Simulate work
            
            # Verify counters incremented
            mock_labels.assert_called_with(
                agent_type="TestAgent",
                model="gpt-4", 
                operation="run",
                status="started"
            )
            mock_labels.return_value.inc.assert_called_once()
            
            # Verify histogram timing
            mock_histogram.assert_called_with(
                agent_type="TestAgent",
                model="gpt-4",
                operation="run"
            )
            mock_timer.assert_called_once()

def test_agent_run_context_error(mock_agent, exporter):
    """Test agent run error handling."""
    operation = "run"
    
    with pytest.raises(ValueError):
        with exporter.start_agent_run(mock_agent, operation):
            raise ValueError("Test error")
    
    # Verify error counter incremented
    mock_error_labels = Mock()
    with patch.object(exporter._metrics["lola_agent_errors_total"], "labels") as mock_error_counter:
        mock_error_counter.return_value = mock_error_labels
        pytest.fail("Should not reach here")  # Force exit to trigger error counter
    
    mock_error_labels.inc.assert_called_once()

def test_llm_call_context(exporter):
    """Test LLM call context manager."""
    model = "gpt-4"
    provider = "openai"
    
    with patch.object(exporter._metrics["lola_llm_calls_total"], "labels") as mock_counter:
        with patch.object(exporter._metrics["lola_llm_call_duration_seconds"], "labels") as mock_histogram:
            with exporter.llm_call(model, provider):
                time.sleep(0.01)  # Simulate API call
            
            # Verify metrics recorded
            mock_counter.assert_called_with(
                model="gpt-4",
                provider="openai",
                operation="completion",
                status="completed"
            )
            mock_counter.return_value.inc.assert_called_once()
            
            mock_histogram.assert_called_with(
                model="gpt-4",
                provider="openai", 
                operation="completion"
            )
            mock_histogram.return_value.observe.assert_called_once()

def test_record_llm_tokens(exporter):
    """Test LLM token recording."""
    model = "gpt-4"
    tokens_in = 100
    tokens_out = 50
    cost_usd = 0.02
    
    with patch.object(exporter._metrics["lola_llm_tokens_total"], "labels") as mock_tokens:
        with patch.object(exporter._metrics["lola_llm_cost_usd"], "labels") as mock_cost:
            exporter.record_llm_tokens(model, tokens_in, tokens_out, cost_usd)
            
            # Verify token counters
            mock_tokens.assert_has_calls([
                # Input tokens
                call().labels(model="gpt-4", direction="input").inc(100),
                # Output tokens  
                call().labels(model="gpt-4", direction="output").inc(50)
            ])
            
            # Verify cost gauge
            mock_cost.assert_called_with(model="gpt-4")
            mock_cost.return_value.set.assert_called_with(0.02)

def test_record_evm_call(exporter):
    """Test EVM call metrics."""
    chain = "ethereum"
    operation = "contract_call"
    duration = 0.5
    gas_used = 21000
    success = True
    
    with patch.object(exporter._metrics["lola_evm_calls_total"], "labels") as mock_calls:
        with patch.object(exporter._metrics["lola_evm_call_duration_seconds"], "labels") as mock_duration:
            with patch.object(exporter._metrics["lola_evm_gas_used_total"], "labels") as mock_gas:
                exporter.record_evm_call(chain, operation, duration, gas_used, success)
                
                # Verify calls counter
                mock_calls.assert_called_with(
                    chain="ethereum",
                    operation="contract_call", 
                    status="success"
                )
                mock_calls.return_value.inc.assert_called_once()
                
                # Verify duration histogram
                mock_duration.assert_called_with(
                    chain="ethereum",
                    operation="contract_call"
                )
                mock_duration.return_value.observe.assert_called_with(0.5)
                
                # Verify gas counter
                mock_gas.assert_called_with(
                    chain="ethereum",
                    operation="contract_call"
                )
                mock_gas.return_value.inc.assert_called_with(21000)

def test_record_evm_error(exporter):
    """Test EVM error recording."""
    chain = "polygon"
    operation = "transaction_failed"
    duration = 1.2
    success = False
    
    with patch.object(exporter._metrics["lola_evm_errors_total"], "labels") as mock_errors:
        exporter.record_evm_call(chain, operation, duration, success=success)
        
        mock_errors.assert_called_with(
            chain="polygon",
            operation="transaction_failed"
        )
        mock_errors.return_value.inc.assert_called_once()

def test_tool_call_metrics(exporter):
    """Test tool execution metrics."""
    tool_name = "web_search"
    duration = 0.3
    success = False
    error_type = "timeout"
    
    with patch.object(exporter._metrics["lola_tool_calls_total"], "labels") as mock_calls:
        with patch.object(exporter._metrics["lola_tool_call_duration_seconds"], "labels") as mock_duration:
            with patch.object(exporter._metrics["lola_tool_errors_total"], "labels") as mock_errors:
                exporter.record_tool_call(tool_name, duration, success, error_type)
                
                # Verify call counter with error status
                mock_calls.assert_called_with(
                    tool="web_search",
                    status="error"
                )
                mock_calls.return_value.inc.assert_called_once()
                
                # Verify duration
                mock_duration.assert_called_with(tool="web_search")
                mock_duration.return_value.observe.assert_called_with(0.3)
                
                # Verify error counter
                mock_errors.assert_called_with(
                    tool="web_search",
                    error_type="timeout"
                )
                mock_errors.return_value.inc.assert_called_once()

def test_rag_retrieval_metrics(exporter):
    """Test RAG retrieval metrics."""
    retriever_type = "llamaindex"
    duration = 0.15
    hits = 3
    total_searched = 25
    
    with patch.object(exporter._metrics["lola_rag_queries_total"], "labels") as mock_queries:
        with patch.object(exporter._metrics["lola_rag_retrieval_latency_seconds"], "labels") as mock_latency:
            with patch.object(exporter._metrics["lola_rag_hits_total"], "labels") as mock_hits:
                with patch.object(exporter._metrics["lola_rag_hit_rate"], "labels") as mock_rate:
                    exporter.record_rag_retrieval(retriever_type, duration, hits, total_searched)
                    
                    # Verify query counter
                    mock_queries.assert_called_with(retriever="llamaindex")
                    mock_queries.return_value.inc.assert_called_once()
                    
                    # Verify latency
                    mock_latency.assert_called_with(retriever="llamaindex")
                    mock_latency.return_value.observe.assert_called_with(0.15)
                    
                    # Verify hits
                    mock_hits.assert_called_with(retriever="llamaindex")
                    mock_hits.return_value.inc.assert_called_with(3)
                    
                    # Verify hit rate (3/25 = 0.12)
                    mock_rate.assert_called_with(retriever="llamaindex")
                    mock_rate.return_value.set.assert_called_with(0.12)

def test_get_metrics_content(exporter):
    """Test metrics content generation."""
    content = exporter.get_metrics_content()
    
    assert isinstance(content, bytes)
    assert len(content) > 0
    assert b"# HELP" in content or b"# TYPE" in content  # Prometheus format

def test_factory_function(mock_config):
    """Test singleton factory."""
    exporter1 = get_lola_prometheus()
    exporter2 = get_lola_prometheus()
    
    assert exporter1 is exporter2  # Same instance

def test_noop_context_when_disabled(mock_config):
    """Test no-op behavior when disabled."""
    with patch('lola.utils.config.get_config', return_value={"prometheus_enabled": False}):
        exporter = LolaPrometheusExporter()
        
        # Should not raise errors
        with exporter.start_agent_run(Mock(), "test"):
            pass  # No-op context

# Integration test for system metrics
def test_system_metrics_collector():
    """Test system metrics are collected."""
    from lola.libs.prometheus.exporter import SystemMetricsCollector
    
    collector = SystemMetricsCollector()
    metrics = collector.collect()
    
    assert len(metrics) >= 2  # At least CPU and memory
    assert any("cpu_percent" in str(m) for m in metrics)
    assert any("memory_mb" in str(m) for m in metrics)

# Coverage marker
@pytest.mark.coverage
def test_coverage_marker(exporter):
    """Marker for coverage reporting."""
    pass

# Run with: pytest tests/test_libs_prometheus_exporter.py -v --cov=lola/libs/prometheus --cov-report=html
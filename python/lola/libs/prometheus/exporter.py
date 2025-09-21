# Standard imports
import typing as tp
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict
import time
from contextlib import contextmanager
import threading

# Third-party
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, CollectorRegistry, 
        generate_latest, CONTENT_TYPE_LATEST, REGISTRY, MultiProcessCollector
    )
    from prometheus_client.openmetrics.exposition import generate_openmetrics
    import psutil
except ImportError:
    raise ImportError("Prometheus client not installed. Run 'poetry add prometheus-client psutil'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.core.agent import BaseAgent
from lola.core.graph import StateGraph
from lola.agnostic.unified import UnifiedModelManager

"""
File: Prometheus metrics exporter for LOLA OS.
Purpose: Provides comprehensive observability with counters, gauges, histograms 
         for agent performance, LLM usage, EVM calls, and system resources.
How: Uses prometheus-client with LOLA-specific metric families; supports 
     multi-process mode and custom collectors; exposes /metrics endpoint.
Why: Essential for production monitoring, alerting, and performance analysis 
     of distributed agent systems.
Full Path: lola-os/python/lola/libs/prometheus/exporter.py
"""

class LolaPrometheusExporter:
    """LolaPrometheusExporter: Comprehensive Prometheus integration for LOLA OS.
    Does NOT start HTTP server—provides metrics collection and registry management."""

    # Metric families
    METRIC_FAMILIES = {
        # Agent metrics
        "lola_agent_runs_total": Counter,
        "lola_agent_run_duration_seconds": Histogram,
        "lola_agent_errors_total": Counter,
        
        # LLM metrics  
        "lola_llm_calls_total": Counter,
        "lola_llm_call_duration_seconds": Histogram,
        "lola_llm_tokens_total": Counter,
        "lola_llm_cost_usd": Gauge,
        
        # EVM metrics
        "lola_evm_calls_total": Counter,
        "lola_evm_call_duration_seconds": Histogram,
        "lola_evm_gas_used_total": Counter,
        "lola_evm_errors_total": Counter,
        
        # Graph metrics
        "lola_graph_executions_total": Counter,
        "lola_graph_node_executions_total": Counter,
        "lola_graph_node_duration_seconds": Histogram,
        
        # RAG metrics
        "lola_rag_queries_total": Counter,
        "lola_rag_retrieval_latency_seconds": Histogram,
        "lola_rag_hits_total": Counter,
        "lola_rag_hit_rate": Gauge,
        
        # System metrics
        "lola_process_cpu_percent": Gauge,
        "lola_process_memory_mb": Gauge,
        "lola_system_load": Gauge,
        
        # Tool metrics
        "lola_tool_calls_total": Counter,
        "lola_tool_call_duration_seconds": Histogram,
        "lola_tool_errors_total": Counter,
    }

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initializes Prometheus exporter.
        Args:
            registry: Optional custom registry (default: global REGISTRY).
        """
        config = get_config()
        self.enabled = config.get("prometheus_enabled", True)
        self.namespace = config.get("prometheus_namespace", "lola")
        self.registry = registry or REGISTRY
        
        if not self.enabled:
            logger.info("Prometheus exporter disabled in configuration")
            return
        
        # Multi-process support for production
        if config.get("prometheus_multiprocess", True):
            try:
                MultiProcessCollector(self.registry)
                logger.info("Prometheus multi-process collector enabled")
            except Exception as exc:
                logger.warning(f"Multi-process collector setup failed: {exc}")
        
        self._metrics = {}
        self._lock = threading.RLock()
        self._initialize_metrics()
        
        # System metrics collector
        self._system_collector = SystemMetricsCollector()
        self.registry.register(self._system_collector)
        
        logger.info(f"Prometheus exporter initialized with {len(self._metrics)} metric families")

    def _initialize_metrics(self) -> None:
        """Initializes all LOLA-specific metric families."""
        with self._lock:
            for metric_name, metric_class in self.METRIC_FAMILIES.items():
                # Add LOLA namespace prefix
                full_name = f"{self.namespace}_{metric_name}"
                
                # Create metric with common labels
                if metric_class in [Counter, Gauge]:
                    self._metrics[metric_name] = metric_class(
                        full_name,
                        f"LOLA {metric_name.replace('_', ' ').title()}",
                        ['agent_type', 'model', 'operation', 'status'],
                        registry=self.registry
                    )
                elif metric_class in [Histogram, Summary]:
                    self._metrics[metric_name] = metric_class(
                        full_name,
                        f"LOLA {metric_name.replace('_', ' ').title()}",
                        ['agent_type', 'model', 'operation'],
                        registry=self.registry
                    )
        
        logger.debug(f"Initialized {len(self._metrics)} Prometheus metrics")

    def start_agent_run(self, agent: BaseAgent, operation: str) -> tp.Any:
        """
        Starts timing for agent operation.
        Args:
            agent: BaseAgent instance.
            operation: Operation name (run, plan, etc.).
        Returns:
            Context manager for timing.
        """
        if not self.enabled:
            return _NoopContext()
        
        agent_type = agent.__class__.__name__
        model = getattr(agent, 'model', 'unknown')
        
        # Start histogram timing
        duration = self._metrics["lola_agent_run_duration_seconds"].labels(
            agent_type=agent_type,
            model=model,
            operation=operation
        ).time()
        
        # Increment run counter
        self._metrics["lola_agent_runs_total"].labels(
            agent_type=agent_type,
            model=model,
            operation=operation,
            status="started"
        ).inc()
        
        return AgentRunContext(
            self, agent_type, model, operation, duration=duration
        )

    @contextmanager
    def llm_call(self, model: str, provider: str = "unknown", operation: str = "completion"):
        """
        Context manager for LLM calls.
        Args:
            model: Model name.
            provider: Provider (openai, anthropic, etc.).
            operation: Operation type.
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        tokens_in = 0
        tokens_out = 0
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            # Record metrics
            self._metrics["lola_llm_calls_total"].labels(
                model=model,
                provider=provider,
                operation=operation,
                status="completed"
            ).inc()
            
            self._metrics["lola_llm_call_duration_seconds"].labels(
                model=model,
                provider=provider,
                operation=operation
            ).observe(duration)
            
            # Record tokens if available
            if tokens_in > 0:
                self._metrics["lola_llm_tokens_total"].labels(
                    model=model,
                    direction="input"
                ).inc(tokens_in)
                
            if tokens_out > 0:
                self._metrics["lola_llm_tokens_total"].labels(
                    model=model,
                    direction="output"
                ).inc(tokens_out)

    def record_llm_tokens(self, model: str, tokens_in: int, tokens_out: int, cost_usd: float = 0.0):
        """
        Records LLM token usage and cost.
        Args:
            model: Model name.
            tokens_in: Input tokens.
            tokens_out: Output tokens.
            cost_usd: Cost in USD.
        """
        if not self.enabled:
            return
        
        # Token counters
        if tokens_in > 0:
            self._metrics["lola_llm_tokens_total"].labels(
                model=model,
                direction="input"
            ).inc(tokens_in)
            
        if tokens_out > 0:
            self._metrics["lola_llm_tokens_total"].labels(
                model=model,
                direction="output"
            ).inc(tokens_out)
        
        # Cost gauge
        if cost_usd > 0:
            self._metrics["lola_llm_cost_usd"].labels(model=model).set(cost_usd)

    def record_evm_call(self, chain: str, operation: str, duration: float, 
                       gas_used: int = 0, success: bool = True):
        """
        Records EVM operation metrics.
        Args:
            chain: Chain name (ethereum, polygon, etc.).
            operation: Operation type (call, query, etc.).
            duration: Operation duration in seconds.
            gas_used: Gas consumed.
            success: Whether operation succeeded.
        """
        if not self.enabled:
            return
        
        status = "success" if success else "error"
        
        self._metrics["lola_evm_calls_total"].labels(
            chain=chain,
            operation=operation,
            status=status
        ).inc()
        
        self._metrics["lola_evm_call_duration_seconds"].labels(
            chain=chain,
            operation=operation
        ).observe(duration)
        
        if gas_used > 0:
            self._metrics["lola_evm_gas_used_total"].labels(
                chain=chain,
                operation=operation
            ).inc(gas_used)
        
        if not success:
            self._metrics["lola_evm_errors_total"].labels(
                chain=chain,
                operation=operation
            ).inc()

    def record_tool_call(self, tool_name: str, duration: float, 
                        success: bool = True, error_type: Optional[str] = None):
        """
        Records tool execution metrics.
        Args:
            tool_name: Tool name.
            duration: Execution time.
            success: Success status.
            error_type: Error type if failed.
        """
        if not self.enabled:
            return
        
        labels = {
            "tool": tool_name,
            "status": "success" if success else "error"
        }
        if error_type:
            labels["error_type"] = error_type
            
        self._metrics["lola_tool_calls_total"].labels(**labels).inc()
        self._metrics["lola_tool_call_duration_seconds"].labels(
            tool=tool_name
        ).observe(duration)
        
        if not success:
            self._metrics["lola_tool_errors_total"].labels(
                tool=tool_name,
                error_type=error_type or "unknown"
            ).inc()

    def record_rag_retrieval(self, retriever_type: str, duration: float, 
                           hits: int, total_searched: int):
        """
        Records RAG retrieval metrics.
        Args:
            retriever_type: Retriever used (llamaindex, haystack, etc.).
            duration: Retrieval time.
            hits: Number of relevant documents retrieved.
            total_searched: Total documents searched.
        """
        if not self.enabled:
            return
        
        hit_rate = hits / total_searched if total_searched > 0 else 0.0
        
        self._metrics["lola_rag_queries_total"].labels(
            retriever=retriever_type
        ).inc()
        
        self._metrics["lola_rag_retrieval_latency_seconds"].labels(
            retriever=retriever_type
        ).observe(duration)
        
        self._metrics["lola_rag_hits_total"].labels(
            retriever=retriever_type
        ).inc(hits)
        
        self._metrics["lola_rag_hit_rate"].labels(
            retriever=retriever_type
        ).set(hit_rate)

    def get_metrics_content(self, format: str = "prometheus") -> bytes:
        """
        Generates metrics content for HTTP endpoint.
        Args:
            format: Output format ("prometheus" or "openmetrics").
        Returns:
            Metrics content as bytes.
        """
        if not self.enabled:
            return b"# Prometheus metrics disabled\n"
        
        try:
            if format == "openmetrics":
                return generate_openmetrics(registry=self.registry)
            else:
                content_type = CONTENT_TYPE_LATEST
                content = generate_latest(registry=self.registry)
                return content
        except Exception as exc:
            logger.error(f"Metrics generation failed: {str(exc)}")
            return b"# Error generating metrics\n"

    def clear_metrics(self) -> None:
        """Clears all metrics (for testing)."""
        if not self.enabled:
            return
        
        with self._lock:
            for metric in self._metrics.values():
                metric.clear()
        logger.debug("Prometheus metrics cleared")

    def get_registry(self) -> CollectorRegistry:
        """Returns the Prometheus registry."""
        return self.registry


class SystemMetricsCollector:
    """Custom collector for system metrics."""
    
    def __init__(self):
        self._last_cpu_time = time.time()
        self._last_cpu_usage = psutil.cpu_percent(interval=None)

    def collect(self) -> List[Any]:
        """Collects system metrics."""
        metrics = []
        
        try:
            # CPU usage
            current_cpu = psutil.cpu_percent(interval=0.1)
            cpu_gauge = Gauge(
                'lola_process_cpu_percent',
                'CPU usage percentage',
                registry=REGISTRY
            )
            cpu_gauge.set(current_cpu)
            metrics.append(cpu_gauge)
            
            # Memory usage
            memory = psutil.virtual_memory()
            mem_gauge = Gauge(
                'lola_process_memory_mb', 
                'Memory usage in MB',
                registry=REGISTRY
            )
            mem_gauge.set(memory.used / 1024 / 1024)
            metrics.append(mem_gauge)
            
            # System load
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
                load_gauge = Gauge(
                    'lola_system_load',
                    'System load average',
                    ['period'],
                    registry=REGISTRY
                )
                load_gauge.labels(period='1min').set(load_avg[0])
                load_gauge.labels(period='5min').set(load_avg[1])
                load_gauge.labels(period='15min').set(load_avg[2])
                metrics.append(load_gauge)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_gauge = Gauge(
                'lola_disk_usage_percent',
                'Disk usage percentage',
                registry=REGISTRY
            )
            disk_gauge.set(disk.percent)
            metrics.append(disk_gauge)
            
        except Exception as exc:
            logger.debug(f"System metrics collection failed: {str(exc)}")
        
        return metrics


class AgentRunContext:
    """Context manager for agent run timing."""
    
    def __init__(self, exporter: LolaPrometheusExporter, agent_type: str, 
                model: str, operation: str, duration: tp.Any = None):
        self.exporter = exporter
        self.agent_type = agent_type
        self.model = model
        self.operation = operation
        self.duration = duration
        self.start_time = time.time()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if self.duration:
            self.duration()
        
        status = "success"
        if exc_type is not None:
            status = "error"
            self.exporter._metrics["lola_agent_errors_total"].labels(
                agent_type=self.agent_type,
                model=self.model,
                operation=self.operation,
                status=status
            ).inc()
        
        self.exporter._metrics["lola_agent_runs_total"].labels(
            agent_type=self.agent_type,
            model=self.model,
            operation=self.operation,
            status=status
        ).inc()


class _NoopContext:
    """No-op context manager when metrics disabled."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


# Global instance
_lola_prometheus = None

def get_lola_prometheus(registry: Optional[CollectorRegistry] = None) -> LolaPrometheusExporter:
    """Singleton Prometheus exporter."""
    global _lola_prometheus
    if _lola_prometheus is None:
        _lola_prometheus = LolaPrometheusExporter(registry)
    return _lola_prometheus

__all__ = [
    "LolaPrometheusExporter",
    "get_lola_prometheus",
    "AgentRunContext"
]
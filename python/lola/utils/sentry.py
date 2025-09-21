# Standard imports
import typing as tp
from typing import Optional, Dict, Any, Callable
import functools
import logging
from contextlib import contextmanager

# Third-party
try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.aiohttp import AioHttpIntegration
    from sentry_sdk.integrations.stdlib import StdlibIntegration
except ImportError:
    raise ImportError("Sentry SDK not installed. Run 'poetry add sentry-sdk'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.core.agent import BaseAgent  # For agent context
from lola.core.graph import StateGraph  # For graph context

"""
File: Sentry integration for LOLA OS error tracking and observability.
Purpose: Provides centralized error capture, performance monitoring, and 
         context enrichment for LOLA agents and workflows.
How: Initializes Sentry client with LOLA-specific integrations; provides 
     decorators and context managers for agent/graph tracing; captures 
     breadcrumbs for debugging.
Why: Enables production observability, error correlation across distributed 
     agent executions, and performance insights without changing application code.
Full Path: lola-os/python/lola/libs/sentry/sdk.py
"""

class LolaSentryIntegration:
    """LolaSentryIntegration: Sentry client wrapper for LOLA OS.
    Does NOT initialize Sentry globally—lazy initialization per config."""

    def __init__(self):
        """
        Initializes Sentry integration with LOLA configuration.
        Does Not: Capture events—requires explicit calls or decorators.
        """
        config = get_config()
        self.dsn = config.get("sentry_dsn")
        self.environment = config.get("environment", "development")
        self.sample_rate = config.get("sentry_sample_rate", 1.0)
        self.traces_sample_rate = config.get("sentry_traces_sample_rate", 0.2)
        self.release = config.get("sentry_release", "lola-os@1.0.0")
        
        self._client = None
        self._initialized = False
        self._agent_context: Optional[Dict[str, Any]] = None
        self._graph_context: Optional[Dict[str, Any]] = None
        
        if self.dsn:
            self._initialize_client()
            logger.info("Sentry integration initialized")
        else:
            logger.info("Sentry DSN not configured - error tracking disabled")

    def _initialize_client(self) -> None:
        """Initializes Sentry client with LOLA-specific configuration."""
        try:
            # Basic Sentry configuration
            sentry_sdk.init(
                dsn=self.dsn,
                environment=self.environment,
                release=self.release,
                sample_rate=self.sample_rate,
                traces_sample_rate=self.traces_sample_rate,
                
                # Integrations
                integrations=[
                    LoggingIntegration(
                        level=logging.ERROR,  # Capture ERROR and above
                        event_level=logging.ERROR
                    ),
                    StdlibIntegration(),
                    AioHttpIntegration()
                ],
                
                # LOLA-specific configuration
                before_send=self._before_send,
                before_send_transaction=self._before_send_transaction
            )
            
            # Add LOLA-specific tags
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("app", "lola-os")
                scope.set_tag("version", self.release)
                scope.set_tag("language", "python")
            
            self._client = sentry_sdk
            self._initialized = True
            
        except Exception as exc:
            logger.error(f"Sentry initialization failed: {str(exc)}")
            self._initialized = False

    def _before_send(self, event: Dict[str, Any], hint: Optional[tp.Any] = None) -> Optional[Dict[str, Any]]:
        """
        Custom before_send hook to enrich events with LOLA context.
        Args:
            event: Sentry event dictionary.
            hint: Event hint.
        Returns:
            Modified event or None to skip sending.
        """
        if not self._initialized:
            return event
        
        try:
            # Add agent context if available
            if self._agent_context:
                event.setdefault("contexts", {}).setdefault("lola", {}).update({
                    "agent": self._agent_context,
                    "agent_id": self._agent_context.get("agent_id")
                })
            
            # Add graph context if available
            if self._graph_context:
                event.setdefault("contexts", {}).setdefault("lola", {}).setdefault("graph", 
                    self._graph_context)
            
            # Add current config environment
            config = get_config()
            event.setdefault("tags", {})["lola_config_mode"] = config.get("mode", "default")
            
            return event
            
        except Exception as exc:
            logger.debug(f"Sentry before_send error: {str(exc)}")
            return event

    def _before_send_transaction(self, transaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Custom transaction processing."""
        if not self._initialized:
            return transaction
        
        try:
            # Enrich transactions with LOLA-specific data
            transaction["contexts"]["lola"] = {
                "agent_active": bool(self._agent_context),
                "graph_active": bool(self._graph_context)
            }
            return transaction
            
        except Exception:
            return transaction

    def capture_exception(self, exc: Exception, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Captures an exception with optional LOLA context.
        Args:
            exc: Exception to capture.
            context: Optional context dictionary (agent, graph, etc.).
        Returns:
            Event ID from Sentry.
        """
        if not self._initialized:
            logger.warning("Sentry not initialized, cannot capture exception")
            return "not-initialized"
        
        try:
            with self._set_context(context or {}):
                event_id = self._client.capture_exception(exc)
                logger.debug(f"Sentry captured exception: {event_id}")
                return event_id
                
        except Exception as sentry_exc:
            logger.error(f"Sentry capture failed: {str(sentry_exc)}")
            return "capture-failed"

    def capture_message(self, message: str, level: str = "info", 
                       context: Optional[Dict[str, Any]] = None) -> str:
        """
        Captures a custom message.
        Args:
            message: Message to capture.
            level: Log level (info, warning, error, fatal).
            context: Optional context.
        Returns:
            Event ID.
        """
        if not self._initialized:
            return "not-initialized"
        
        try:
            with self._set_context(context or {}):
                event_id = self._client.capture_message(message, level=level)
                return event_id
                
        except Exception as exc:
            logger.error(f"Sentry message capture failed: {str(exc)}")
            return "capture-failed"

    def start_agent_transaction(self, agent: BaseAgent, operation: str) -> str:
        """
        Starts a Sentry transaction for agent execution.
        Args:
            agent: LOLA BaseAgent instance.
            operation: Operation name (e.g., "run", "plan").
        Returns:
            Transaction ID.
        """
        if not self._initialized:
            return "not-initialized"
        
        try:
            # Set agent context
            self._agent_context = {
                "agent_type": agent.__class__.__name__,
                "model": getattr(agent, 'model', 'unknown'),
                "tools_count": len(getattr(agent, 'tools', [])),
                "operation": operation
            }
            
            # Start transaction
            with self._client.start_transaction(
                op="agent",
                name=f"{agent.__class__.__name__}.{operation}"
            ) as transaction:
                transaction.set_tag("lola.agent_type", agent.__class__.__name__)
                transaction.set_tag("lola.model", agent.model)
                transaction.set_tag("lola.operation", operation)
                
                # Set user context if available
                if hasattr(agent, 'user_id'):
                    transaction.set_user({"id": agent.user_id})
            
            return transaction.span_id
            
        except Exception as exc:
            logger.error(f"Sentry agent transaction failed: {str(exc)}")
            return "transaction-failed"

    @contextmanager
    def agent_context(self, agent: BaseAgent, operation: str):
        """
        Context manager for agent operations.
        Args:
            agent: BaseAgent instance.
            operation: Operation name.
        """
        transaction_id = self.start_agent_transaction(agent, operation)
        try:
            self._agent_context = {
                "agent_type": agent.__class__.__name__,
                "model": getattr(agent, 'model', 'unknown'),
                "operation": operation,
                "transaction_id": transaction_id
            }
            yield transaction_id
        finally:
            self._agent_context = None
            if self._initialized:
                self._client.get_current_hub().scope.clear()

    def start_graph_transaction(self, graph: StateGraph, operation: str) -> str:
        """
        Starts transaction for graph execution.
        Args:
            graph: StateGraph instance.
            operation: Graph operation name.
        Returns:
            Transaction ID.
        """
        if not self._initialized:
            return "not-initialized"
        
        try:
            self._graph_context = {
                "graph_type": "StateGraph",
                "nodes_count": len(graph.nodes) if hasattr(graph, 'nodes') else 0,
                "operation": operation
            }
            
            with self._client.start_transaction(
                op="graph",
                name=f"StateGraph.{operation}"
            ) as transaction:
                transaction.set_tag("lola.graph_type", "StateGraph")
                transaction.set_tag("lola.operation", operation)
                
                # Add node spans
                if hasattr(graph, 'nodes'):
                    for node in graph.nodes:
                        span = transaction.start_child({
                            "op": "graph.node",
                            "description": f"Node: {node.name if hasattr(node, 'name') else 'unknown'}"
                        })
                        span.finish()
            
            return transaction.span_id
            
        except Exception as exc:
            logger.error(f"Sentry graph transaction failed: {str(exc)}")
            return "transaction-failed"

    @contextmanager
    def graph_context(self, graph: StateGraph, operation: str):
        """Context manager for graph operations."""
        transaction_id = self.start_graph_transaction(graph, operation)
        try:
            self._graph_context = {
                "graph_type": "StateGraph",
                "operation": operation,
                "transaction_id": transaction_id
            }
            yield transaction_id
        finally:
            self._graph_context = None

    def add_breadcrumb(self, message: str, level: str = "info", 
                      category: Optional[str] = None) -> None:
        """
        Adds breadcrumb for debugging.
        Args:
            message: Breadcrumb message.
            level: Log level.
            category: Optional category (agent, graph, tool, etc.).
        """
        if not self._initialized:
            return
        
        try:
            self._client.add_breadcrumb(
                message=message,
                level=level,
                category=category or "lola",
                timestamp=None  # Use current time
            )
        except Exception as exc:
            logger.debug(f"Sentry breadcrumb failed: {str(exc)}")

    def set_user_context(self, user_id: str, email: Optional[str] = None, 
                        extra: Optional[Dict[str, Any]] = None) -> None:
        """Sets user context for the current scope."""
        if not self._initialized:
            return
        
        try:
            with self._client.configure_scope() as scope:
                scope.user = {
                    "id": user_id,
                    "email": email
                }
                if extra:
                    scope.user.update(extra)
        except Exception as exc:
            logger.debug(f"Sentry user context failed: {str(exc)}")

    def set_extra_context(self, key: str, value: Any) -> None:
        """Sets extra context data."""
        if not self._initialized:
            return
        
        try:
            with self._client.configure_scope() as scope:
                scope.set_extra(key, value)
        except Exception as exc:
            logger.debug(f"Sentry extra context failed: {str(exc)}")

    @contextmanager
    def _set_context(self, context: Dict[str, Any]):
        """Internal context manager for setting LOLA context."""
        if not self._initialized:
            yield
            return
        
        # Save current context
        prev_agent = self._agent_context
        prev_graph = self._graph_context
        
        try:
            if "agent" in context:
                self._agent_context = context["agent"]
            if "graph" in context:
                self._graph_context = context["graph"]
            
            yield
        finally:
            # Restore previous context
            self._agent_context = prev_agent
            self._graph_context = prev_graph

    def is_enabled(self) -> bool:
        """Checks if Sentry integration is enabled."""
        return self._initialized and bool(self.dsn)

    def flush(self, timeout: float = 2.0) -> bool:
        """Flushes queued events."""
        if not self._initialized:
            return True
        
        try:
            return self._client.flush(timeout=timeout)
        except Exception as exc:
            logger.warning(f"Sentry flush failed: {str(exc)}")
            return False


# Global singleton instance
_lola_sentry = None

def get_lola_sentry() -> LolaSentryIntegration:
    """Gets the global LOLA Sentry instance."""
    global _lola_sentry
    if _lola_sentry is None:
        _lola_sentry = LolaSentryIntegration()
    return _lola_sentry


# Decorators for easy integration
def sentry_agent_operation(operation_name: str):
    """
    Decorator for agent operations.
    Usage: @sentry_agent_operation("run") def my_agent_method(self): ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            sentry = get_lola_sentry()
            if not sentry.is_enabled():
                return func(self, *args, **kwargs)
            
            with sentry.agent_context(self, operation_name):
                try:
                    sentry.add_breadcrumb(f"Starting {operation_name}", "info", "agent")
                    result = func(self, *args, **kwargs)
                    sentry.add_breadcrumb(f"Completed {operation_name}", "info", "agent")
                    return result
                except Exception as exc:
                    sentry.add_breadcrumb(f"Failed {operation_name}", "error", "agent")
                    sentry.capture_exception(exc, {
                        "agent": {"type": self.__class__.__name__, "operation": operation_name}
                    })
                    raise
        return wrapper
    return decorator


def sentry_graph_node(node_name: str):
    """
    Decorator for graph nodes.
    Usage: @sentry_graph_node("llm_call") def node_function(state): ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: tp.Any, *args, **kwargs):
            sentry = get_lola_sentry()
            if not sentry.is_enabled():
                return func(state, *args, **kwargs)
            
            try:
                sentry.add_breadcrumb(f"Executing node {node_name}", "info", "graph")
                result = func(state, *args, **kwargs)
                sentry.add_breadcrumb(f"Node {node_name} completed", "info", "graph")
                return result
            except Exception as exc:
                sentry.add_breadcrumb(f"Node {node_name} failed", "error", "graph")
                sentry.capture_exception(exc, {"graph": {"node": node_name}})
                raise
        return wrapper
    return decorator


# Convenience functions
def capture_lola_error(exc: Exception, context: Dict[str, Any] = None) -> str:
    """Quick function to capture LOLA-specific errors."""
    sentry = get_lola_sentry()
    return sentry.capture_exception(exc, context or {})


def log_agent_breadcrumb(agent: BaseAgent, message: str, level: str = "info"):
    """Quick breadcrumb for agent events."""
    sentry = get_lola_sentry()
    if sentry.is_enabled():
        sentry.add_breadcrumb(message, level, "agent")


__all__ = [
    "LolaSentryIntegration",
    "get_lola_sentry",
    "sentry_agent_operation",
    "sentry_graph_node",
    "capture_lola_error",
    "log_agent_breadcrumb"
]
# Standard imports
import typing as tp
from abc import ABC, abstractmethod

# Third-party
try:
    from langchain.tools import BaseTool as LangChainTool
    from langchain.chains import LLMChain
except ImportError:
    raise ImportError("LangChain not installed. Run 'poetry add langchain' or equivalent.")

from sentry_sdk import capture_exception  # For error tracking

# Local
from lola.utils.config import get_config  # Assumes global config loader from Phase 3
from lola.tools.base import BaseTool  # LOLA's base tool from Phase 2
from lola.utils.logging import logger  # Structured logger from Phase 3

"""
File: Base adapter for integrating LangChain components into LOLA OS.
Purpose: Allows seamless use of LangChain's tools and chains without direct dependency, 
         enabling configuration-based toggling.
How: Provides abstract methods for wrapping LangChain tools/chains into LOLA interfaces; 
     loads config to enable/disable.
Why: Enhances LOLA's toolbox with LangChain's vast components while maintaining 
     abstraction and no vendor lock-in.
Full Path: lola-os/python/lola/libs/langchain/adapter.py
"""

class LangChainAdapter(ABC):
    """LangChainAdapter: Abstract base for wrapping LangChain into LOLA. 
    Does NOT handle LLM calls—use agnostic/unified.py."""

    def __init__(self):
        """
        Initializes the adapter, checking config for enablement.
        Does Not: Initialize LangChain-specific resources—done in subclasses.
        """
        config = get_config()
        self.enabled = config.get("use_langchain", False)
        if not self.enabled:
            logger.warning("LangChain integration disabled in config. Adapter will raise on use.")
        self._sentry_dsn = config.get("sentry_dsn", None)  # For error logging

    def _handle_error(self, exc: Exception, msg: str) -> None:
        """
        Handles exceptions with logging and Sentry capture.
        Args:
            exc: The exception raised.
            msg: Context message.
        Does Not: Retry—handled by callers if needed (e.g., agnostic/fallback.py).
        """
        logger.error(f"{msg}: {str(exc)}")
        if self._sentry_dsn:
            capture_exception(exc)

    @abstractmethod
    def wrap_tool(self, lc_tool: LangChainTool) -> BaseTool:
        """
        Abstract method to wrap a LangChain tool into LOLA BaseTool.
        Args:
            lc_tool: LangChain tool instance.
        Returns:
            LOLA BaseTool wrapper.
        """
        pass

    @abstractmethod
    def wrap_chain(self, lc_chain: LLMChain) -> tp.Callable:
        """
        Abstract method to wrap a LangChain chain into a callable for LOLA graphs.
        Args:
            lc_chain: LangChain chain instance.
        Returns:
            Callable for use in nodes.
        """
        pass


class LangChainToolAdapter(LangChainAdapter):
    """Concrete adapter for LangChain tools."""

    def wrap_tool(self, lc_tool: LangChainTool) -> BaseTool:
        """
        Wraps LangChain tool into LOLA BaseTool.
        Args:
            lc_tool: LangChain tool to wrap.
        Returns:
            LOLA BaseTool with executed wrapped.
        Does Not: Validate inputs—handled by BaseTool.
        """
        if not self.enabled:
            raise ValueError("LangChain disabled in config.")

        class WrappedTool(BaseTool):
            def __init__(self, name: str, description: str):
                super().__init__(name=name, description=description)
                self._lc_tool = lc_tool

            def execute(self, *args, **kwargs) -> tp.Any:
                try:
                    # Inline: Use LangChain's run method with error boundaries
                    return self._lc_tool.run(*args, **kwargs)
                except Exception as exc:
                    self._handle_error(exc, f"LangChain tool execution failed: {self.name}")
                    raise  # Re-raise for caller handling

        return WrappedTool(name=lc_tool.name, description=lc_tool.description)

    def wrap_chain(self, lc_chain: LLMChain) -> tp.Callable:
        """
        Wraps LangChain chain into a callable.
        Args:
            lc_chain: LangChain chain.
        Returns:
            Callable that runs the chain.
        """
        if not self.enabled:
            raise ValueError("LangChain disabled in config.")

        def wrapped_chain(input_data: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
            """
            Executes the wrapped LangChain chain.
            Args:
                input_data: Input dictionary for the chain.
            Returns:
                Result dictionary from chain execution.
            """
            try:
                # Inline: LangChain chains expect dict inputs, return dict outputs
                result = lc_chain.run(input_data)
                return {"result": result}
            except Exception as exc:
                self._handle_error(exc, "LangChain chain execution failed")
                raise

        return wrapped_chain


# Global instance for easy import
get_langchain_adapter = LangChainToolAdapter

__all__ = ["LangChainAdapter", "LangChainToolAdapter", "get_langchain_adapter"]
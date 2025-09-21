# Standard imports
import typing as tp
from typing import Optional

# Third-party
try:
    from langchain.tools import tool as lc_tool_decorator
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain.tools.base import BaseTool as LangChainTool
except ImportError:
    raise ImportError("LangChain tools not installed. Run 'poetry add langchain langchain-community' or equivalent.")

# Local
from lola.libs.langchain.adapter import get_langchain_adapter
from lola.tools.base import BaseTool
from lola.utils.config import get_config
from lola.utils.logging import logger

"""
File: Wraps LangChain tools into LOLA BaseTool format.
Purpose: Provides a bridge to use LangChain's pre-built tools (e.g., search, math) in LOLA agents.
How: Uses LangChainToolAdapter to wrap; loads config for enablement; provides common tool factory.
Why: Leverages LangChain's ecosystem while keeping LOLA's simple tool interface.
Full Path: lola-os/python/lola/libs/langchain/tools.py
"""

class LangChainToolsWrapper:
    """LangChainToolsWrapper: Utility to convert LangChain tools to LOLA format. 
    Does NOT create new tools—wraps existing."""

    def __init__(self):
        """
        Initializes the wrapper with config and adapter.
        Does Not: Load all LangChain tools upfront—lazy loading.
        """
        config = get_config()
        self.enabled = config.get("use_langchain", False)
        self.adapter = get_langchain_adapter() if self.enabled else None
        self._available_tools = self._get_available_tools()

    def _get_available_tools(self) -> tp.Dict[str, tp.Callable]:
        """
        Returns dictionary of available LangChain tools.
        Returns:
            Dict mapping tool names to factory functions.
        """
        if not self.enabled:
            return {}
        
        # Inline: Define common LangChain tools that map well to LOLA use cases
        return {
            "duckduckgo_search": self._create_duckduckgo_search,
            "python_repl": self._create_python_repl,
            "wikipedia": self._create_wikipedia_search,
        }

    def get_wrapped_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Gets a wrapped LangChain tool by name (e.g., 'duckduckgo_search').
        Args:
            tool_name: Name of LangChain tool to wrap.
        Returns:
            LOLA BaseTool or None if disabled/not found.
        Does Not: Import dynamically—uses pre-defined available tools.
        """
        if not self.enabled:
            logger.warning("LangChain tools disabled in config.")
            return None

        if tool_name not in self._available_tools:
            logger.error(f"Unknown LangChain tool requested: {tool_name}")
            return None

        try:
            # Inline: Factory pattern for tool creation - ensures consistency
            factory = self._available_tools[tool_name]
            lc_tool = factory()
            wrapped_tool = self.adapter.wrap_tool(lc_tool)
            logger.info(f"Successfully wrapped LangChain tool: {tool_name}")
            return wrapped_tool
        except Exception as exc:
            self.adapter._handle_error(exc, f"Failed to wrap LangChain tool: {tool_name}")
            return None

    def create_custom_wrapped_tool(
        self, 
        func: tp.Callable, 
        name: str, 
        description: str
    ) -> BaseTool:
        """
        Creates a custom LangChain tool from a function and wraps it.
        Args:
            func: Function to wrap as a tool.
            name: Tool name.
            description: Tool description.
        Returns:
            Wrapped LOLA BaseTool.
        Does Not: Handle function validation—caller responsibility.
        """
        if not self.enabled:
            raise ValueError("LangChain disabled in config.")

        try:
            # Inline: Use LangChain's @tool decorator to create proper tool instance
            @lc_tool_decorator
            def decorated_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            decorated_func.name = name
            decorated_func.description = description
            lc_tool = decorated_func
            
            wrapped_tool = self.adapter.wrap_tool(lc_tool)
            logger.info(f"Created custom wrapped tool: {name}")
            return wrapped_tool
        except Exception as exc:
            self.adapter._handle_error(exc, f"Failed to create custom LangChain tool: {name}")
            raise

    # Factory methods for common tools
    def _create_duckduckgo_search(self) -> LangChainTool:
        """Creates DuckDuckGo search tool."""
        return DuckDuckGoSearchRun()

    def _create_python_repl(self) -> LangChainTool:
        """Creates Python REPL tool (requires langchain_experimental)."""
        try:
            from langchain_experimental.tools import PythonREPLTool
            return PythonREPLTool()
        except ImportError:
            logger.warning("PythonREPLTool not available. Install langchain-experimental.")
            raise

    def _create_wikipedia_search(self) -> LangChainTool:
        """Creates Wikipedia search tool."""
        try:
            from langchain_community.tools import WikipediaQueryRun
            from langchain_community.utilities import WikipediaAPIWrapper
            return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        except ImportError:
            logger.warning("Wikipedia tools not available. Install langchain-community.")
            raise

    def get_all_available_tools(self) -> tp.List[str]:
        """
        Returns list of all available LangChain tool names.
        Returns:
            List of tool names that can be wrapped.
        """
        return list(self._available_tools.keys()) if self.enabled else []


# Convenience functions for easy import
def get_langchain_search_tool() -> Optional[BaseTool]:
    """Quick access to DuckDuckGo search tool."""
    wrapper = LangChainToolsWrapper()
    return wrapper.get_wrapped_tool("duckduckgo_search")

def get_langchain_python_tool() -> Optional[BaseTool]:
    """Quick access to Python REPL tool."""
    wrapper = LangChainToolsWrapper()
    return wrapper.get_wrapped_tool("python_repl")

__all__ = [
    "LangChainToolsWrapper", 
    "get_langchain_search_tool", 
    "get_langchain_python_tool"
]
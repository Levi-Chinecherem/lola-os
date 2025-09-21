# Standard imports
import typing as tp
import requests

# Local
from .base import BaseTool

"""
File: Defines the WebSearchTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to perform web searches for information retrieval.
How: Uses DuckDuckGo API for real web search results.
Why: Supports data gathering from the web, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/tools/web_search.py
Future Optimization: Migrate to Rust for faster search handling (post-TMVP 1).
"""

class WebSearchTool(BaseTool):
    """WebSearchTool: Performs web searches using DuckDuckGo API. Does NOT persist results—use StateManager."""

    name: str = "web_search"

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Execute a web search with the given query.

        Args:
            input_data: Query string for search.

        Returns:
            List of search results with titles and snippets.

        Does Not: Handle authentication—use api_client.py for authenticated searches.
        """
        if not isinstance(input_data, str):
            raise ValueError("Input data must be a string query.")
        url = f"https://api.duckduckgo.com/?q={input_data}&format=json&pretty=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results = [
                {"title": item["Title"], "snippet": item["Abstract"], "url": item["FirstURL"]}
                for item in data.get("RelatedTopics", []) if "Title" in item
            ]
            return results
        raise ValueError(f"Web search failed with status {response.status_code}")

__all__ = ["WebSearchTool"]
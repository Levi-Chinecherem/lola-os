# Standard imports
import typing as tp
import requests
import json

# Local
from .base import BaseTool

"""
File: Defines the APIClientTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to make authenticated API calls.
How: Uses requests for HTTP operations.
Why: Supports external data fetching, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/tools/api_client.py
Future Optimization: Migrate to Rust for fast HTTP (post-TMVP 1).
"""

class APIClientTool(BaseTool):
    """APIClientTool: Makes authenticated API calls. Does NOT persist responses—use StateManager."""

    name: str = "api_client"

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Make an API call.

        Args:
            input_data: Dict with 'url', 'method' (GET/POST), 'headers', 'data'.

        Returns:
            API response JSON or text.

        Does Not: Handle retries—use tenacity in perf_opt/.
        """
        if not isinstance(input_data, dict) or 'url' not in input_data:
            raise ValueError("Input data must be dict with 'url'.")
        method = input_data.get('method', 'GET').upper()
        headers = input_data.get('headers', {})
        data = input_data.get('data', None)
        if method == "GET":
            response = requests.get(input_data['url'], headers=headers, params=data)
        elif method == "POST":
            response = requests.post(input_data['url'], headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text
        raise ValueError(f"API call failed with status {response.status_code}")

__all__ = ["APIClientTool"]
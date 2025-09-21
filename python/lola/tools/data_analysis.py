# Standard imports
import typing as tp
import pandas as pd

# Local
from .base import BaseTool

"""
File: Defines the DataAnalysisTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to perform data analysis on datasets.
How: Uses pandas for real data manipulation and analysis.
Why: Supports data-driven decisions, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/tools/data_analysis.py
Future Optimization: Migrate to Rust for high-performance analysis (post-TMVP 1).
"""

class DataAnalysisTool(BaseTool):
    """DataAnalysisTool: Performs data analysis using pandas. Does NOT persist data—use StateManager."""

    name: str = "data_analysis"

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Execute data analysis with the given operation and dataset.

        Args:
            input_data: Dict with 'operation' (e.g., "mean") and 'data' (list of dicts).

        Returns:
            Analysis result.

        Does Not: Handle file I/O—use file_io.py.
        """
        if not isinstance(input_data, dict) or 'operation' not in input_data or 'data' not in input_data:
            raise ValueError("Input data must be dict with 'operation' and 'data'.")
        df = pd.DataFrame(input_data['data'])
        operation = input_data['operation']
        if operation == "mean":
            return df.mean().to_dict()
        elif operation == "describe":
            return df.describe().to_dict()
        raise ValueError(f"Unsupported operation: {operation}")

__all__ = ["DataAnalysisTool"]
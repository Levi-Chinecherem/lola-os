# Standard imports
import typing as tp
from pathlib import Path

# Local
from .base import BaseTool

"""
File: Defines the FileIOTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to read/write files.
How: Uses Pathlib for file operations.
Why: Supports file-based tasks, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/tools/file_io.py
Future Optimization: Migrate to Rust for secure file handling (post-TMVP 1).
"""

class FileIOTool(BaseTool):
    """FileIOTool: Handles file read/write operations. Does NOT persist state—use StateManager."""

    name: str = "file_io"

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Perform file I/O operation.

        Args:
            input_data: Dict with 'operation' (read/write), 'path', and 'content' (for write).

        Returns:
            File content (for read) or success message (for write).

        Does Not: Handle permissions—use guardrails/tool_permissions.py.
        """
        if not isinstance(input_data, dict) or 'operation' not in input_data or 'path' not in input_data:
            raise ValueError("Input data must be dict with 'operation' and 'path'.")
        path = Path(input_data['path'])
        operation = input_data['operation']
        if operation == "read":
            if path.exists():
                return path.read_text()
            raise FileNotFoundError(f"File {path} not found.")
        elif operation == "write":
            content = input_data.get('content', '')
            path.write_text(content)
            return "File written successfully."
        raise ValueError(f"Unsupported operation: {operation}")

__all__ = ["FileIOTool"]
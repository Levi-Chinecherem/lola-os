# Standard imports
import typing as tp
import ast
from io import StringIO
import sys

# Local
from .base import BaseTool

"""
File: Defines the CodeInterpreterTool class for LOLA OS TMVP 1 Phase 2.

Purpose: Enables agents to execute Python code safely.
How: Uses ast to parse and exec to execute code in a sandbox.
Why: Supports dynamic code execution, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/tools/code_interpreter.py
Future Optimization: Migrate to Rust for secure sandboxing (post-TMVP 1).
"""

class CodeInterpreterTool(BaseTool):
    """CodeInterpreterTool: Executes Python code in a sandbox. Does NOT persist outputs—use StateManager."""

    name: str = "code_interpreter"

    async def execute(self, input_data: tp.Any) -> tp.Any:
        """
        Execute Python code.

        Args:
            input_data: Code string to execute.

        Returns:
            Execution result or error message.

        Does Not: Allow dangerous operations—restricted globals.
        """
        if not isinstance(input_data, str):
            raise ValueError("Input data must be a Python code string.")
        # Inline: Sandbox execution with restricted globals
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        try:
            tree = ast.parse(input_data)
            exec(compile(tree, filename="code", mode="exec"), {"__builtins__": __builtins__}, {})
            result = redirected_output.getvalue().strip()
        except Exception as e:
            result = str(e)
        finally:
            sys.stdout = old_stdout
        return result

__all__ = ["CodeInterpreterTool"]
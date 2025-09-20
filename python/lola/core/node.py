# Standard imports
import typing as tp
from dataclasses import dataclass

"""
File: Defines Node and Edge dataclasses for LOLA OS TMVP 1.

Purpose: Provides core components for the StateGraph workflow engine.
How: Defines Node (task unit) and Edge (transition) dataclasses for graph construction.
Why: Separates graph structure definitions for clarity, per Single Responsibility Principle.
Full Path: lola-os/python/lola/core/node.py
Future Optimization: Migrate to Rust for memory-efficient graph structures (post-TMVP 1).
"""

@dataclass
class Node:
    """Represents a unit of work in the graph (e.g., LLM call, tool call)."""
    id: str
    type: str  # "llm", "tool", "logic"
    function: tp.Callable
    description: str = ""

@dataclass
class Edge:
    """Defines a transition between nodes."""
    source: str
    target: str
    condition: tp.Optional[tp.Callable] = None
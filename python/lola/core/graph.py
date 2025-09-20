# Standard imports
import typing as tp
from dataclasses import dataclass

# Local
from .state import State

"""
File: Defines the StateGraph, Node, and Edge classes for LOLA OS TMVP 1.

Purpose: Provides the orchestration engine for agent workflows.
How: Implements a directed acyclic graph (DAG) with nodes (tasks) and edges (transitions).
Why: Enables explicit, traceable workflows, per Radical Reliability and Explicit over Implicit.
Full Path: lola-os/python/lola/core/graph.py
Future Optimization: Migrate to Rust for parallel node execution via Tokio (post-TMVP 1).
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
    """Defines a transition between nodes with traversal logic."""
    source: str
    target: str
    condition: tp.Optional[tp.Callable] = None

    def can_traverse(self, state: tp.Any) -> bool:
        """
        Check if the edge can be traversed based on the condition.

        Args:
            state: Current state.
        Returns:
            bool: True if traversable, False otherwise.
        """
        if self.condition is None:
            return True
        return self.condition(state)

class StateGraph:
    """Main class for building and executing agent workflows."""

    def __init__(self, state: State):
        """
        Initialize the graph with an initial state.

        Args:
            state: Initial State object.
        """
        self.state = state
        self.nodes: tp.Dict[str, Node] = {}
        self.edges: tp.List[Edge] = []
        self.current_node: tp.Optional[str] = None

    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph.

        Args:
            node: Node instance to add.
        """
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the graph.

        Args:
            edge: Edge instance defining transition.
        """
        self.edges.append(edge)

    async def execute(self) -> State:
        """
        Execute the graph until completion.

        Returns:
            State: Final state after execution.
        Does Not: Persist state—use StateManager.
        """
        # Inline: Why loop? Ensures explicit node traversal per Explicit over Implicit.
        while self.current_node:
            node = self.nodes[self.current_node]
            result = await node.function(self.state)
            self.state.update(result)
            self.current_node = self._next_node()
        return self.state

    def _next_node(self) -> tp.Optional[str]:
        """Determine the next node based on edges and conditions."""
        for edge in self.edges:
            if edge.source == self.current_node and edge.can_traverse(self.state):
                return edge.target
        return None
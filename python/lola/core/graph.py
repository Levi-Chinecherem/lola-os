import asyncio
import typing as tp
from dataclasses import dataclass
from .state import State

"""
File: Defines the StateGraph, Node, Edge, and DynamicGraphBuilder for LOLA OS workflow orchestration.

Purpose: Provides a DAG-based engine for executing agent tasks with parallel and conditional logic.
How: StateGraph manages nodes (tasks) and edges (transitions), with asyncio for concurrency and DynamicGraphBuilder for runtime modifications.
Why: Enables explicit, traceable workflows per LOLA's Explicit over Implicit tenet, with flexibility for dynamic tasks.
Full Path: lola-os/python/lola/core/graph.py
"""

@dataclass
class Node:
    """Node: A unit of work in the graph (LLM call, tool call, or logic)."""
    id: str
    action: tp.Callable[[State], tp.Awaitable[State]]
    type: str  # "llm", "tool", "logic"

@dataclass
class Edge:
    """Edge: Defines transitions between nodes with optional conditions."""
    source: str
    target: str
    condition: tp.Optional[tp.Callable[[State], bool]] = None

class StateGraph:
    """StateGraph: Orchestrates a DAG of nodes and edges for agent execution."""

    def __init__(self, state_type: tp.Type[State]):
        """
        Initialize StateGraph with a state type.

        Args:
            state_type: Pydantic State class for validation.

        Does Not: Execute tasks—use run().
        """
        self.nodes: tp.Dict[str, Node] = {}
        self.edges: tp.List[Edge] = []
        self.state_type = state_type
        self.current_state = state_type()

    def add_node(self, node: Node) -> None:
        """
        Adds a node to the graph.

        Args:
            node: Node instance to add.

        Does Not: Validate node—caller must ensure uniqueness.
        """
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """
        Adds an edge to the graph.

        Args:
            edge: Edge instance to add.

        Does Not: Check for cycles—handled in Phase 2.
        """
        self.edges.append(edge)

    async def run(self) -> State:
        """
        Executes the graph asynchronously, updating state.

        Returns:
            Final State after execution.

        Does Not: Handle errors—caller must wrap in try/except.
        """
        current_node_id = list(self.nodes.keys())[0]  # Start with first node
        while current_node_id:
            node = self.nodes[current_node_id]
            # Inline: Execute node action and update state
            self.current_state = await node.action(self.current_state)
            # Find next node based on edges and conditions
            next_node_id = None
            for edge in self.edges:
                if edge.source == current_node_id:
                    if edge.condition is None or edge.condition(self.current_state):
                        next_node_id = edge.target
                        break
            current_node_id = next_node_id
        return self.current_state

    def is_complete(self) -> bool:
        """Checks if the graph has completed execution."""
        return not any(edge.source in self.nodes for edge in self.edges)

class DynamicGraphBuilder:
    """DynamicGraphBuilder: Allows runtime graph modifications (MVP 2, Item 8)."""

    def __init__(self, graph: StateGraph):
        """
        Initialize with a StateGraph.

        Args:
            graph: StateGraph to modify.

        Does Not: Persist changes—use StateManager in memory.py.
        """
        self.graph = graph

    def add_dynamic_node(self, node_id: str, action: tp.Callable[[State], tp.Awaitable[State]], type: str) -> None:
        """
        Adds a node dynamically at runtime.

        Args:
            node_id: Unique node identifier.
            action: Async callable for node action.
            type: Node type ("llm", "tool", "logic").

        Does Not: Validate action—caller must ensure correctness.
        """
        self.graph.add_node(Node(id=node_id, action=action, type=type))

    def add_dynamic_edge(self, source: str, target: str, condition: tp.Optional[tp.Callable[[State], bool]] = None) -> None:
        """
        Adds an edge dynamically at runtime.

        Args:
            source: Source node ID.
            target: Target node ID.
            condition: Optional condition for transition.

        Does Not: Check for cycles—handled in Phase 2.
        """
        self.graph.add_edge(Edge(source=source, target=target, condition=condition))

__all__ = ["StateGraph", "Node", "Edge", "DynamicGraphBuilder"]
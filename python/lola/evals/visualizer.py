# Standard imports
import typing as tp
from graphviz import Digraph

# Local
from lola.core.graph import StateGraph

"""
File: Defines the GraphVisualizer class for LOLA OS TMVP 1 Phase 2.

Purpose: Visualizes agent graphs for debugging.
How: Uses graphviz to generate DOT graphs from StateGraph.
Why: Aids in understanding workflows, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/evals/visualizer.py
Future Optimization: Migrate to Rust for fast visualization (post-TMVP 1).
"""

class GraphVisualizer:
    """GraphVisualizer: Visualizes StateGraph structures. Does NOT persist graphs—use StateManager."""

    def visualize(self, graph: StateGraph, output_file: str = "graph.dot") -> None:
        """
        Generates a DOT graph visualization.

        Args:
            graph: StateGraph instance.
            output_file: Output DOT file path.

        Returns:
            None (saves to file).

        Does Not: Render image—use graphviz tools.
        """
        dot = Digraph()
        for node in graph.nodes.values():
            dot.node(node.id, label=f"{node.id} ({node.type})")
        for edge in graph.edges:
            label = "conditional" if edge.condition else ""
            dot.edge(edge.source, edge.target, label=label)
        dot.render(output_file, view=False)

__all__ = ["GraphVisualizer"]
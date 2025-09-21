# Standard imports
import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock
from pathlib import Path
import json
import typing as tp

# Third-party
import litellm

# Local
from lola.core.agent import BaseAgent
from lola.core.graph import StateGraph, Node, Edge, DynamicGraphBuilder
from lola.core.state import State
from lola.core.memory import StateManager, ConversationMemory, EntityMemory

"""
File: Comprehensive tests for LOLA OS core components in Phase 1.

Purpose: Validates BaseAgent, StateGraph, State, StateManager, ConversationMemory, and EntityMemory with real functionality and mocks for external calls.
How: Uses pytest with async support, unittest.mock for LLM calls, and test data for persistence.
Why: Ensures core reliability with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_core.py
"""

@pytest.mark.asyncio
async def test_base_agent(mocker):
    """Test BaseAgent initialization and LLM call."""
    mocker.patch('litellm.completion', return_value={'choices': [{'message': {'content': 'Test response'}}]})
    class TestAgent(BaseAgent):
        async def run(self, query: str) -> State:
            self.state.update(query=query)
            self.state.output = self.call_llm(query)
            return self.state

    agent = TestAgent(model="test/model")
    state = await agent.run("Test query")
    assert state.query == "Test query"
    assert state.output == "Test response"

@pytest.mark.asyncio
async def test_state_graph():
    """Test StateGraph execution with nodes and edges."""
    async def node1_action(state: State) -> State:
        state.output = "Node 1 executed"
        return state

    async def node2_action(state: State) -> State:
        state.output += " -> Node 2 executed"
        return state

    graph = StateGraph(State)
    graph.add_node(Node(id="node1", action=node1_action, type="logic"))
    graph.add_node(Node(id="node2", action=node2_action, type="logic"))
    graph.add_edge(Edge(source="node1", target="node2"))
    state = await graph.run()
    assert state.output == "Node 1 executed -> Node 2 executed"

def test_state():
    """Test State validation and update."""
    state = State(query="Test", history=[{"role": "user", "content": "Hi"}])
    state.update(output="Result")
    assert state.query == "Test"
    assert state.output == "Result"
    # Inline: Test invalid data
    with pytest.raises(ValueError):
        State(query=123)  # Type error

def test_state_manager(tmp_path):
    """Test StateManager persistence."""
    storage_path = tmp_path / "test_state.json"
    manager = StateManager(str(storage_path))
    state = State(query="Test", output="Result")
    manager.save_state(state)
    loaded = manager.load_state()
    assert loaded.query == "Test"
    assert loaded.output == "Result"

def test_conversation_memory():
    """Test ConversationMemory management."""
    memory = ConversationMemory()
    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi")
    assert len(memory.history) == 2
    assert memory.get_context() == "user: Hello\nassistant: Hi"

@pytest.mark.asyncio
async def test_entity_memory(mocker):
    """Test EntityMemory extraction with mocked LLM."""
    mocker.patch('litellm.completion', return_value={'choices': [{'message': {'content': "Alice, London"}}]})
    memory = EntityMemory()
    await memory.extract_entities("Alice lives in London", model="test/model")
    assert "Alice" in memory.entities
    assert "London" in memory.entities

@pytest.mark.asyncio
async def test_dynamic_graph_builder():
    """Test DynamicGraphBuilder for runtime modifications."""
    async def action(state: State) -> State:
        state.output = "Dynamic node"
        return state

    graph = StateGraph(State)
    builder = DynamicGraphBuilder(graph)
    builder.add_dynamic_node("dynamic", action, "logic")
    builder.add_dynamic_edge("start", "dynamic")
    state = await graph.run()
    assert state.output == "Dynamic node"

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()
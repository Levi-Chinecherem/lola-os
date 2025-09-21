# Standard imports
import pytest
import asyncio
import time
import json
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import os

# Third-party
import litellm
from tenacity import retry

# Local
from lola.core.agent import BaseAgent
from lola.core.graph import StateGraph, Node, Edge, DynamicGraphBuilder
from lola.core.state import State
from lola.core.memory import StateManager, ConversationMemory, EntityMemory, StateDB

"""
File: Comprehensive tests for LOLA OS core components in Phase 1.

Purpose: Validates BaseAgent, StateGraph, State, StateManager, ConversationMemory, and EntityMemory with real functionality and mocks for external calls.
How: Uses pytest with async support, unittest.mock for LLM calls, and test data for persistence (including multi-backend).
Why: Ensures core reliability with >80% coverage, per Radical Reliability tenet.
Full Path: lola-os/tests/test_core.py
"""

@pytest.fixture
def temp_config():
    """Fixture for temp config with different backends."""
    return {"backend": "json", "path": "temp_state.json"}

@pytest.fixture
def temp_path():
    """Temp path for JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "state.json"

@pytest.mark.asyncio
async def test_base_agent(mocker, temp_config):
    """Test BaseAgent initialization and LLM call with retry."""
    mock_response = {'choices': [{'message': {'content': 'Test response'}}]}
    mocker.patch('litellm.completion', side_effect=[Exception("Retry"), mock_response])  # Simulate retry
    class TestAgent(BaseAgent):
        async def run(self, query: str) -> State:
            self.state.update(query=query)
            self.state.output = self.call_llm(query)
            return self.state

    agent = TestAgent(model="test/model")
    state = await agent.run("Test query")
    assert state.query == "Test query"
    assert state.output == "Test response"
    litellm.completion.assert_called_with(model="test/model", messages=[{"role": "user", "content": "Test query"}], max_tokens=1000)

@pytest.mark.asyncio
async def test_state_graph():
    """Test StateGraph execution with nodes and edges, including parallel."""
    async def node1_action(state: State) -> State:
        state.output = "Node 1 executed"
        return state

    async def node2_action(state: State) -> State:
        state.output += " -> Node 2 executed"
        return state

    async def node3_action(state: State) -> State:
        state.metadata["parallel"] = "Node 3 executed"
        return state

    graph = StateGraph(State)
    graph.add_node(Node(id="node1", action=node1_action, type="logic"))
    graph.add_node(Node(id="node2", action=node2_action, type="logic"))
    graph.add_node(Node(id="node3", action=node3_action, type="logic"))
    graph.add_edge(Edge(source="node1", target="node2"))
    graph.add_edge(Edge(source="node1", target="node3"))  # Parallel
    state = await graph.run()
    assert "Node 1 executed -> Node 2 executed" in state.output
    assert state.metadata["parallel"] == "Node 3 executed"

def test_state():
    """Test State validation and update."""
    state = State(query="Test", history=[{"role": "user", "content": "Hi"}])
    state.update(output="Result", extra="Value")
    assert state.query == "Test"
    assert state.output == "Result"
    assert state.metadata["extra"] == "Value"
    # Inline: Test invalid data
    with pytest.raises(ValueError):
        State(query=123)  # Type error

    # Test serialization
    json_str = state.to_json()
    loaded = State.from_json(json_str)
    assert loaded.query == "Test"

def test_state_manager_json(temp_path, temp_config):
    """Test StateManager with JSON backend."""
    temp_config["path"] = str(temp_path)
    manager = StateManager(temp_config)
    state = State(query="Test", output="Result")
    manager.save_state(state, key="test_key")
    loaded = manager.load_state(key="test_key")
    assert loaded.query == "Test"
    assert loaded.output == "Result"
    # Test non-existent key
    empty = manager.load_state(key="missing")
    assert empty.query == ""

@pytest.mark.parametrize("backend", ["sqlalchemy", "redis"])
def test_state_manager_alternative_backends(backend, mocker):
    """Test StateManager with SQLAlchemy and Redis (mocked)."""
    config = {"backend": backend}
    if backend == "redis":
        mocker.patch('redis.Redis', return_value=MagicMock(get=lambda k: json.dumps({"query": "Test"}), set=lambda k, v: None))
    elif backend == "sqlalchemy":
        mocker.patch('sqlalchemy.orm.sessionmaker', return_value=MagicMock(query=lambda: MagicMock(filter_by=lambda k: MagicMock(first=lambda: MagicMock(data={"query": "Test"})))))

    manager = StateManager(config)
    state = State(query="Test")
    manager.save_state(state, key="test")
    loaded = manager.load_state(key="test")
    assert loaded.query == "Test"

def test_conversation_memory():
    """Test ConversationMemory management."""
    memory = ConversationMemory()
    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi")
    assert len(memory.history) == 2
    assert memory.get_context() == "user: Hello\nassistant: Hi"

@pytest.mark.asyncio
async def test_entity_memory(mocker):
    """Test EntityMemory extraction with mocked LLM and retry."""
    mock_response = {'choices': [{'message': {'content': "Alice, London"}}]}
    mocker.patch('litellm.completion', side_effect=[Exception("Retry"), mock_response])
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
    builder.add_dynamic_edge("start", "dynamic")  # Assume start node added implicitly
    graph.add_node(Node(id="start", action=lambda s: s, type="logic"))  # For testing
    state = await graph.run()
    assert state.output == "Dynamic node"

# Coverage check: Run with pytest-cov to verify >80%

if __name__ == "__main__":
    pytest.main(["-v"])
# LOLA OS TMVP 1

LOLA OS (Layered Orchestration for Logic and Automation, or Live Onchain Logical Agents) is an open-source Python framework for building read-only AI agents with Ethereum Virtual Machine (EVM)-native capabilities. TMVP 1 delivers a modular SDK (`pip install lola-os`) with core abstractions, agent templates, tools, chains, CLI, utilities, examples, and documentation, designed for developer onboarding and extensibility. Built with the principles of Developer Sovereignty, EVM-Native, Radical Reliability, Choice by Design, and Explicit over Implicit, LOLA OS enables developers to create AI agents for on-chain and off-chain tasks. This README reflects the completed mocked prototype (Phases 1-4), with all 21 tests passing, ready for full development in TMVP 2.

## Purpose

LOLA OS TMVP 1 provides a foundation for AI agent development with:
- **Modular Architecture**: Core abstractions (`agent.py`, `graph.py`, `state.py`, `memory.py`) for flexible agent design.
- **Agent Templates**: ReAct, PlanExecute, and Collaborative agents for diverse use cases.
- **EVM-Native Tools**: Read-only contract calls, event listeners, and oracle fetching via `web3.py`.
- **Advanced Features**: Retrieval-Augmented Generation (RAG), multi-agent orchestration, human-in-the-loop (HITL), guardrails, evaluations, LLM-agnostic interfaces, and performance optimizations.
- **CLI Tools**: Commands for project creation (`create`), execution (`run`), building (`build`), and deployment (`deploy`).
- **Developer Experience**: Example agents (`research_agent`, `onchain_analyst`) and comprehensive documentation for onboarding.
- **Testing**: A robust test suite with 21 passing tests covering all components.

The mocked prototype (Phases 1-4) validates the architecture, setting the stage for TMVP 2 to implement real EVM calls, Pinecone integration, and production deployment.

## Project Structure

```
lola-os/
├── python/
│   ├── lola/
│   │   ├── core/                   # Core abstractions (agent.py, graph.py, state.py, memory.py)
│   │   ├── agents/                 # Agent templates (react.py, plan_execute.py, collaborative.py)
│   │   ├── tools/                  # Tools (web_search.py, contract_call.py, human_input.py)
│   │   ├── chains/                 # EVM read abstractions (connection.py, contract.py)
│   │   ├── rag/                    # Retrieval-Augmented Generation (multimodal.py)
│   │   ├── guardrails/             # Safety and permissions (validator.py)
│   │   ├── evals/                  # Agent evaluation suite
│   │   ├── orchestration/          # Multi-agent coordination (orchestrator.py)
│   │   ├── hitl/                   # Human-in-the-loop workflows
│   │   ├── agnostic/               # LLM-agnostic interfaces
│   │   ├── perf_opt/               # Performance optimizations
│   │   ├── specialization/         # Agent fine-tuning
│   │   ├── cli/                    # CLI commands (create.py, run.py, build.py, deploy.py)
│   │   ├── utils/                  # Utilities (logging.py, config.py, telemetry.py)
├── tests/                          # Pytest suite (test_core.py, test_agents.py, test_examples.py, etc.)
├── examples/
│   ├── research_agent/             # ReAct-based web research agent
│   │   ├── agent.py
│   │   ├── config.yaml
│   │   ├── README.md
│   ├── onchain_analyst/            # EVM-based NFT floor price analyst
│   │   ├── agent.py
│   │   ├── config.yaml
│   │   ├── README.md
├── docs/                           # Documentation (Sphinx-compatible)
│   ├── concepts/                   # Core concepts (core.md, agents.md, tools.md)
│   ├── tutorials/                  # Guides (build_your_first_agent.md, building_onchain_tools.md)
│   ├── index.md                    # Documentation entry point
│   ├── quickstart.md               # Installation and first agent guide
├── documentation/
│   ├── tmvp1-development-plan.md   # TMVP 1 blueprint
│   ├── changelog.md                # Release notes
├── pyproject.toml                  # Poetry configuration
├── README.md                       # This file
├── setup_phase4.sh                 # Setup script for Phase 4
```

## Installation

LOLA OS requires **Python 3.10+** and uses [Poetry](https://python-poetry.org/) for dependency management.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/xai/lola-os
   cd lola-os
   ```

2. **Set Up PYTHONPATH**:
   ```bash
   export PYTHONPATH=$(pwd)/python:$PYTHONPATH
   ```

3. **Install Dependencies**:
   ```bash
   poetry install
   ```
   If timeouts occur:
   ```bash
   poetry config http-request-timeout 60
   poetry install --verbose
   ```

4. **Verify Installation**:
   ```bash
   poetry run pytest tests/ -v
   ```
   Expected output:
   ```
   collected 21 items
   tests/test_agents.py .       [  4%]
   tests/test_chains.py .       [  9%]
   tests/test_cli.py ....       [ 28%]
   tests/test_core.py .         [ 33%]
   tests/test_evals.py .        [ 38%]
   tests/test_examples.py ..    [ 47%]
   tests/test_guardrails.py .   [ 52%]
   tests/test_hitl.py .         [ 57%]
   tests/test_orchestration.py .[ 61%]
   tests/test_perf_opt.py .     [ 66%]
   tests/test_rag.py .          [ 71%]
   tests/test_specialization.py .[ 76%]
   tests/test_tools.py .        [ 80%]
   tests/test_utils.py ....     [100%]
   21 passed in 7.70s
   ```

## CLI Usage

The `lola` CLI simplifies agent development and execution:

- **Create a Project**:
  ```bash
  poetry run lola create my_project
  cd my_project
  poetry install
  ```

- **Run an Agent**:
  ```bash
  poetry run lola run my_project.agent.BasicAgent "Test query"
  ```

- **Build for Deployment**:
  ```bash
  poetry run lola build my_project
  ```

- **Deploy (Stub)**:
  ```bash
  poetry run lola deploy build --target docker
  ```

## Examples

LOLA OS includes two example agents to demonstrate its capabilities:

1. **Research Agent** (`examples/research_agent/`):
   A ReAct-based agent for web research using `WebSearchTool` and `HumanInputTool`.
   ```bash
   cd examples/research_agent
   poetry install
   poetry run python agent.py "Research the latest AI trends"
   ```
   Configure `config.yaml` with a valid API key:
   ```yaml
   model: openai/gpt-4o
   api_key: your-api-key-here
   ```

2. **Onchain Analyst** (`examples/onchain_analyst/`):
   A ReAct-based agent for reading NFT floor prices using `ContractCallTool`.
   ```bash
   cd examples/onchain_analyst
   poetry install
   poetry run python agent.py "Get the floor price of Bored Ape Yacht Club"
   ```
   Configure `config.yaml` with a valid RPC URL, API key, and contract details:
   ```yaml
   model: openai/gpt-4o
   rpc_url: https://mainnet.infura.io/v3/your-infura-key
   api_key: your-api-key-here
   contract_address: "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"  # Bored Ape Yacht Club
   abi: '[{"constant":true,"inputs":[],"name":"getFloorPrice","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"}]'
   ```
   Note: The ABI is illustrative; replace with the actual contract ABI for real use.

## Documentation

Comprehensive documentation is available in the `docs/` directory, built with Sphinx:
- **Quickstart** (`docs/quickstart.md`): Install and run your first agent.
- **Concepts** (`docs/concepts/`): Core abstractions, agents, tools, and more.
- **Tutorials** (`docs/tutorials/`): Guides for building ReAct agents and EVM tools.
- **Index** (`docs/index.md`): Entry point for all documentation.

Build and view the documentation:
```bash
cd docs
poetry run sphinx-build -b html . _build
```
Open `docs/_build/index.html` in a browser.

## Dependencies

- **Core**: `litellm (^1.0)`, `pydantic (^2.0)`, `web3 (^6.0)`, `websockets (^13.0)`, `click (^8.1)`, `pyyaml (^6.0)`, `tenacity (^8.0)`
- **Documentation**: `sphinx (^7.0)`, `sphinx-rtd-theme (^2.0)`, `babel (^2.17)`
- **Dev**: `pytest (^8.0)`, `pytest-asyncio (^0.24.0)`

## Running Tests

The test suite covers all components:
```bash
poetry run pytest tests/ -v
```
All 21 tests pass, validating core functionality, CLI, utilities, and examples. Tests are organized in `tests/`:
- `test_core.py`: Core abstractions
- `test_agents.py`: Agent templates
- `test_tools.py`: Tool execution
- `test_chains.py`: EVM abstractions
- `test_rag.py`: RAG components
- `test_guardrails.py`: Safety features
- `test_evals.py`: Evaluations
- `test_orchestration.py`: Multi-agent coordination
- `test_hitl.py`: Human-in-the-loop
- `test_perf_opt.py`: Performance optimizations
- `test_specialization.py`: Agent specialization
- `test_cli.py`: CLI commands
- `test_utils.py`: Utilities
- `test_examples.py`: Example agents

## Usage Example

Create and run a custom ReAct agent:
```python
from lola.agents.react import ReActAgent
from lola.tools.human_input import HumanInputTool
from lola.core.state import State

agent = ReActAgent(tools=[HumanInputTool()], model="openai/gpt-4o")
state = await agent.run("Test query")
print(state.data)  # {'result': 'Stubbed response for: Test query'}
```

## Development Progress

### Phases 1-4 (Mocked Prototype, Completed)
- **Phase 1: Core (Weeks 1-4)**:
  - Implemented `core/agent.py`, `core/graph.py`, `core/state.py`, `core/memory.py`.
  - Tests: `test_core.py` (1 test).
- **Phase 2: Components (Weeks 5-8)**:
  - Added agents (`react.py`, `plan_execute.py`, `collaborative.py`), tools (`web_search.py`, `contract_call.py`, etc.), chains (`connection.py`, `contract.py`), RAG (`multimodal.py`), guardrails, evals, orchestration, HITL, LLM-agnostic interfaces, performance optimizations, and specialization.
  - Tests: `test_agents.py`, `test_tools.py`, `test_chains.py`, `test_rag.py`, `test_guardrails.py`, `test_evals.py`, `test_orchestration.py`, `test_hitl.py`, `test_perf_opt.py`, `test_specialization.py` (10 tests).
- **Phase 3: Production Envelope (Weeks 9-12)**:
  - Implemented CLI (`cli/create.py`, `cli/run.py`, `cli/build.py`, `cli/deploy.py`) and utilities (`utils/logging.py`, `utils/config.py`, `utils/telemetry.py`).
  - Tests: `test_cli.py`, `test_utils.py` (8 tests).
- **Phase 4: Developer Experience (Weeks 13-16)**:
  - Added examples (`research_agent`, `onchain_analyst`) and documentation (`docs/`).
  - Tests: `test_examples.py` (2 tests).
- **Status**: All 21 tests pass, confirming a stable mocked prototype.

### TMVP 2: Full Development (Planned)
TMVP 2 will replace mocks with production-ready implementations:
- **EVM-Native Tools**: Real contract calls (`contract_call.py`), event listeners (`event_listener.py`), and oracle fetching (`oracle_fetch.py`) using `web3.py`.
- **RAG with Pinecone**: Implement `rag/multimodal.py` with Pinecone for vector storage.
- **Deployment**: Add FastAPI and Docker support in `cli/deploy.py`.
- **Agent Enhancements**: Real LLM calls in `agents/react.py` and multi-agent workflows in `orchestration/orchestrator.py`.
- **Timeline**: October–November 2025.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request.

## Troubleshooting

- **Test Failures**:
  - Verify `PYTHONPATH`: `echo $PYTHONPATH` (should include `/path/to/lola-os/python`).
  - Check dependencies: `poetry show`.
  - Run specific tests: `poetry run pytest tests/test_examples.py -v`.
- **Dependency Issues**:
  - Retry with increased timeout: `poetry config http-request-timeout 60 && poetry install --verbose`.
  - Use a PyPI mirror: `poetry config repositories.tsinghua https://pypi.tuna.tsinghua.edu.cn/simple`.
- **Documentation Build**:
  - Ensure `sphinx` and `babel` are installed: `poetry show sphinx babel`.
  - Debug: `cd docs && poetry run sphinx-build -b html . _build -v`.

## License

MIT License (TBD).

## Contact

Levi Chinecherem Chidi <lchinecherem2018@gmail.com>

## Next Steps

- **Release v1.0**:
  - Update `documentation/changelog.md` with release notes.
  - Publish to PyPI:
    ```bash
    poetry build
    poetry publish
    ```
  - Push to GitHub and promote for 10,000+ stars.
- **TMVP 2**:
  - Implement real EVM calls, Pinecone integration, and deployment.
  - Update tests and documentation for production use.

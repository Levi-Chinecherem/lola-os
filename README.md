# LOLA OS TMVP 1

LOLA OS (Layered Orchestration for Logic and Automation, or Live Onchain Logical Agents) is an open-source Python framework for building AI agents with Ethereum Virtual Machine (EVM)-native capabilities. TMVP 1 delivers a production-ready modular SDK (`pip install lola-os`) with core abstractions, agent templates, tools, chains, CLI, utilities, examples, and documentation, designed for developer onboarding and extensibility. Built with the principles of Developer Sovereignty, EVM-Native, Radical Reliability, Choice by Design, and Explicit over Implicit, LOLA OS enables developers to create AI agents for on-chain and off-chain tasks. This README reflects the completed Phases 1-4, transitioning from a mocked prototype to production-ready implementations, with a robust test suite (21 tests) ready for validation after the upcoming Phase 5.

## Purpose

LOLA OS TMVP 1 provides a production-ready foundation for AI agent development with:
- **Modular Architecture**: Core abstractions (`agent.py`, `graph.py`, `state.py`, `memory.py`) using `pydantic` and `asyncio` for flexible, concurrent agent design.
- **Agent Templates**: ReAct, PlanExecute, and Collaborative agents for diverse use cases, powered by `litellm` for real LLM calls.
- **EVM-Native Tools**: Real contract calls (`contract_call.py`), event listeners (`event_listener.py`), and oracle fetching (`oracle_fetch.py`) using `web3.py`.
- **Advanced Features**: Retrieval-Augmented Generation (RAG) with Pinecone (`rag/multimodal.py`), multi-agent orchestration, human-in-the-loop (HITL), guardrails, evaluations, LLM-agnostic interfaces, and performance optimizations.
- **CLI Tools**: Production-ready commands (`create`, `run`, `build`, `deploy`) using `click` for project scaffolding, execution, packaging, and local deployment.
- **Developer Experience**: Runnable example agents (`research_agent`, `onchain_analyst`) and comprehensive Sphinx documentation for onboarding.
- **Utilities**: Structured logging (`logging.py`), configuration management (`config.py` with `pyyaml`), and telemetry (`telemetry.py` with `opentelemetry`) for production reliability.

Phases 1-4 have replaced mocks with production-ready code, integrating real EVM reads, Pinecone RAG, and LLM calls. Phase 5 (to be defined) will precede comprehensive testing to ensure integration and reliability.

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
│   │   ├── guardrails/             # Safety and permissions (content_safety.py)
│   │   ├── evals/                  # Agent evaluation suite (benchmarker.py)
│   │   ├── orchestration/          # Multi-agent coordination (swarm.py)
│   │   ├── hitl/                   # Human-in-the-loop workflows (approval.py)
│   │   ├── agnostic/               # LLM-agnostic interfaces (unified.py)
│   │   ├── perf_opt/               # Performance optimizations (prompt_compressor.py)
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
├── docs/                           # Sphinx documentation
│   ├── concepts/                   # Core concepts (core.md)
│   ├── tutorials/                  # Guides (build_your_first_agent.md, building_onchain_tools.md)
│   ├── index.md                    # Documentation entry point
│   ├── quickstart.md               # Installation and first agent guide
├── documentation/
│   ├── tmvp1_exec.md               # Execution plan for production
│   ├── changelog.md                # Release notes
├── pyproject.toml                  # Poetry configuration
├── README.md                       # This file
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
   poetry run pip install llama-index-vector-stores-pinecone==0.11.0
   ```
   If timeouts occur:
   ```bash
   export POETRY_REQUESTS_TIMEOUT=60
   poetry install --verbose
   ```

4. **Set Environment Variables**:
   ```bash
   export PINECONE_API_KEY="your-pinecone-api-key"
   export OPENAI_API_KEY="your-openai-api-key"
   export WEB3_PROVIDER_URI="https://sepolia.infura.io/v3/your-infura-key"
   export REDIS_URL="redis://localhost:6379/0"
   ```

5. **Verify Installation** (Testing Planned Post-Phase 5):
   ```bash
   poetry run pytest tests/ -v --cov=lola --cov-report=html --cov-report=term-missing
   ```
   Expected: 21 tests covering core, agents, tools, chains, RAG, CLI, utilities, and examples, to be validated after Phase 5.

## CLI Usage

The production-ready `lola` CLI simplifies agent development and execution:

- **Create a Project**:
  ```bash
  poetry run lola create my_project --template react
  cd my_project
  poetry install
  ```

- **Run an Agent**:
  ```bash
  poetry run lola run agents/main.py "Test query"
  ```

- **Build for Deployment**:
  ```bash
  poetry run lola build .
  ```

- **Deploy Locally (Docker Stub)**:
  ```bash
  poetry run lola deploy .
  ```

## Examples

LOLA OS includes two production-ready example agents:

1. **Research Agent** (`examples/research_agent/`):
   A ReAct-based agent using `WebSearchTool` (DuckDuckGo/SearXNG) and `MultiModalRetriever` (Pinecone) for web and RAG queries.
   ```bash
   cd examples/research_agent
   poetry install
   poetry run python agent.py "What is the capital of France?"
   ```
   Expected output:
   ```
   Research result: The capital of France is Paris.
   ```
   Configure `config.yaml` with valid API keys:
   ```yaml
   pinecone_api_key: your-pinecone-api-key
   openai_api_key: your-openai-api-key
   ```

2. **Onchain Analyst** (`examples/onchain_analyst/`):
   A ReAct-based agent using `ContractCallTool` (`web3.py`) to query Uniswap contract data on Sepolia.
   ```bash
   cd examples/onchain_analyst
   poetry install
   poetry run python agent.py "Get Uniswap pair price"
   ```
   Expected output:
   ```
   On-chain analysis result: The USDC/WETH pair price is approximately 0.0005 ETH.
   ```
   Configure `config.yaml`:
   ```yaml
   openai_api_key: your-openai-api-key
   web3_provider_uri: https://sepolia.infura.io/v3/your-infura-key
   ```

## Documentation

Comprehensive Sphinx documentation is in `docs/`:
- **Quickstart** (`docs/quickstart.md`): Install and run your first agent.
- **Concepts** (`docs/concepts/core.md`): Core abstractions, agents, tools, RAG.
- **Tutorials** (`docs/tutorials/`): Guides for building ReAct agents and EVM tools.
- **Index** (`docs/index.md`): Entry point for all documentation.

Build and view:
```bash
cd docs
poetry run make html
xdg-open _build/html/index.html
```

## Dependencies

- **Core**: `litellm (^1.0)`, `pydantic (^2.0)`, `web3 (^6.0)`, `websockets (^13.0)`, `click (^8.1)`, `pyyaml (^6.0)`, `tenacity (^8.0)`, `pinecone-client (^3.0)`, `langchain (^0.3)`, `llama-index (^0.11)`, `redis (^5.0)`, `pandas (^2.0)`, `requests (^2.0)`, `cryptography (^43.0)`, `scikit-learn (^1.0)`, `graphviz (^0.20)`, `opentelemetry-api (^1.27)`, `opentelemetry-exporter-otlp (^1.27)`.
- **Documentation**: `sphinx (^7.0)`, `sphinx-rtd-theme (^2.0)`, `myst-parser (^3.0)`.
- **Dev**: `pytest (^8.0)`, `pytest-asyncio (^0.24.0)`, `pytest-mock (^3.12.0)`, `pytest-cov (^5.0)`.

## Running Tests

The test suite covers all components, to be fully validated after Phase 5:
```bash
poetry run pytest tests/ -v --cov=lola --cov-report=html --cov-report=term-missing
```
Tests are organized in `tests/`:
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
- `test_cli.py`: CLI commands
- `test_utils.py`: Utilities
- `test_examples.py`: Example agents

## Usage Example

Create and run a production-ready ReAct agent:
```python
import asyncio
from lola.agents.react import ReActAgent
from lola.tools.web_search import WebSearchTool
from lola.utils.config import config, load_config

load_config("config.yaml")
agent = ReActAgent(tools=[WebSearchTool()], model=config.get("openai_api_key"))
result = asyncio.run(agent.run("What is the capital of France?"))
print(result.output)  # The capital of France is Paris.
```

## Development Progress

### Phases 1-4 (Production-Ready, Completed)
- **Phase 1: Foundation (Weeks 1-4)**:
  - Implemented `core/agent.py`, `core/graph.py`, `core/state.py`, `core/memory.py` with `pydantic`, `asyncio`, and `litellm` for real LLM calls.
  - Contribution: Production-ready orchestration engine.
- **Phase 2: Components (Weeks 5-8)**:
  - Added agents (`react.py`, `plan_execute.py`, `collaborative.py`), tools (`web_search.py`, `contract_call.py`, etc.), chains (`connection.py`, `contract.py`), RAG (`multimodal.py` with Pinecone), guardrails, evals, orchestration, HITL, and LLM-agnostic interfaces.
  - Contribution: Real EVM reads (`web3.py`), RAG queries, and multi-agent workflows.
- **Phase 3: Production Envelope (Weeks 9-12)**:
  - Implemented CLI (`cli/main.py`, `create.py`, `run.py`, `build.py`, `deploy.py`) with `click` and utilities (`logging.py`, `config.py`, `telemetry.py`) with `opentelemetry`.
  - Contribution: Production-ready developer interface and diagnostics.
- **Phase 4: Developer Experience (Weeks 13-16)**:
  - Added examples (`research_agent`, `onchain_analyst`) with real `web3.py` and Pinecone integrations, and Sphinx documentation (`docs/`).
  - Contribution: Seamless onboarding with runnable examples and guides.
- **Status**: Phases 1-4 are production-ready, replacing mocks with real implementations. Phase 5 (TBD) will precede comprehensive testing.

### Phase 5 (Planned)
- To be defined; will focus on additional integration or features before final testing.
- Comprehensive testing to follow, targeting >80% coverage for all components.

### TMVP 2 (Planned)
TMVP 2 will extend TMVP 1 with:
- EVM write operations (`tools/onchain/transaction.py`) using `web3.py` and `eth-account`.
- Tokenized incentives (`tokenization/incentives.py`) for agent rewards.
- Gas relay contracts for transaction sponsorship.
- FastAPI and Docker support in `cli/deploy.py`.
- Timeline: October–November 2025.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request, ensuring tests pass and docs are updated.

## Troubleshooting

- **Installation Issues**:
  - Verify `PYTHONPATH`: `echo $PYTHONPATH` (should include `/path/to/lola-os/python`).
  - Check dependencies: `poetry show`.
  - Retry with increased timeout: `export POETRY_REQUESTS_TIMEOUT=60 && poetry install --verbose`.
- **Documentation Build**:
  - Ensure `sphinx` and `myst-parser` are installed: `poetry show sphinx myst-parser`.
  - Debug: `cd docs && poetry run make html -v`.
- **Environment Variables**:
  - Ensure API keys and RPC URLs are set correctly.

## License

MIT License (TBD).

## Contact

Levi Chinecherem Chidi <lchinecherem2018@gmail.com>

## Next Steps

- **Complete Phase 5**: Define and implement the third phase within TMVP 1 to finalize integration or features.
- **Comprehensive Testing**: Run full test suite post-Phase 5, targeting >80% coverage:
  ```bash
  poetry run pytest tests/ -v --cov=lola --cov-report=html --cov-report=term-missing
  ```
- **Release v1.0**:
  - Update `documentation/changelog.md`.
  - Publish to PyPI: `poetry build && poetry publish`.
  - Push to GitHub and promote for 10,000+ stars.
- **TMVP 2**: Plan EVM writes, tokenized features, and production deployment.
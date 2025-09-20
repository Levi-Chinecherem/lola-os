# LOLA OS TMVP 1

LOLA OS (Layered Orchestration for Logic and Automation, or Live Onchain Logical Agents) is a modular, open-source framework for building read-only AI agents with EVM-native capabilities, designed with Radical Reliability, Choice by Design, EVM-Native, Developer Sovereignty, and Explicit over Implicit tenets. TMVP 1 delivers a Python SDK (`pip install lola-os`) with core abstractions, agent templates, EVM read tools, and comprehensive documentation.

## Purpose
LOLA OS TMVP 1 enables developers to create AI agents with:
- **Modular Architecture**: Core abstractions, LLM integration via `litellm`, and EVM reads via `web3.py`.
- **Agent Templates**: ReAct, PlanExecute, Collaborative, and more.
- **EVM-Native Support**: Read-only Ethereum interactions.
- **Advanced Features**: RAG, multi-agent orchestration, guardrails, and performance optimization.
- **CLI**: Tools for project creation, running, building, and deploying.
- **Examples and Docs**: Practical demos and guides for onboarding.

## Project Structure
```
lola-os/
├── python/
│   ├── lola/
│   │   ├── core/               # Core abstractions
│   │   ├── agnostic/           # LLM-agnostic interfaces
│   │   ├── tools/              # Agent tools
│   │   ├── agents/             # Agent templates
│   │   ├── chains/             # EVM integration
│   │   ├── rag/                # Retrieval-Augmented Generation
│   │   ├── guardrails/         # Safety and permissions
│   │   ├── evals/              # Agent evaluation
│   │   ├── orchestration/      # Multi-agent coordination
│   │   ├── hitl/               # Human-in-the-loop
│   │   ├── perf_opt/           # Performance optimization
│   │   ├── specialization/     # Agent fine-tuning
│   │   ├── cli/                # CLI commands
│   │   ├── utils/              # Logging, config, telemetry
├── tests/                      # Pytest test suite
├── examples/                   # Example agents (research, onchain)
├── docs/                       # Documentation (quickstart, concepts, tutorials)
├── README.md                   # This file
├── pyproject.toml              # Poetry configuration
```

## Installation
Requires Python 3.10+. Uses [Poetry](https://python-poetry.org/) for dependency management.

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
   poetry install
   ```

4. **Verify Installation**:
   ```bash
   poetry run pytest tests/
   ```

## CLI Usage
The `lola` CLI simplifies agent development:
- **Create a Project**:
  ```bash
  poetry run lola create my_project
  cd my_project
  poetry install
  ```
- **Run an Agent**:
  ```bash
  poetry run lola run my_project.agent.BasicAgent "test query"
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
Try the example agents:
- **Research Agent**:
  ```bash
  cd examples/research_agent
  poetry install
  poetry run python agent.py "Research AI trends"
  ```
- **Onchain Analyst**:
  ```bash
  cd examples/onchain_analyst
  poetry install
  poetry run python agent.py "Get NFT floor price"
  ```

## Documentation
Explore the full documentation:
- [Quickstart](docs/quickstart.md): Install and run your first agent.
- [Concepts](docs/concepts/): Understand core components.
- [Tutorials](docs/tutorials/): Build custom agents and tools.

## Dependencies
- **Core**: `litellm (^1.0)`, `pydantic (^2.0)`, `web3 (^6.0)`, `websockets (^13.0)`, `click (^8.1)`, `pyyaml (^6.0)`, `sphinx (^7.0)`, `sphinx-rtd-theme (^2.0)`
- **Dev**: `pytest (^8.0)`, `pytest-asyncio (^0.24.0)`

## Running Tests
```bash
poetry run pytest tests/
```
Expected output:
```
collected 21 items
tests/test_agents.py .       [  4%]
tests/test_chains.py .       [  9%]
tests/test_cli.py ....       [ 28%]
tests/test_core.py .         [ 33%]
tests/test_evals.py .        [ 38%]
tests/test_guardrails.py .   [ 42%]
tests/test_hitl.py .         [ 47%]
tests/test_orchestration.py .[ 52%]
tests/test_perf_opt.py .     [ 57%]
tests/test_rag.py .          [ 61%]
tests/test_specialization.py .[ 66%]
tests/test_tools.py .        [ 71%]
tests/test_utils.py ....     [ 90%]
tests/test_examples.py ..    [100%]
21 passed in X.XXs
```

## Usage Example
```python
from lola.agents.react import ReActAgent
from lola.tools.human_input import HumanInputTool
from lola.core.state import State

agent = ReActAgent(tools=[HumanInputTool()], model="openai/gpt-4o")
state = await agent.run("Test query")
print(state.data)  # {'results': 'Stubbed response for: Test query'}
```

## Development
- **Extending Stubs**: Enhance modules (e.g., Pinecone in `rag/multimodal.py`, EVM calls in `chains/contract.py`).
- **Documentation**: Build docs with Sphinx:
  ```bash
  cd docs
  poetry run sphinx-build -b html . _build
  ```
- **Future**: Full deployment with FastAPI/Docker in TMVP 2.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
MIT License (TBD).

## Contact
Levi Chinecherem Chidi <lchinecherem2018@gmail.com>
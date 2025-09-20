# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial setup for CHANGELOG.md following Keep a Changelog format.

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [1.0.0] - 2025-09-20

### Added
- **Phases 1-4 of LOLA OS TMVP 1 Mocked Prototype**:
  - **Phase 1: Foundation**: Core abstractions (`core/agent.py`, `core/graph.py`, `core/state.py`, `core/memory.py`) for agent orchestration, state management, and memory persistence.
  - **Phase 2: Components**: Agent templates (`agents/react.py`, `agents/plan_execute.py`, `agents/collaborative.py`, etc.), tools (`tools/web_search.py`, `tools/contract_call.py`, `tools/human_input.py`), EVM abstractions (`chains/connection.py`, `chains/contract.py`), RAG (`rag/multimodal.py`), guardrails (`guardrails/content_safety.py`), evals (`evals/benchmarker.py`), orchestration (`orchestration/swarm.py`), HITL (`hitl/approval.py`), LLM-agnostic interfaces (`agnostic/unified.py`), performance optimization (`perf_opt/prompt_compressor.py`), and specialization (`specialization/fine_tuning.py`).
  - **Phase 3: Production Envelope**: CLI commands (`cli/create.py`, `cli/run.py`, `cli/build.py`, `cli/deploy.py`) for project scaffolding, execution, packaging, and deployment stubs; utilities (`utils/logging.py`, `utils/config.py`, `utils/telemetry.py`) for structured logging, YAML/env config, and OpenTelemetry export.
  - **Phase 4: Developer Experience**: Example agents (`examples/research_agent/agent.py`, `examples/onchain_analyst/agent.py`) for web research and NFT analysis; documentation (`docs/index.md`, `docs/quickstart.md`, `docs/concepts/core.md`, `docs/tutorials/build_your_first_agent.md`, `docs/tutorials/building_onchain_tools.md`) built with Sphinx.
- **Test Suite**: 21 comprehensive pytest tests (`test_core.py`, `test_agents.py`, `test_tools.py`, `test_chains.py`, `test_rag.py`, `test_guardrails.py`, `test_evals.py`, `test_orchestration.py`, `test_hitl.py`, `test_perf_opt.py`, `test_specialization.py`, `test_cli.py`, `test_utils.py`, `test_examples.py`) covering all components, with 100% pass rate.
- **Dependencies**: Poetry configuration (`pyproject.toml`) with `litellm (^1.0)`, `pydantic (^2.0)`, `web3 (^6.0)`, `click (^8.1)`, `pyyaml (^6.0)`, `sphinx (^7.0)`, `pytest-asyncio (^0.24.0)` for core, EVM, CLI, and documentation support.
- **Developer Tools**: Setup scripts (`setup_phase1.sh`, `setup_phase2.sh`, `setup_phase3.sh`, `setup_phase4.sh`) for automated folder/file creation.
- **Documentation**: Sphinx-ready docs (`docs/`) with concepts, tutorials, and quickstart; example configs (`examples/*/config.yaml`) for API keys and RPC URLs.

### Changed
- **Versioning**: Updated to `1.0.0` for TMVP 1 mocked prototype release.
- **README.md**: Comprehensive overview with installation, CLI usage, examples, testing, and TMVP 2 roadmap.
- **pyproject.toml**: Added `packages = [{include = "lola", from = "python"}]` for Poetry package recognition; included Sphinx for docs.

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- **Import Errors**: Resolved `ModuleNotFoundError` for `lola.core.edge` by consolidating `Edge` class in `core/graph.py`.
- **Abstract Class Instantiation**: Updated tests to use concrete classes (`ReActAgent`, `HumanInputTool`) instead of abstract `BaseAgent`/`BaseTool`.
- **EVM Address Validation**: Fixed `InvalidAddress` errors with valid checksum addresses (`0x0000000000000000000000000000000000000000`).
- **Async Test Warnings**: Added `pytest-asyncio` and `asyncio_mode = "auto"` in `pyproject.toml`.
- **CLI Import Issues**: Fixed `test_run_command` with proper `sys.path` manipulation and `__init__.py` creation.
- **Tool Execution**: Handled async `ContractCallTool.execute` with conditional awaiting in `test_tools.py`.
- **ContractCallTool**: Updated `__init__` to accept `web3`, `contract_address`, `abi`; implemented real contract calls with retry logic using `tenacity`.

### Security
- N/A (mocked prototype; TMVP 2 will add key management in `chains/key_manager.py`).

## [0.1.0] - 2025-09-20 (Initial Prototype)

### Added
- Initial mocked prototype for Phases 1-2, with core abstractions, agent templates, tools, and chains.
- Test suite with 11 passing tests.

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- Initial setup issues (Poetry lock, PYTHONPATH, missing modules).

### Security
- N/A
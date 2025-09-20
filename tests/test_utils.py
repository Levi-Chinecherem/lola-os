# Standard imports
import pytest
import os
from pathlib import Path
from io import StringIO
import logging

# Local
from lola.utils.logging import setup_logging, logger
from lola.utils.config import load_config
from lola.utils.telemetry import Telemetry

"""
File: Tests for utils module in LOLA OS TMVP 1 Phase 3.

Purpose: Verifies utility component functionality.
How: Uses pytest to test logging, config, and telemetry.
Why: Ensures robust utilities, per Radical Reliability.
Full Path: lola-os/tests/test_utils.py
"""

def test_logging(capsys):
    """Test structured logging setup."""
    setup_logging(level="DEBUG")
    logger.debug("Test log")
    captured = capsys.readouterr()
    assert "Test log" in captured.out
    assert '"level": "DEBUG"' in captured.out

def test_config_loading(tmp_path):
    """Test configuration loading."""
    config_path = tmp_path / "config.yaml"
    config_data = {"model": "openai/gpt-4o"}
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(config_data, f)
    
    config = load_config(config_path)
    assert config["model"] == "openai/gpt-4o"

def test_config_env_override(monkeypatch):
    """Test environment variable override."""
    monkeypatch.setenv("LOLA_MODEL", "anthropic/claude-3")
    config = load_config("nonexistent.yaml")
    assert config["model"] == "anthropic/claude-3"

def test_telemetry():
    """Test telemetry export stub."""
    telemetry = Telemetry()
    telemetry.export_metric("agent_execution_time", 1.5, {"agent": "ReActAgent"})
    # No assertion as it's a stub; logs output
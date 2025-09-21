# Standard imports
import pytest
from pathlib import Path
import logging
import os
from unittest.mock import patch, MagicMock

# Local imports
from lola.utils.logging import setup_logging, logger
from lola.utils.config import load_config, config
from lola.utils.telemetry import setup_telemetry

"""
File: Tests for LOLA OS utilities in TMVP 1 Phase 3.

Purpose: Validates logging, configuration, and telemetry functionality.
How: Uses pytest to test logging output, config loading, and telemetry setup.
Why: Ensures reliable utilities, per Radical Reliability.
Full Path: lola-os/tests/test_utils.py
"""

@pytest.fixture
def temp_config(tmp_path):
    """Fixture to create a temporary YAML config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "pinecone_api_key: test_key\n"
        "openai_api_key: test_openai\n"
        "web3_provider_uri: http://localhost:8545\n"
    )
    return config_file

def test_setup_logging(capsys):
    """Test logging setup with JSON output."""
    setup_logging(verbose=True)
    logger.debug("Test debug message")
    captured = capsys.readouterr()
    assert '"level": "DEBUG"' in captured.out
    assert '"message": "Test debug message"' in captured.out

def test_load_config(temp_config):
    """Test config loading from YAML and env vars."""
    os.environ["PINECONE_API_KEY"] = "env_key"
    loaded_config = load_config(temp_config)
    assert loaded_config["pinecone_api_key"] == "env_key"
    assert loaded_config["openai_api_key"] == "test_openai"
    assert loaded_config["web3_provider_uri"] == "http://localhost:8545"

def test_setup_telemetry(mocker):
    """Test telemetry setup with OpenTelemetry."""
    mocker.patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter")
    mocker.patch("opentelemetry.sdk.trace.TracerProvider")
    setup_telemetry(endpoint="http://test:4317")
    assert logger.info.called_with("Telemetry initialized with endpoint http://test:4317")

if __name__ == "__main__":
    pytest.main()
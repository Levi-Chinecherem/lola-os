# Standard imports
import pytest
import os
import sys
from click.testing import CliRunner
from pathlib import Path

# Local
from lola.cli.main import cli
from lola.utils.logging import setup_logging, logger

"""
File: Tests for CLI module in LOLA OS TMVP 1 Phase 3.

Purpose: Verifies CLI command functionality.
How: Uses Click's CliRunner to simulate commands.
Why: Ensures reliable CLI, per Radical Reliability.
Full Path: lola-os/tests/test_cli.py
"""

@pytest.fixture
def runner():
    setup_logging(level="DEBUG")
    return CliRunner()

def test_create_command(runner):
    """Test project creation."""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["create", "test_project"])
        assert result.exit_code == 0
        assert Path("test_project").exists()
        assert Path("test_project/agent.py").exists()
        assert Path("test_project/__init__.py").exists()

def test_run_command(runner):
    """Test running an agent (stub)."""
    with runner.isolated_filesystem():
        result_create = runner.invoke(cli, ["create", "test_project"])
        assert result_create.exit_code == 0, f"Create failed: {result_create.output}"
        # Debug: Log filesystem contents
        logger.debug(f"Filesystem contents: {list(Path('.').glob('**/*'))}")
        # Add parent directory of test_project to sys.path
        test_project_path = Path("test_project").resolve()
        sys.path.append(str(test_project_path.parent))
        logger.debug(f"sys.path: {sys.path}")
        result = runner.invoke(cli, ["run", "test_project.agent.BasicAgent", "test query"])
        assert result.exit_code == 0, f"Run failed: {result.output}"
        assert "Agent result" in result.output

def test_build_command(runner):
    """Test building a project."""
    with runner.isolated_filesystem():
        runner.invoke(cli, ["create", "test_project"])
        result = runner.invoke(cli, ["build", "test_project"])
        assert result.exit_code == 0
        assert Path("build").exists()
        assert Path("build/agent.py").exists()

def test_deploy_command(runner):
    """Test deployment stub."""
    with runner.isolated_filesystem():
        runner.invoke(cli, ["create", "test_project"])
        runner.invoke(cli, ["build", "test_project"])
        result = runner.invoke(cli, ["deploy", "build", "--target", "docker"])
        assert result.exit_code == 0
        assert "Deploying" in result.output
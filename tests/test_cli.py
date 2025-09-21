# Standard imports
import pytest
import click
from click.testing import CliRunner
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch, MagicMock

# Local imports
from lola.cli.main import cli
from lola.utils.logging import setup_logging

"""
File: Tests for LOLA OS CLI in TMVP 1 Phase 3.

Purpose: Validates CLI commands (create, run, build, deploy).
How: Uses pytest and click.testing to simulate CLI execution.
Why: Ensures reliable developer interface, per Radical Reliability.
Full Path: lola-os/tests/test_cli.py
"""

@pytest.fixture
def runner():
    """Fixture for Click CLI testing."""
    return CliRunner()

@pytest.fixture
def temp_project(tmp_path):
    """Fixture to create a temporary project directory."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'\nversion = '0.1.0'\n")
    (project_dir / "agents").mkdir()
    (project_dir / "agents" / "main.py").write_text(
        "from lola.agents.base import BaseAgent\n"
        "class TestAgent(BaseAgent):\n"
        "    async def run(self, query): return type('State', (), {'output': 'Test'})()\n"
    )
    return project_dir

def test_cli_main(runner):
    """Test main CLI command with --help."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "create" in result.output
    assert "run" in result.output
    assert "build" in result.output
    assert "deploy" in result.output

def test_create_command(runner, tmp_path, mocker):
    """Test create command with react template."""
    mocker.patch("lola.cli.commands.create.Path.exists", return_value=True)
    mocker.patch("shutil.copytree", return_value=None)
    mocker.patch("shutil.copy", return_value=None)
    result = runner.invoke(cli, ["create", str(tmp_path / "new_project"), "--template", "react"])
    assert result.exit_code == 0
    assert "Created project" in result.output

def test_run_command(runner, temp_project):
    """Test run command with a valid agent."""
    result = runner.invoke(cli, ["run", str(temp_project / "agents/main.py"), "test query"])
    assert result.exit_code == 0
    assert "Agent output: Test" in result.output

def test_build_command(runner, temp_project, mocker):
    """Test build command with a valid project."""
    mocker.patch("subprocess.run", return_value=MagicMock(stdout="Built wheel"))
    result = runner.invoke(cli, ["build", str(temp_project)])
    assert result.exit_code == 0
    assert "Build successful" in result.output

def test_deploy_command(runner, temp_project, mocker):
    """Test deploy command with Docker stub."""
    mocker.patch("subprocess.run", return_value=MagicMock())
    result = runner.invoke(cli, ["deploy", str(temp_project)])
    assert result.exit_code == 0
    assert "Deployed" in result.output

if __name__ == "__main__":
    pytest.main()
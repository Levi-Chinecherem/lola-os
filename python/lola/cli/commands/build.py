# Standard imports
import click
from pathlib import Path
import subprocess

# Local imports
from lola.utils.logging import logger

"""
File: CLI command to build agent projects for LOLA OS TMVP 1 Phase 3.

Purpose: Packages a project into a distributable wheel using Poetry.
How: Uses click for arguments, subprocess to call Poetry build.
Why: Prepares agents for deployment, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/commands/build.py
"""

@click.command()
@click.argument('project_dir', type=click.Path(exists=True))
def build(project_dir: str) -> None:
    """
    Build a LOLA OS agent project into a wheel.

    Args:
        project_dir: Path to the project directory.

    Does not:
        Deploy the project; only builds the package.
    """
    project_path = Path(project_dir)
    pyproject_toml = project_path / "pyproject.toml"
    if not pyproject_toml.exists():
        logger.error(f"No pyproject.toml found in {project_path}.")
        raise click.Abort()

    # Run Poetry build
    try:
        result = subprocess.run(
            ["poetry", "build"],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Build successful: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e.stderr}")
        raise click.Abort()
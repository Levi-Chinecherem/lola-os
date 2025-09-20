# Standard imports
import click
import os
from pathlib import Path
import shutil

# Local
from lola.utils.logging import logger

"""
File: CLI command to package a LOLA OS agent for deployment.

Purpose: Creates a distributable package for the agent.
How: Copies project files to a build directory, excluding tests/docs.
Why: Prepares agents for production, per Radical Reliability.
Full Path: lola-os/python/lola/cli/commands/build.py
"""

@click.command()
@click.argument("project_path")
@click.option("--output", default="build", help="Output directory for build")
def build(project_path: str, output: str) -> None:
    """
    Build a LOLA OS project for deployment.

    Args:
        project_path: Path to the project directory.
        output: Output directory for the build.
    Does Not: Deploy the package—use `lola deploy`.
    """
    project_path = Path(project_path)
    output_path = Path(output)

    if not project_path.exists():
        logger.error(f"Project directory {project_path} does not exist.")
        raise click.Abort()

    output_path.mkdir(exist_ok=True)
    exclude = {"tests", "docs", "__pycache__"}

    try:
        for item in project_path.iterdir():
            if item.name not in exclude:
                if item.is_dir():
                    shutil.copytree(item, output_path / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy(item, output_path)
        logger.info(f"Built project at {output_path}")
    except Exception as e:
        logger.error(f"Failed to build project: {e}")
        raise click.Abort()
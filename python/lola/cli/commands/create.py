# Standard imports
import click
import os
import shutil
from pathlib import Path

# Local
from lola.utils.logging import logger

"""
File: CLI command to scaffold a new LOLA OS project.

Purpose: Creates a new project with agent templates and configuration.
How: Copies a template directory and adds __init__.py to the project root.
Why: Simplifies onboarding for developers, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/commands/create.py
"""

@click.command()
@click.argument("project_name")
def create(project_name: str) -> None:
    """
    Create a new LOLA OS project with the given name.

    Args:
        project_name: Name of the project directory.
    Does Not: Install dependencies—use `poetry install` after creation.
    """
    project_path = Path(project_name)
    template_path = Path(__file__).parent.parent / "templates" / "basic_agent"

    if project_path.exists():
        logger.error(f"Directory {project_path} already exists.")
        raise click.Abort()

    try:
        shutil.copytree(template_path, project_path)
        # Create __init__.py to make the project a valid Python module
        (project_path / "__init__.py").touch()
        logger.info(f"Created LOLA OS project at {project_path}")
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise click.Abort()
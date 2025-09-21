# Standard imports
import click
from pathlib import Path
import shutil
import os

# Local imports
from lola.utils.logging import logger

"""
File: CLI command to scaffold new agent projects for LOLA OS TMVP 1 Phase 3.

Purpose: Creates a project directory with template files for agents and configs.
How: Uses click to define the create command, copies template files.
Why: Simplifies onboarding for developers, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/commands/create.py
"""

@click.command()
@click.argument('project_name')
@click.option('--template', type=click.Choice(['react', 'onchain']), default='react', help='Agent template type.')
def create(project_name: str, template: str) -> None:
    """
    Create a new LOLA OS agent project.

    Args:
        project_name: Name of the project directory.
        template: Template type (react or onchain).

    Does not:
        Execute or deploy the project; only scaffolds files.
    """
    project_dir = Path(project_name)
    if project_dir.exists():
        logger.error(f"Project directory {project_dir} already exists.")
        raise click.Abort()

    # Create project directory structure
    project_dir.mkdir()
    (project_dir / "agents").mkdir()
    (project_dir / "config").mkdir()

    # Copy template files based on type
    template_dir = Path(__file__).parent.parent / "templates" / template
    if not template_dir.exists():
        logger.error(f"Template {template} not found.")
        raise click.Abort()

    # Copy agent and config files
    shutil.copytree(template_dir / "agents", project_dir / "agents", dirs_exist_ok=True)
    shutil.copy(template_dir / "config.yaml", project_dir / "config/config.yaml")

    # Create README
    with (project_dir / "README.md").open("w") as f:
        f.write(f"# {project_name}\n\nLOLA OS {template} agent project.\n")

    logger.info(f"Created project {project_name} with {template} template.")
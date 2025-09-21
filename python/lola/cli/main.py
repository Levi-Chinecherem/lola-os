# Standard imports
import click
import sys
from pathlib import Path

# Local imports
from lola.cli.commands.create import create
from lola.cli.commands.run import run
from lola.cli.commands.build import build
from lola.cli.commands.deploy import deploy
from lola.utils.logging import setup_logging
from lola.utils.config import load_config

"""
File: CLI entry point for LOLA OS TMVP 1 Phase 3.

Purpose: Provides the main CLI command group for agent operations.
How: Uses click to define commands (create, run, build, deploy) and initialize logging/config.
Why: Centralizes developer interaction for production workflows, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/main.py
"""

@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to config YAML file.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def cli(config: str, verbose: bool) -> None:
    """
    LOLA OS CLI: Layered Orchestration for Logic and Automation.

    Args:
        config: Path to YAML configuration file.
        verbose: Enable verbose logging if True.

    Does not:
        Execute agent logic directly; delegates to subcommands.
    """
    # Initialize logging with verbosity
    setup_logging(verbose=verbose)
    # Load configuration from YAML or environment
    if config:
        load_config(Path(config))
    # Ensure Python version compatibility
    if sys.version_info < (3, 10):
        click.echo("LOLA OS requires Python 3.10 or higher.", err=True)
        sys.exit(1)

# Add subcommands
cli.add_command(create)
cli.add_command(run)
cli.add_command(build)
cli.add_command(deploy)

if __name__ == "__main__":
    cli()
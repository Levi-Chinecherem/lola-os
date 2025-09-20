# Standard imports
import click

# Local
from .commands.create import create
from .commands.run import run
from .commands.build import build
from .commands.deploy import deploy
from lola.utils.logging import setup_logging

"""
File: Main CLI entry point for LOLA OS TMVP 1 Phase 3.

Purpose: Provides the `lola` command with subcommands for project management.
How: Uses Click to define CLI structure; delegates to command modules.
Why: Simplifies developer interaction with LOLA OS, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/main.py
"""

@click.group()
def cli():
    """LOLA OS CLI: Create, run, build, and deploy AI agents."""
    setup_logging()

cli.add_command(create)
cli.add_command(run)
cli.add_command(build)
cli.add_command(deploy)

if __name__ == "__main__":
    cli()
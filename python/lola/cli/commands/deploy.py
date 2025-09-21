# Standard imports
import click
from pathlib import Path
import subprocess

# Local imports
from lola.utils.logging import logger

"""
File: CLI command to deploy agents locally for LOLA OS TMVP 1 Phase 3.

Purpose: Deploys an agent project to a local environment (Docker stub).
How: Uses click for arguments, subprocess for Docker commands.
Why: Enables local deployment testing, per Developer Sovereignty.
Full Path: lola-os/python/lola/cli/commands/deploy.py
"""

@click.command()
@click.argument('project_dir', type=click.Path(exists=True))
def deploy(project_dir: str) -> None:
    """
    Deploy a LOLA OS agent project locally (Docker stub).

    Args:
        project_dir: Path to the project directory.

    Does not:
        Deploy to production; only local Docker environment.
    """
    project_path = Path(project_dir)
    dockerfile = project_path / "Dockerfile"
    if not dockerfile.exists():
        # Create a basic Dockerfile
        with dockerfile.open("w") as f:
            f.write(
                """FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install poetry
RUN poetry install
CMD ["poetry", "run", "python", "agents/main.py"]
"""
            )
        logger.info(f"Created Dockerfile in {project_path}.")

    # Build and run Docker container
    try:
        subprocess.run(
            ["docker", "build", "-t", f"lola-{project_path.name}", "."],
            cwd=project_path,
            check=True,
        )
        subprocess.run(
            ["docker", "run", "--rm", f"lola-{project_path.name}"],
            cwd=project_path,
            check=True,
        )
        logger.info(f"Deployed {project_path.name} locally with Docker.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Deployment failed: {e.stderr}")
        raise click.Abort()
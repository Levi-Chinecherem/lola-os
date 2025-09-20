# Standard imports
import os
from typing import Dict, Any
import yaml

# Local
from lola.utils.logging import logger

"""
File: Configuration loader for LOLA OS TMVP 1 Phase 3.

Purpose: Loads configuration from YAML files or environment variables.
How: Parses YAML or env vars with precedence for litellm keys.
Why: Simplifies configuration management, per Developer Sovereignty.
Full Path: lola-os/python/lola/utils/config.py
"""

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML or environment variables.

    Args:
        config_path: Path to YAML config file (defaults to config.yaml).
    Returns:
        Dict with configuration values.
    Does Not: Validate config—use Pydantic in calling code.
    """
    config = {}

    # Load from YAML if exists
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")

    # Override with environment variables
    for key in ["MODEL", "API_KEY"]:
        if env_value := os.environ.get(f"LOLA_{key}"):
            config[key.lower()] = env_value

    return config
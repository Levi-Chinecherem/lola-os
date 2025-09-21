# Standard imports
from pathlib import Path
import os
import yaml
from typing import Dict, Any

# Local imports
from lola.utils.logging import logger

"""
File: Configuration management for LOLA OS TMVP 1 Phase 3.

Purpose: Loads configuration from YAML files and environment variables.
How: Uses pyyaml for YAML parsing, os for env vars.
Why: Centralizes configuration for flexibility, per Developer Sovereignty.
Full Path: lola-os/python/lola/utils/config.py
"""

config: Dict[str, Any] = {}

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and override with environment variables.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict of configuration settings.

    Does not:
        Modify the config file; only reads it.
    """
    global config
    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

    # Override with environment variables
    env_vars = {
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "WEB3_PROVIDER_URI": os.getenv("WEB3_PROVIDER_URI"),
        "REDIS_URL": os.getenv("REDIS_URL"),
    }
    for key, value in env_vars.items():
        if value:
            config[key.lower()] = value

    logger.info(f"Loaded configuration from {config_path}")
    return config
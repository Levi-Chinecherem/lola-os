# Standard imports
import logging

# Third-party
from prometheus_client import start_http_server

"""
File: Initializes Prometheus for LOLA OS TMVP 1 Phase 5.

Purpose: Provides metrics export for monitoring.
How: Starts Prometheus HTTP server for metrics scraping.
Why: Ensures production-ready monitoring, per Radical Reliability tenet.
Full Path: lola-os/python/lola/utils/prometheus.py
"""

logger = logging.getLogger(__name__)

def init_prometheus(port: int = 8000) -> None:
    """
    Initialize Prometheus metrics server.

    Args:
        port: Port for metrics endpoint (default 8000).

    Does Not: Handle visualization—use grafana.py.
    """
    try:
        start_http_server(port)
        logger.info(f"Prometheus server started on port {port}")
    except Exception as e:
        logger.error(f"Prometheus initialization failed: {e}")

__all__ = ["init_prometheus"]
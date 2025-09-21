# Standard imports
import typing as tp
import os

"""
File: Defines the HardwareOptimizer class for LOLA OS TMVP 1 Phase 2.

Purpose: Optimizes hardware for agent execution.
How: Sets environment variables for CUDA/FlashAttention.
Why: Improves performance on GPU, per Radical Reliability tenet.
Full Path: lola-os/python/lola/perf_opt/hardware.py
Future Optimization: Migrate to Rust for hardware detection (post-TMVP 1).
"""

class HardwareOptimizer:
    """HardwareOptimizer: Optimizes hardware settings. Does NOT persist config—use utils/config.py."""

    def optimize(self) -> None:
        """
        Optimizes for GPU if available.

        Does Not: Handle CPU optimization—expand in TMVP 2.
        """
        if os.system("nvidia-smi") == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["USE_FLASH_ATTENTION"] = "1"

__all__ = ["HardwareOptimizer"]
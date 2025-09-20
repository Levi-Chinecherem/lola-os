"""
File: Defines the BaseTool abstract class for LOLA OS TMVP 1.

Purpose: Provides a base for tool implementations used by agents.
How: Defines an abstract interface for tool execution.
Why: Ensures consistent tool integration, per Developer Sovereignty.
Full Path: lola-os/python/lola/tools/base.py
"""
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Abstract base class for tools used by LOLA OS agents."""
    
    name: str = ""

    @abstractmethod
    def execute(self, *args, **kwargs) -> dict:
        """
        Execute the tool with given arguments.
        
        Returns:
            dict: Result of tool execution.
        """
        pass

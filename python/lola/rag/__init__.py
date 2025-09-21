"""
File: Initializes the RAG module for LOLA OS TMVP 1 Phase 2.

Purpose: Exports advanced RAG components for agents.
How: Defines package-level exports for RAG tools.
Why: Centralizes access to RAG utilities, per Developer Sovereignty.
Full Path: lola-os/python/lola/rag/__init__.py
"""

from .multimodal import MultiModalRetriever
from .hyde import HyDEGenerator
from .evaluator import RAGEvaluator
from .connector import DynamicDataConnector

__all__ = [
    "MultiModalRetriever",
    "HyDEGenerator",
    "RAGEvaluator",
    "DynamicDataConnector",
]
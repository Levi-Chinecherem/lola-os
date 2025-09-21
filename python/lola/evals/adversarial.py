# Standard imports
import typing as tp

# Local
from lola.agents.base import BaseTemplateAgent
from lola.core.state import State

"""
File: Defines the AdversarialTestingSuite class for LOLA OS TMVP 1 Phase 2.

Purpose: Tests agents against adversarial inputs.
How: Generates jailbreak prompts and checks responses.
Why: Uncovers vulnerabilities, per Radical Reliability tenet.
Full Path: lola-os/python/lola/evals/adversarial.py
Future Optimization: Migrate to Rust for advanced attacks (post-TMVP 1).
"""

class AdversarialTestingSuite:
    """AdversarialTestingSuite: Tests agents against adversarial inputs. Does NOT persist results—use StateManager."""

    async def test_adversarial(self, agent: BaseTemplateAgent, adversarial_queries: tp.List[str]) -> dict:
        """
        Tests agent with adversarial queries.

        Args:
            agent: BaseTemplateAgent instance.
            adversarial_queries: List of adversarial queries.

        Returns:
            Dict with pass/fail for each query.

        Does Not: Generate queries—use external tools.
        """
        results = []
        for query in adversarial_queries:
            state = await agent.run(query)
            passed = "safe" in state.output.lower()  # Simple check; expand in TMVP 2
            results.append({"query": query, "passed": passed})
        return {"results": results}

__all__ = ["AdversarialTestingSuite"]
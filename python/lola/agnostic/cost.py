# Standard imports
import typing as tp

# Third-party
from litellm import completion

"""
File: Defines the CostOptimizer class for LOLA OS TMVP 1 Phase 2.

Purpose: Optimizes LLM calls for cost.
How: Routes to cheapest model based on complexity.
Why: Reduces operational costs, per Developer Sovereignty tenet.
Full Path: lola-os/python/lola/agnostic/cost.py
Future Optimization: Migrate to Rust for real-time optimization (post-TMVP 1).
"""

class CostOptimizer:
    """CostOptimizer: Optimizes LLM calls for cost. Does NOT persist logs—use StateManager."""

    def __init__(self, model_costs: tp.Dict[str, float]):
        """
        Initialize with model costs.

        Args:
            model_costs: Dict of model to cost per token.
        """
        self.model_costs = model_costs
        self.cheap_model = min(model_costs, key=model_costs.get)

    async def call(self, prompt: str, complexity: str = "low") -> str:
        """
        Calls LLM with cost-optimized model.

        Args:
            prompt: Input prompt.
            complexity: "low" or "high" for model selection.

        Returns:
            Response string.

        Does Not: Handle fallbacks—use fallback.py.
        """
        model = self.cheap_model if complexity == "low" else max(self.model_costs, key=self.model_costs.get)
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

__all__ = ["CostOptimizer"]
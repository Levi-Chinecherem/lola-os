# Standard imports
import typing as tp

# Local
from .base import BaseTemplateAgent
from lola.core.state import State

"""
File: Defines the PlanExecuteAgent class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements the Plan-Execute agent pattern for goal-oriented tasks.
How: Generates a plan with LLM, then executes steps sequentially.
Why: Supports structured task completion, per Choice by Design tenet.
Full Path: lola-os/python/lola/agents/plan_execute.py
Future Optimization: Migrate to Rust for parallel plan execution (post-TMVP 1).
"""

class PlanExecuteAgent(BaseTemplateAgent):
    """PlanExecuteAgent: Implements Plan-Execute pattern. Does NOT persist state—use StateManager."""

    async def run(self, query: str) -> State:
        """
        Execute the Plan-Execute loop for the query.

        Args:
            query: User input string.
        Returns:
            State: Final state after plan generation and execution.
        Does Not: Handle concurrent execution—use perf_opt/hardware.py.
        """
        self.state.update({"query": query, "plan": [], "results": []})
        # Inline: Generate plan with LLM
        plan_prompt = f"Generate a step-by-step plan for: {query}"
        plan = await self._call_llm(plan_prompt)
        self.state.data["plan"] = plan.split("\n")
        # Inline: Execute each plan step
        for step in self.state.data["plan"]:
            if step.strip():
                action = await self._call_llm(f"Execute step: {step}\nContext: {self.state.history}")
                tool_name, tool_params = self.parse_action(action)
                result = await self.execute_tool(tool_name, tool_params)
                self.state.data["results"].append(result)
                self.state.history.append({"role": "execution", "content": f"Step: {step}, Result: {result}"})
        self.state.update({"output": "Plan completed"})
        return self.state

    def parse_action(self, action: str) -> tp.Tuple[str, tp.Dict[str, tp.Any]]:
        """
        Parses action string into tool name and parameters.

        Args:
            action: LLM-generated action string.

        Returns:
            Tuple of tool name and parameters dict.

        Does Not: Validate parameters—use guardrails/prompt_shield.py.
        """
        # Inline: Simple parsing; improve with NLP in TMVP 2
        tool_name = action.split("(")[0].strip()
        params_str = action.split("(")[1].split(")")[0]
        params = {p.split("=")[0].strip(): p.split("=")[1].strip() for p in params_str.split(",")}
        return tool_name, params

__all__ = ["PlanExecuteAgent"]
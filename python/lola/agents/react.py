# Standard imports
import typing as tp

# Local
from .base import BaseTemplateAgent
from lola.core.state import State

"""
File: Defines the ReActAgent class for LOLA OS TMVP 1 Phase 2.

Purpose: Implements the ReAct (Reason-Act) agent pattern for iterative reasoning and action.
How: Alternates between LLM reasoning and tool execution until task completion.
Why: Supports complex problem-solving with explicit steps, per Choice by Design tenet.
Full Path: lola-os/python/lola/agents/react.py
Future Optimization: Migrate to Rust for high-throughput reasoning loops (post-TMVP 1).
"""

class ReActAgent(BaseTemplateAgent):
    """ReActAgent: Implements the Reason-Act pattern. Does NOT persist state—use StateManager."""

    async def run(self, query: str) -> State:
        """
        Execute the ReAct loop for the query.

        Args:
            query: User input string.
        Returns:
            State: Final state after reasoning and actions.
        Does Not: Handle multi-agent collaboration—use orchestration/swarm.py.
        """
        self.state.update({"query": query, "steps": []})
        max_steps = 10  # Prevent infinite loops
        for step in range(max_steps):
            # Inline: Generate reasoning with LLM
            prompt = f"Reason about: {self.state.data['query']}\nHistory: {self.state.history}"
            reasoning = await self._call_llm(prompt)
            self.state.history.append({"role": "reasoning", "content": reasoning})
            self.state.data["steps"].append({"step": step, "reasoning": reasoning})
            # Inline: Decide action or conclude
            action_prompt = f"Based on reasoning: {reasoning}, decide next action or final answer."
            action = await self._call_llm(action_prompt)
            if "final answer" in action.lower():
                self.state.update({"output": action.split("final answer:")[1].strip()})
                break
            tool_name, tool_params = self.parse_action(action)  # Stub for parsing
            result = await self.execute_tool(tool_name, tool_params)
            self.state.history.append({"role": "observation", "content": str(result)})
            self.state.data["steps"].append({"action": tool_name, "result": result})
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

__all__ = ["ReActAgent"]
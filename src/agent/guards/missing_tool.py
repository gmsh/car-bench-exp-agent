"""MissingToolGuard — verifies every LLM-generated tool name exists in the live schema.

Hallucination tasks intentionally remove certain tools to test whether
the agent admits it cannot perform an action (correct) versus
hallucinating a tool call (incorrect). The executor wraps this guard's
output with one retry on violation; if the retry still references a
missing tool, the executor strips ``tool_calls`` entirely so a missing
tool is never dispatched.
"""
from __future__ import annotations

from ..policy import PolicyViolation


class MissingToolGuard:
    """
    Verifies every LLM-generated tool name exists in the current live tool schema.

    On violation: the executor injects a CAPABILITY_UNAVAILABLE observation
    and gives the LLM one retry. Double-failure fallback: if the retry still
    calls a missing tool, the executor strips all tool_calls and returns a
    plain text response — a missing tool is never dispatched.
    """

    def check(
        self,
        tool_calls: list[dict],
        available_tool_names: set[str],
    ) -> list[PolicyViolation]:
        violations = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            if name not in available_tool_names:
                violations.append(PolicyViolation(
                    tc["id"], name,
                    f"CAPABILITY_UNAVAILABLE: Tool '{name}' is NOT available in this session — "
                    f"do NOT call '{name}' again. "
                    f"All other tools in the available tool list remain fully accessible. "
                    f"Continue reasoning and use whichever available tools can help accomplish "
                    f"the user's goal. "
                    f"Only if the goal truly cannot be achieved with any available tool should "
                    f"you inform the user that the required capability is not available. "
                    f"Do NOT claim to have performed the action without a tool call."
                ))
        return violations

"""ParamSchemaGuard — checks every parameter key the LLM included in a
tool call exists in that tool's live schema properties.

Catches LLM hallucinations of parameter names that look plausible but
are not actually exposed by the underlying tool. A retry is given on
violation; double-failure strips ``tool_calls`` entirely.
"""
from __future__ import annotations

import json

from ..policy import PolicyViolation


class ParamSchemaGuard:
    """
    Checks that every parameter key the LLM included in a tool call actually
    exists in that tool's live schema properties.

    Lifecycle:
      1. Instantiate once in the executor's ``__init__``.
      2. Call ``init_from_schemas(tools)`` on first schema receipt
         (no-op after that).
      3. Call ``check(tool_calls)`` on every LLM turn before dispatching.
    """

    def __init__(self) -> None:
        self._schema_props: dict[str, set[str]] = {}  # {tool_name: set of valid param names}
        self._initialized = False

    def init_from_schemas(self, tool_schemas: list[dict]) -> None:
        if self._initialized:
            return
        for tool in tool_schemas:
            fn = tool.get("function", {})
            name = fn.get("name", "")
            props = fn.get("parameters", {}).get("properties", {})
            self._schema_props[name] = set(props.keys())
        self._initialized = True

    def check(self, tool_calls: list[dict]) -> list[PolicyViolation]:
        if not self._initialized:
            return []
        violations = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            valid_params = self._schema_props.get(name)
            if valid_params is None:
                continue  # MissingToolGuard handles unknown tool names
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                continue
            for param_key in args:
                if param_key not in valid_params:
                    violations.append(PolicyViolation(
                        tc["id"], name,
                        f"PARAM_NOT_IN_SCHEMA: Parameter '{param_key}' is NOT available "
                        f"in the current schema for tool '{name}'. "
                        f"Do NOT include '{param_key}' in the tool call again. "
                        f"Use only parameters that appear in the live schema. "
                        f"If this parameter is required to fulfill the user request, "
                        f"explain to the user that this capability is not available "
                        f"in this session."
                    ))
        return violations

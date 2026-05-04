"""ParamSchemaGuard — checks LLM tool-call arguments against live schemas.

Catches LLM hallucinations of parameter names that look plausible but
are not actually exposed by the underlying tool, and catches calls that
omit required parameters. A retry is given on violation; double-failure
strips ``tool_calls`` entirely.
"""
from __future__ import annotations

import json

from ..policy import PolicyViolation


class ParamSchemaGuard:
    """
    Checks that every parameter key the LLM included in a tool call actually
    exists in that tool's live schema properties, and that all required
    parameters are present.

    Lifecycle:
      1. Instantiate once in the executor's ``__init__``.
      2. Call ``init_from_schemas(tools)`` whenever a live schema arrives.
         Hallucination tasks may remove tools or params per context, so this
         index is intentionally rebuilt rather than cached forever.
      3. Call ``check(tool_calls)`` on every LLM turn before dispatching.
    """

    def __init__(self) -> None:
        self._schema_props: dict[str, set[str]] = {}  # {tool_name: set of valid param names}
        self._required_params: dict[str, list[str]] = {}  # {tool_name: required param names}
        self._required_param_options: dict[str, list[tuple[str, ...]]] = {}
        self._initialized = False

    def init_from_schemas(self, tool_schemas: list[dict]) -> None:
        self._schema_props = {}
        self._required_params = {}
        self._required_param_options = {}
        for tool in tool_schemas:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", tool)
            if not isinstance(fn, dict):
                continue
            name = fn.get("name", "")
            params = fn.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            props = params.get("properties", {})
            if not isinstance(props, dict):
                props = {}
            self._schema_props[name] = set(props.keys())
            required = [p for p in params.get("required", []) if isinstance(p, str)]
            self._required_params[name] = required
            self._required_param_options[name] = [(p,) for p in required]
            for options in self._fallback_required_param_options(name):
                if not any(set(group) & set(options) for group in self._required_param_options[name]):
                    self._required_param_options[name].append(options)
        self._initialized = True

    def _fallback_required_param_options(self, tool_name: str) -> list[tuple[str, ...]]:
        return {
            "get_location_id_by_location_name": [("location",)],
            "open_close_sunshade": [("percentage",)],
            "open_close_window": [("percentage",)],
            "search_poi_along_the_route": [("category_poi",)],
            "search_poi_at_location": [("category_poi",)],
            "send_email": [("email_addresses",), ("content_message",)],
            "set_ambient_lights": [("lightcolor",)],
            "set_fan_speed": [("level", "speed")],
            "set_new_navigation": [("route_ids",)],
            "set_reading_light": [("position",)],
            "set_seat_heating": [("level",), ("seat_zone",)],
        }.get(tool_name, [])

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
            for required_options in self._required_param_options.get(name, []):
                available_options = [p for p in required_options if p in valid_params]
                if not available_options:
                    violations.append(PolicyViolation(
                        tc["id"], name,
                        f"PARAM_REQUIRED_UNAVAILABLE: Tool '{name}' requires "
                        f"{_format_required_options(required_options)} to be usable, but "
                        f"{_format_required_options(required_options, quote=False)} "
                        f"{'is' if len(required_options) == 1 else 'are'} not available in "
                        f"the current live schema. Do NOT call '{name}' in this session; "
                        f"explain that the required parameter is not available. Also do NOT "
                        f"claim you can or will perform the action, and do NOT ask the user "
                        f"to confirm a parameter that cannot be sent."
                    ))
                elif not any(p in args for p in available_options):
                    violations.append(PolicyViolation(
                        tc["id"], name,
                        f"PARAM_REQUIRED_MISSING: Required parameter "
                        f"{_format_required_options(available_options)} is missing "
                        f"from tool call '{name}'. Check the live schema for '{name}' before "
                        f"retrying. If the required parameter appears in the live schema, include it. "
                        f"If it does NOT appear in the live schema, the requested "
                        f"action cannot be completed in this session: do NOT call '{name}' again, "
                        f"do NOT claim you can or will perform the action, and do NOT ask the user "
                        f"to confirm an action you cannot execute. Instead, explain that the "
                        f"required capability is unavailable and offer any safe non-executing help "
                        f"such as drafting text or reporting looked-up information."
                    ))
        return violations


def _format_required_options(options: tuple[str, ...] | list[str], quote: bool = True) -> str:
    names = [f"'{p}'" if quote else p for p in options]
    if len(names) == 1:
        return names[0]
    return "one of " + ", ".join(names)

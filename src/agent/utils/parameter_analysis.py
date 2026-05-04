"""Schema-driven parameter analysis utilities for
:class:`~agent.guards.universal_ambiguity.UniversalAmbiguityGuard`.

These helpers translate raw JSON Schemas into the cascade's
per-parameter strategy: classifying each parameter by shape, and
deriving P1 regex patterns from numeric ranges. They are pure
functions of the schema — no observation context is required.
"""
from __future__ import annotations

import re

from .parameter_classifier import (
    _classify_parameter,
    PARAM_BOOLEAN,
    PARAM_ENUM,
    PARAM_FREE_STRING,
    PARAM_NUMBER,
)


def _analyze_tool_parameters(tool_schema: dict) -> dict:
    """
    Classify all parameters of a tool from its JSON Schema definition.

    Return structure:
      kind            : "no_arg" | "single_arg" | "multi_arg"
      count           : int — total parameter count
      required        : list[str] — required parameter names
      parameters      : dict[str, str] — {parameter_name: PARAM_* label}
      has_enum        : bool
      has_number      : bool
      has_boolean     : bool
      has_free_string : bool

    """
    fn = tool_schema.get("function", tool_schema)
    if not isinstance(fn, dict):
        fn = {}
    parameters_schema = fn.get("parameters", {})
    if not isinstance(parameters_schema, dict):
        parameters_schema = {}
    properties = parameters_schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    required_raw = parameters_schema.get("required", [])
    required = [p for p in required_raw if isinstance(p, str)] if isinstance(required_raw, list) else []

    parameters: dict[str, str] = {
        pname: _classify_parameter(pschema)
        for pname, pschema in properties.items()
        if isinstance(pschema, dict)
    }
    count = len(parameters)

    return {
        "kind":            "no_arg" if count == 0 else ("single_arg" if count == 1 else "multi_arg"),
        "count":           count,
        "required":        required,
        "parameters":      parameters,
        "has_enum":        any(t == PARAM_ENUM        for t in parameters.values()),
        "has_number":      any(t == PARAM_NUMBER      for t in parameters.values()),
        "has_boolean":     any(t == PARAM_BOOLEAN     for t in parameters.values()),
        "has_free_string": any(t == PARAM_FREE_STRING for t in parameters.values()),
    }


def _number_p1_pattern(parameter_name: str, parameter_schema: dict) -> "re.Pattern | None":
    """
    Build a P1 regex for a PARAM_NUMBER parameter.
    """
    desc = parameter_schema.get("description", "").lower()
    minimum = parameter_schema.get("minimum")
    maximum = parameter_schema.get("maximum")

    # ── Strategy 1: unit-anchored ─────────────────────────────────────────────
    if any(kw in desc for kw in ("celsius", "degree")):
        return re.compile(
            r'\b\d{1,2}(?:\.\d)?\s*(?:degree|celsius|°|C|degrees?)\b', re.I
        )
    if "percent" in desc:
        # Percentage — always has % or "percent" suffix
        return re.compile(r'\b\d{1,3}\s*(?:%|percent)\b', re.I)

    # ── Strategy 2: range-constrained integer alternation ────────────────────
    if minimum is not None and maximum is not None:
        min_int = int(minimum)
        max_int = int(maximum)
        span = max_int - min_int
        if 0 < span <= 50:
            # Enumerate every valid integer — more precise than character-class shortcuts.
            # E.g. [16..28] → (?:16|17|...|28) rather than (?:1[5-9]|2\d)
            values = [str(i) for i in range(min_int, max_int + 1)]
            range_pat = "(?:" + "|".join(values) + ")"
            # Reject matches that are immediately followed by a non-matching unit
            not_unit = (
                r"(?!\s*(?:km\b|m\b|min(?:ute)?s?\b|sec(?:ond)?s?\b"
                r"|mph\b|kph\b|litre?s?\b|day\b|hour\b|%|px\b))"
            )
            return re.compile(r"\b" + range_pat + r"(?:\.\d)?\b" + not_unit, re.I)

    return None  # cannot auto-derive safely — leave P1 unsatisfied so the cascade falls to P2/P4

"""Schema-driven parameter classification.

Used by :class:`~agent.guards.param_schema.ParamSchemaGuard` and by
:class:`~agent.guards.universal_ambiguity.UniversalAmbiguityGuard`'s
``_analyze_tool_parameters`` helper. The classification labels
(PARAM_BOOLEAN, PARAM_ENUM, PARAM_NUMBER, PARAM_FREE_STRING,
PARAM_OTHER) drive the cascade's per-parameter strategy without
needing a per-tool branch.
"""
from __future__ import annotations


def _classify_parameter(parameter_schema: dict) -> str:
    """
    Classify a single parameter's JSON Schema into one of the PARAM_* labels.

    Disambiguation implication:
      PARAM_BOOLEAN     — user always says "on"/"off" → P1 by definition, no guard
      PARAM_ENUM        — finite choices → P1 if user names a value; else P2/P4/P5
      PARAM_NUMBER      — continuous range → P5 candidate; P1 via unit-anchored pattern
      PARAM_FREE_STRING — user-specified content (body, subject) → no auto-guard
      PARAM_OTHER       — arrays / objects → must be handled explicitly in config
    """
    ptype = parameter_schema.get("type", "")
    if ptype == "boolean":
        return PARAM_BOOLEAN
    if ptype == "string":
        return PARAM_ENUM if "enum" in parameter_schema else PARAM_FREE_STRING
    if ptype in ("number", "integer"):
        return PARAM_NUMBER
    return PARAM_OTHER


# ──────────────────────────────────────────────────────────────

# Parameter type labels — use these constants everywhere to avoid typos.
PARAM_BOOLEAN     = "boolean"       # {"type": "boolean"}
PARAM_ENUM        = "enum"          # {"type": "string", "enum": [...]}
PARAM_NUMBER      = "number"        # {"type": "number"} or {"type": "integer"}
PARAM_FREE_STRING = "free_string"   # {"type": "string"} without enum
PARAM_OTHER       = "other"         # array, object, or unknown

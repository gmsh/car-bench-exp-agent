"""UniversalAmbiguityGuard — single guard enforcing the P1→P2→P4→P5
disambiguation cascade for every tool listed in
``PARAM_RESOLUTION_REGISTRY``.

Resolution semantics per tool call:

  P1  — if the user's message already states the parameter value
        explicitly, skip P2 (and P4 if ``p4_p1_bypass=True``).
  P2  — if the parameter has a configured preference category and
        that category has not yet been observed via
        ``get_user_preferences`` this turn, emit a
        "check preferences first" violation.
  P4  — if the tool's required state-check tools have not been
        called yet, emit a "check state first" violation. Reached
        only when P2 passes (or no P2 is configured).
  P5  — if the parameter has no resolvable default (e.g. continuous
        range like 0–100%) and the user did not specify a value,
        emit a "must ask user" violation. Reached only after
        P2 and P4 pass.

The executor injects synthetic tool errors for any violation and gives
the LLM one retry to repair the batch.
"""
from __future__ import annotations

import json
import re

from ..policy import PolicyViolation
from ..utils.parameter_classifier import PARAM_NUMBER
from ..utils.parameter_analysis import (
    _analyze_tool_parameters,
    _number_p1_pattern,
)


class UniversalAmbiguityGuard:
    """
    Single guard that enforces the P1→P2→P4→P5 disambiguation cascade for all
    tools listed in ``PARAM_RESOLUTION_REGISTRY``.

    Lifecycle:
      1. Instantiate once in ``CARBenchAgentExecutor.__init__``.
      2. Call ``init_from_schemas(tools)`` when the live tool schemas are
         first received. This auto-generates per-parameter P1 patterns for
         entries marked ``p1_from_enum=True`` or ``p1_from_schema=True``.
      3. Call ``check(...)`` on every LLM turn before dispatching tool calls.
    """

    def __init__(self) -> None:
        # Nested: {tool_name: {parameter_name: compiled_P1_regex}}
        # Built from p1_from_enum and p1_from_schema entries in PARAM_RESOLUTION_REGISTRY.
        self._parameter_p1_patterns: dict[str, dict[str, re.Pattern]] = {}
        # Maps tool_name → full _analyze_tool_parameters result (cached for reuse).
        self._tool_parameter_analysis: dict[str, dict] = {}
        self._schemas_initialized = False

    def init_from_schemas(self, tool_schemas: list[dict]) -> None:
        """
        Analyze all tool schemas and auto-generate per-parameter P1 patterns.

        For each tool listed in ``PARAM_RESOLUTION_REGISTRY``:
          1. Run ``_analyze_tool_parameters`` and cache the result.
          2. For each parameter in ``cfg["params"]``:
             - ``p1_from_enum=True``  → collect enum values for THIS parameter only
               (not all parameters — avoids cross-parameter pattern contamination)
             - ``p1_from_schema=True`` → call ``_number_p1_pattern`` for this parameter's
               schema (unit-anchored or range-constrained)

        Subsequent calls are no-ops.
        """
        if self._schemas_initialized:
            return

        for tool in tool_schemas:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", {})
            if not isinstance(fn, dict):
                continue
            name = fn.get("name", "")

            # Always analyze and cache the tool's parameter structure
            analysis = _analyze_tool_parameters(tool)
            self._tool_parameter_analysis[name] = analysis

            cfg = PARAM_RESOLUTION_REGISTRY.get(name)
            if not cfg:
                continue

            params = fn.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            props = params.get("properties", {})
            if not isinstance(props, dict):
                props = {}
            parameter_configs = cfg.get("params", {})

            for parameter_name, parameter_config in parameter_configs.items():
                pschema = props.get(parameter_name, {})
                if not isinstance(pschema, dict):
                    pschema = {}

                if parameter_config.get("p1_from_enum"):
                    # Build regex from THIS parameter's enum values only
                    enum_values = [str(v) for v in pschema.get("enum", [])]
                    if enum_values:
                        terms = [re.escape(v.lower()) for v in enum_values]
                        self._parameter_p1_patterns.setdefault(name, {})[parameter_name] = re.compile(
                            r'\b(' + '|'.join(terms) + r')\b', re.I
                        )

                elif parameter_config.get("p1_from_schema"):
                    # Build number P1 regex from THIS parameter's schema (unit/range)
                    if analysis["parameters"].get(parameter_name) == PARAM_NUMBER:
                        pattern = _number_p1_pattern(parameter_name, pschema)
                        if pattern is not None:
                            self._parameter_p1_patterns.setdefault(name, {})[parameter_name] = pattern

        self._schemas_initialized = True

    def _p1_satisfied(
        self,
        tool_name: str,
        parameter_name: str,
        parameter_config: dict,
        user_msg: str,
        argument_value=None,
    ) -> bool:
        """
        Return True if the user message or resolved tool argument contains an explicit
        value for this parameter (P1).

        Resolution order:
          1. ``p1_pattern`` in ``parameter_config``
          2. Schema-derived pattern in ``self._parameter_p1_patterns`` (built by
             ``init_from_schemas`` from ``p1_from_enum`` or ``p1_from_schema``).
        """
        pattern = parameter_config.get("p1_pattern")
        if pattern is None and (
            parameter_config.get("p1_from_enum") or parameter_config.get("p1_from_schema")
        ):
            pattern = self._parameter_p1_patterns.get(tool_name, {}).get(parameter_name)
        if pattern is None:
            return False
        if pattern.search(user_msg or ""):
            return True
        if argument_value is None:
            return False
        return bool(pattern.search(str(argument_value)))

    def check(
        self,
        tool_calls: list[dict],
        preference_categories_observed: set[str],
        tools_called: set[str],
        user_msg: str,
    ) -> list[PolicyViolation]:
        """
        Check each tool call against ``PARAM_RESOLUTION_REGISTRY`` — per-parameter.

        For each tool call:
          1. P4 (tool-level): if required state tools not called → violation, skip
             per-parameter rules. ``p4_p1_bypass=True`` skips P4 if ANY parameter's
             P1 is satisfied.
          2. Per-parameter P2: if a parameter's P1 is not satisfied AND its
             preference category has not yet been observed → violation.
             Multiple parameters sharing the same ``p2_pref_cat`` report only
             ONE P2 violation (one preference fetch resolves all of them).
          3. Per-parameter P5: if a parameter has no resolvable default AND the
             user did not specify it → "must ask user" violation.

        Once a higher-priority violation is reported for a parameter, lower-priority
        levels are skipped — the LLM should repair the top issue first.
        """
        violations: list[PolicyViolation] = []

        for tc in tool_calls:
            name = tc["function"]["name"]
            cfg = PARAM_RESOLUTION_REGISTRY.get(name)
            if cfg is None:
                continue  # tool not in registry — no ambiguity rules apply

            parameter_configs: dict = cfg.get("params", {})
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except Exception:
                args = {}

            # ── P4: tool-level state check ────────────────────────────────────
            p4_tools = cfg.get("p4_tools") or []
            if p4_tools:
                # p4_p1_bypass=True: skip P4 when user already gave us the value explicitly
                p4_bypass = cfg.get("p4_p1_bypass", False) and any(
                    self._p1_satisfied(name, pn, pc, user_msg, args.get(pn))
                    for pn, pc in parameter_configs.items()
                )
                if not p4_bypass:
                    p4_all = cfg.get("p4_all", False)
                    if p4_all:
                        missing = [t for t in p4_tools if t not in tools_called]
                    else:
                        missing = p4_tools if not any(t in tools_called for t in p4_tools) else []
                    if missing:
                        violations.append(PolicyViolation(
                            tc["id"], name,
                            f"STATE_CHECK_REQUIRED: Before calling {name}, you MUST first call "
                            f"{' and '.join(missing)} to check current car state (P4 context "
                            f"check). Proceeding without this state check is a policy violation "
                            f"even if the parameter value appears correct."
                        ))
                        continue  # don't check per-parameter rules if P4 not satisfied

            # ── Per-parameter P2 / P5 ─────────────────────────────────────────
            # Parameters sharing a P2 category produce only ONE violation per category
            # (one preferences fetch resolves all of them).
            reported_p2_categories: set[str] = set()

            for parameter_name, parameter_config in parameter_configs.items():
                p1_ok = self._p1_satisfied(
                    name,
                    parameter_name,
                    parameter_config,
                    user_msg,
                    args.get(parameter_name),
                )

                # ── P2: preference observation check ──────────────────────────
                preference_category = parameter_config.get("p2_pref_cat")
                if (
                    preference_category
                    and not p1_ok
                    and preference_category not in preference_categories_observed
                ):
                    if preference_category not in reported_p2_categories:
                        # Pre-compute split parts outside f-string (Python 3.11 restriction)
                        p2_top = preference_category.split(".")[0]
                        p2_leaf = preference_category.split(".")[-1]
                        violations.append(PolicyViolation(
                            tc["id"], name,
                            f"DISAMBIGUATION_REQUIRED: Before calling {name}, parameter "
                            f"'{parameter_name}' is not resolved. You MUST first call "
                            f"get_user_preferences(preference_categories="
                            f"{{'{p2_top}': {{'{p2_leaf}': true}}}}) "
                            f"to check if the user has a saved preference. "
                            f"Only proceed with {name} after retrieving preferences. "
                            f"If no preference is found, ask the user to specify the value."
                        ))
                        reported_p2_categories.add(preference_category)
                    continue  # repair P2 first; don't also report P5 for this parameter

                # ── P5: continuous-range parameter — must ask user ────────────
                if parameter_config.get("p5_required") and not p1_ok:
                    if preference_category and preference_category in preference_categories_observed and args.get(parameter_name) is not None:
                        continue
                    p5_p1 = parameter_config.get("p5_p1_pattern")
                    if not (p5_p1 and p5_p1.search(user_msg or "")):
                        violations.append(PolicyViolation(
                            tc["id"], name,
                            f"PARAMETER_UNRESOLVED: '{parameter_name}' for {name} was not explicitly "
                            f"specified by the user and has no resolvable default "
                            f"(it is a 0–100 continuous range). "
                            f"You MUST ask the user to specify the {parameter_name} value "
                            f"before calling {name}. Do NOT guess or assume any value."
                        ))

        return violations


# Per-tool disambiguation rules — the registry that drives the cascade.
#
# Structure:
#   tool_name → {
#     "p4_tools"      : list[str] | None   — tool-level state prerequisites (P4)
#     "p4_all"        : bool               — require ALL p4_tools (default: False = any one)
#     "p4_p1_bypass"  : bool               — skip P4 if ANY parameter's P1 is satisfied
#     "params"        : {
#       parameter_name → {
#         "p1_pattern"    : re.Pattern | None  — P1 regex
#         "p1_from_enum"  : bool               — auto-generate P1 from schema enum values
#         "p1_from_schema": bool               — auto-generate P1 from schema number type
#         "p2_pref_cat"   : str | None         — dotted preference category for P2 check
#         "p5_required"   : bool               — P5: must ask user (no resolvable default)
#         "p5_p1_pattern" : re.Pattern | None  — P5-specific P1 clarification  
#       }
#     }
#   }
#

PARAM_RESOLUTION_REGISTRY: dict[str, dict] = {



    "set_fan_speed": {

        "p4_tools":     ["get_climate_settings"],
        "p4_all":       False,
        "p4_p1_bypass": False,
        "params": {
            "level": {

                "p1_pattern": re.compile(
                    r'\b(level\s*\d|\d\s*level|speed\s*\d|max(?:imum)?|minimum|full|off)\b', re.I
                ),

                "p2_pref_cat": "vehicle_settings.climate_control",
            },
        },
    },

    "set_fan_airflow_direction": {
        "p4_tools":     ["get_climate_settings"],
        "p4_all":       False,
        "p4_p1_bypass": False,
        "params": {
            "direction": {

                "p1_from_enum": True,
                "p2_pref_cat":  "vehicle_settings.climate_control",
            },
        },
    },

    "set_air_circulation": {
        "p4_tools":     ["get_climate_settings"],
        "p4_all":       False,
        "p4_p1_bypass": False,
        "params": {
            "mode": {



                "p1_pattern": re.compile(r'\b(fresh|recircul|inside|outside|auto)\b', re.I),
                "p2_pref_cat": "vehicle_settings.climate_control",
            },
        },
    },

    "set_window_defrost": {

        "p4_tools":     ["get_climate_settings"],
        "p4_all":       False,
        "p4_p1_bypass": False,
        "params": {
            "defrost_window": {

                "p1_from_enum": True,

                "p2_pref_cat":  "vehicle_settings.climate_control",
            },

        },
    },



    "set_climate_temperature": {

        "p4_tools": None,
        "params": {
            "temperature": {



                "p1_from_schema": True,
                "p2_pref_cat": "vehicle_settings.climate_control",
            },
            "seat_zone": {



                "p1_from_enum": True,
                "p2_pref_cat": "vehicle_settings.climate_control",
            },
        },
    },

    "set_seat_heating": {
        "p4_tools": ["get_seat_heating_level"],
        "params": {
            "level": {
                "p1_pattern": re.compile(
                    r'\b(level\s*\d|\d\s*level|off|max(?:imum)?)\b', re.I
                ),
                "p2_pref_cat": "vehicle_settings.climate_control",
                "p5_required": True,
            },
            "seat_zone": {



                "p1_from_enum": True,
                "p2_pref_cat": "vehicle_settings.climate_control",
            },
        },
    },



    "set_air_conditioning": {

        "p4_tools":     ["get_climate_settings", "get_vehicle_window_positions"],
        "p4_all":       True,
        "p4_p1_bypass": False,
        "params": {

        },
    },



    "open_close_sunroof": {

        "p4_tools":     ["get_sunroof_and_sunshade_position"],
        "p4_all":       False,
        "p4_p1_bypass": False,
        "params": {
            "percentage": {

                "p5_required":   True,
                "p5_p1_pattern": re.compile(
                    r'\b\d{1,3}\s*%|\bfully\b|\bcompletely\b|\ball the way\b|\bhalfway\b|\bhalf\b',
                    re.I,
                ),
            },
        },
    },

    "open_close_sunshade": {
        "p4_tools":     ["get_sunroof_and_sunshade_position"],
        "p4_all":       False,
        "p4_p1_bypass": False,
        "params": {
            "percentage": {
                "p5_required":   True,
                "p5_p1_pattern": re.compile(
                    r'\b\d{1,3}\s*%|\bfully\b|\bcompletely\b|\ball the way\b|\bhalfway\b|\bhalf\b',
                    re.I,
                ),
            },
        },
    },



    "set_ambient_lights": {
        "p4_tools": None,
        "params": {
            "lightcolor": {

                "p1_pattern": re.compile(
                    r'\b(red|blue|green|white|purple|orange|yellow|pink|cyan)\b', re.I
                ),
                "p2_pref_cat": "vehicle_settings.ambient_light",
            },

        },
    },

    "set_reading_light": {

        "p4_tools":     ["get_seats_occupancy"],
        "p4_all":       False,

        "p4_p1_bypass": True,
        "params": {
            "position": {


                "p1_from_enum": True,
            },

        },
    },



    "calculate_charging_time_by_soc": {
        "p4_tools": None,
        "params": {
            "target_state_of_charge": {

                "p1_pattern": re.compile(r'\b\d{1,3}\s*(?:%|percent)\b', re.I),

                "p2_pref_cat": "points_of_interest.charging_stations",
            },

        },
    },



    "send_email": {
        "p4_tools": None,
        "params": {
            "email_addresses": {



                "p1_pattern":  re.compile(r'(?!)', re.I),
                "p2_pref_cat": "productivity_and_communication.email",
            },

        },
    },
}

"""Vehicle-policy invariant enforcement (PolicyChecker) and the
small tool-call argument helpers it shares with the parameter guards.

`PolicyViolation` is the lightweight observation returned by every
guard when it rejects an LLM-proposed tool call. It lives here so
every guard module can import it from ``agent.policy``. The
invariants enforced by ``PolicyChecker`` are deterministic
co-action and ordering constraints over the agent's observed state
(e.g. low-beam must be on before high-beam can be on; AC-on
requires window-close + minimum fan speed); they complement, but
are independent of, the P1–P5 ambiguity-resolution cascade enforced
by ``UniversalAmbiguityGuard``.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .state import _NAV_EDITING_TOOLS

if TYPE_CHECKING:
    from .state import StateCache


# ---------------------------------------------------------------------------
# PolicyViolation — shared observation type returned by every guard
# ---------------------------------------------------------------------------

class PolicyViolation:
    """One rejected tool call: which call failed, which tool, and why."""

    def __init__(self, tool_call_id: str, tool_name: str, message: str):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.message = message


# ---------------------------------------------------------------------------
# Tool-call argument helpers
# ---------------------------------------------------------------------------

def _arg(tool_call: dict, key: str):
    """Read a single argument from a tool_call dict, returning ``None`` on parse failure."""
    try:
        return json.loads(tool_call["function"]["arguments"]).get(key)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PolicyChecker — deterministic vehicle-policy invariants (separate from the cascade)
# ---------------------------------------------------------------------------

class PolicyChecker:
    """
    Enforces the vehicle-policy invariants: deterministic co-action and
    ordering constraints over the agent's actions and the observed
    vehicle state. The invariants are evaluated against the current
    :class:`~agent.state.StateCache`; this complements, but is
    independent of, the P1–P5 ambiguity-resolution cascade.
    """

    def check(self, tool_calls: list[dict], state: "StateCache") -> list[PolicyViolation]:
        if not tool_calls:
            return []
        violations: list[PolicyViolation] = []
        violations.extend(self._check_nav(tool_calls, state))
        violations.extend(self._check_lighting(tool_calls, state))
        violations.extend(self._check_climate(tool_calls, state))
        violations.extend(self._check_sunroof(tool_calls, state))
        return violations

    def _check_nav(self, tool_calls, state):
        violations = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            if name == "set_new_navigation" and state.nav_active is True:
                violations.append(PolicyViolation(
                    tc["id"], name,
                    "Policy violation (018): set_new_navigation cannot be called when navigation is "
                    "already active. Use the appropriate editing tool instead: "
                    "navigation_replace_final_destination, navigation_replace_one_waypoint, "
                    "navigation_add_one_waypoint, navigation_delete_waypoint, "
                    "navigation_delete_destination."
                ))
            elif name in _NAV_EDITING_TOOLS and state.nav_active is False:
                violations.append(PolicyViolation(
                    tc["id"], name,
                    f"Policy violation (017): {name} can only be used when navigation is already "
                    "active. Currently navigation is inactive — use set_new_navigation to start "
                    "navigation first."
                ))
            elif name == "navigation_delete_destination":
                if len(state.nav_waypoints) <= 2:
                    violations.append(PolicyViolation(
                        tc["id"], name,
                        "Policy violation (019): Cannot delete the destination — the route has no "
                        "intermediate waypoints (only start + destination)."
                    ))
        return violations

    def _check_lighting(self, tool_calls, state):
        violations = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            if name == "set_head_lights_high_beams" and _arg(tc, "on") is True:
                if state.fog_lights_on is True:
                    violations.append(PolicyViolation(
                        tc["id"], name,
                        "Policy violation (014): High beam headlights cannot be activated when fog "
                        "lights are already on. Deactivate fog lights first."
                    ))
        fog_on_calls = [
            tc for tc in tool_calls
            if tc["function"]["name"] == "set_fog_lights" and _arg(tc, "on") is True
        ]
        for tc in fog_on_calls:
            missing = []
            if state.low_beam_on is False:
                if not _batch_has(tool_calls, "set_head_lights_low_beams", "on", True):
                    missing.append("set_head_lights_low_beams(on=true)  [low beam was OFF]")
            if state.high_beam_on is True:
                if not _batch_has(tool_calls, "set_head_lights_high_beams", "on", False):
                    missing.append("set_head_lights_high_beams(on=false)  [high beam was ON]")
            if missing:
                violations.append(PolicyViolation(
                    tc["id"], tc["function"]["name"],
                    "Policy violation (013): When activating fog lights the following co-actions "
                    f"are required but missing: {', '.join(missing)}."
                ))
        return violations

    def _check_climate(self, tool_calls, state):
        violations = []
        # Policy 010 — window defrost co-actions (front/all only)
        for tc in [t for t in tool_calls if t["function"]["name"] == "set_window_defrost"]:
            zone = str(_arg(tc, "defrost_window") or "").upper()
            if zone == "REAR":
                continue
            missing = []
            if state.fan_speed is not None and state.fan_speed < 2:
                if not (_batch_has_min(tool_calls, "set_fan_speed", "speed", 2) or
                        _batch_has_min(tool_calls, "set_fan_speed", "level", 2)):
                    missing.append("set_fan_speed(level≥2)  [current speed<2]")
            if state.fan_direction is not None and "WINDSHIELD" not in state.fan_direction.upper():
                if not _batch_has_substr(tool_calls, "set_fan_airflow_direction", "direction", "WINDSHIELD"):
                    missing.append("set_fan_airflow_direction(direction includes WINDSHIELD)")
            if state.ac_on is False:
                if not _batch_has(tool_calls, "set_air_conditioning", "on", True):
                    missing.append("set_air_conditioning(on=true)  [AC was OFF]")
            if missing:
                violations.append(PolicyViolation(
                    tc["id"], tc["function"]["name"],
                    "Policy violation (010): When activating window defrost (front/all), the "
                    f"following co-actions are required but missing: {', '.join(missing)}."
                ))
        # Policy 011 — AC on co-actions
        for tc in [t for t in tool_calls
                   if t["function"]["name"] == "set_air_conditioning" and _arg(t, "on") is True]:
            missing = []
            if state.window_positions is not None:
                for wid, pos in state.window_positions.items():
                    try:
                        pos = int(pos) if pos is not None else 0
                    except (TypeError, ValueError):
                        pos = 0
                    if pos > 20 and not _batch_has_window_close(tool_calls, wid):
                        missing.append(f"set_vehicle_window_position(window_id={wid}, ≤0%)  [was {pos}%]")
            if state.fan_speed == 0:
                if not (_batch_has_min(tool_calls, "set_fan_speed", "speed", 1) or
                        _batch_has_min(tool_calls, "set_fan_speed", "level", 1)):
                    missing.append("set_fan_speed(level=1)  [fan was at 0]")
            if missing:
                violations.append(PolicyViolation(
                    tc["id"], tc["function"]["name"],
                    "Policy violation (011): When activating AC, the following co-actions are "
                    f"required but missing: {', '.join(missing)}."
                ))
        return violations

    def _check_sunroof(self, tool_calls, state):
        violations = []
        for tc in tool_calls:
            if tc["function"]["name"] != "open_close_sunroof":
                continue
            percentage = _arg(tc, "percentage")
            if percentage is None or percentage <= 0:
                continue
            if state.sunshade_position is not None and state.sunshade_position < 100:
                if not _batch_opens_sunshade(tool_calls):
                    violations.append(PolicyViolation(
                        tc["id"], tc["function"]["name"],
                        f"Policy violation (005): Sunroof cannot be opened (percentage={percentage}%) "
                        f"because the sunshade is not fully open "
                        f"(current sunshade position: {state.sunshade_position}%). "
                        "Open the sunshade fully first, or open both in parallel."
                    ))
        return violations


# ---------------------------------------------------------------------------
# Batch-check helpers — query the proposed tool-call batch for a co-action
# ---------------------------------------------------------------------------

def _batch_has(tool_calls, tool_name, key, value):
    """True if the batch contains a call to ``tool_name`` whose ``key`` argument equals ``value``."""
    for tc in tool_calls:
        if tc["function"]["name"] == tool_name and _arg(tc, key) == value:
            return True
    return False


def _batch_has_min(tool_calls, tool_name, key, minimum):
    """True if the batch contains a call to ``tool_name`` whose ``key`` argument is ``>= minimum``."""
    for tc in tool_calls:
        if tc["function"]["name"] == tool_name:
            v = _arg(tc, key)
            try:
                if v is not None and v >= minimum:
                    return True
            except TypeError:
                pass
    return False


def _batch_has_substr(tool_calls, tool_name, key, substr):
    """True if the batch contains a call to ``tool_name`` whose ``key`` argument contains ``substr``."""
    for tc in tool_calls:
        if tc["function"]["name"] == tool_name:
            v = _arg(tc, key)
            if v is not None and substr.upper() in str(v).upper():
                return True
    return False


def _batch_has_window_close(tool_calls, window_id):
    """True if the batch closes the named window (set_vehicle_window_position with position ≤ 0)."""
    for tc in tool_calls:
        if tc["function"]["name"] == "set_vehicle_window_position":
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                continue
            if args.get("window_id") == window_id and args.get("position", 100) <= 0:
                return True
    return False


def _batch_opens_sunshade(tool_calls):
    """True if the batch fully opens the sunshade (percentage=100)."""
    for tc in tool_calls:
        if tc["function"]["name"] == "open_close_sunshade":
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                continue
            if args.get("percentage", -1) >= 100:
                return True
    return False

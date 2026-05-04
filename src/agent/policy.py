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
import re
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from .state import _NAV_EDITING_TOOLS

if TYPE_CHECKING:
    from .state import StateCache


# ---------------------------------------------------------------------------
# PolicyViolation — shared observation type returned by every guard
# ---------------------------------------------------------------------------

class PolicyViolation:
    """One rejected tool call: which call failed, which tool, and why."""

    def __init__(self, tool_call_id: str, tool_name: str, message: str, policy_id: str = ""):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.message = message
        self.policy_id = policy_id


# ---------------------------------------------------------------------------
# PolicyCapability — registry entry for one policy capability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyCapability:
    policy_id: str
    summary: str
    check: Callable[[list[dict], "StateCache", dict], list[PolicyViolation]]


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

    def check(
        self,
        tool_calls: list[dict],
        state: "StateCache",
        ctx: dict | None = None,
    ) -> list[PolicyViolation]:
        if not tool_calls:
            return []
        ctx = ctx or {}
        violations: list[PolicyViolation] = []
        for capability in POLICY_CAPABILITY_REGISTRY:
            violations.extend(capability.check(tool_calls, state, ctx))
        return violations


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

def is_affirmative(text: str | None) -> bool:
    return bool(text) and bool(_CONFIRMATION_AFFIRMATIVE.search(text))


def _arg(tool_call: dict, key: str):
    """Read a single argument from a tool_call dict, returning ``None`` on parse failure."""
    try:
        return json.loads(tool_call["function"]["arguments"]).get(key)
    except Exception:
        return None


def _args(tool_call: dict) -> dict:
    try:
        return json.loads(tool_call["function"]["arguments"])
    except Exception:
        return {}


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


def _find_route_start(route_id: str, route_options: list[dict]) -> str | None:
    if not route_id or not route_options:
        return None
    for route_group in route_options:
        if not isinstance(route_group, dict):
            continue
        for alt in route_group.get("alternatives", []) or []:
            if isinstance(alt, dict) and alt.get("route_id") == route_id:
                return route_group.get("start_id")
        if route_group.get("route_id") == route_id:
            return route_group.get("start_id")
    return None


# ---------------------------------------------------------------------------
# Policy capability checks
# ---------------------------------------------------------------------------

def _check_005(tool_calls, state, ctx):
    out = []
    for tc in tool_calls:
        if tc["function"]["name"] != "open_close_sunroof":
            continue
        percentage = _arg(tc, "percentage")
        if percentage is None or percentage <= 0:
            continue
        if state.sunshade_position is not None and state.sunshade_position < 100:
            if not _batch_opens_sunshade(tool_calls):
                out.append(PolicyViolation(
                    tc["id"], tc["function"]["name"],
                    f"Policy violation (005): Sunroof cannot be opened (percentage={percentage}%) "
                    f"because the sunshade is not fully open "
                    f"(current sunshade position: {state.sunshade_position}%). "
                    "Open the sunshade fully first, or open both in parallel.",
                    policy_id="005",
                ))
    return out


def _check_009(tool_calls, state, ctx):
    out = []
    cur_loc_id = (state.current_location or {}).get("id") if state.current_location else None
    weather = state.weather_by_location.get(cur_loc_id) if cur_loc_id else None
    if weather is None and len(state.weather_by_location) == 1:
        weather = next(iter(state.weather_by_location.values()))
    confirmed = is_affirmative(ctx.get("last_user_msg", ""))

    for tc in tool_calls:
        name = tc["function"]["name"]
        if name == "open_close_sunroof" and (_arg(tc, "percentage") or 0) > 0:
            safe = _SAFE_WEATHER_REGISTRY["sunroof_opening"]
            kind = "sunroof opening"
        elif name == "set_fog_lights" and _arg(tc, "on") is True:
            safe = _SAFE_WEATHER_REGISTRY["fog_light_activation"]
            kind = "fog-light activation"
        else:
            continue

        if not cur_loc_id and weather is None:
            continue
        if weather is None:
            location_hint = cur_loc_id or "the current location"
            out.append(PolicyViolation(
                tc["id"], name,
                f"Policy violation (009): {kind} requires checking current weather "
                f"first via get_weather for {location_hint}.",
                policy_id="009",
            ))
            continue
        condition = ((weather.get("current_slot") or {}).get("condition") or "").lower()
        if condition and condition not in safe and not confirmed:
            out.append(PolicyViolation(
                tc["id"], name,
                f"Policy violation (009): weather is '{condition}', which is not in the "
                f"safe set for {kind}. Explicit user confirmation (yes) required.",
                policy_id="009",
            ))
    return out


def _check_010(tool_calls, state, ctx):
    out = []
    for tc in [t for t in tool_calls if t["function"]["name"] == "set_window_defrost"]:
        zone = str(_arg(tc, "defrost_window") or "").upper()
        if zone == "REAR":
            continue
        missing = []
        if state.fan_speed is not None and state.fan_speed < 2:
            if not (_batch_has_min(tool_calls, "set_fan_speed", "speed", 2) or
                    _batch_has_min(tool_calls, "set_fan_speed", "level", 2)):
                missing.append("set_fan_speed(level>=2)  [current speed<2]")
        if state.fan_direction is not None and "WINDSHIELD" not in state.fan_direction.upper():
            if not _batch_has_substr(tool_calls, "set_fan_airflow_direction", "direction", "WINDSHIELD"):
                missing.append("set_fan_airflow_direction(direction includes WINDSHIELD)")
        if state.ac_on is False:
            if not _batch_has(tool_calls, "set_air_conditioning", "on", True):
                missing.append("set_air_conditioning(on=true)  [AC was OFF]")
        if missing:
            out.append(PolicyViolation(
                tc["id"], tc["function"]["name"],
                "Policy violation (010): When activating window defrost (front/all), the "
                f"following co-actions are required but missing: {', '.join(missing)}.",
                policy_id="010",
            ))
    return out


def _check_011(tool_calls, state, ctx):
    out = []
    for tc in [t for t in tool_calls
               if t["function"]["name"] == "set_air_conditioning" and _arg(t, "on") is True]:
        missing = []
        if state.window_positions is not None:
            for window_id, pos in state.window_positions.items():
                try:
                    pos = int(pos) if pos is not None else 0
                except (TypeError, ValueError):
                    pos = 0
                if pos > 20 and not _batch_has_window_close(tool_calls, window_id):
                    missing.append(f"set_vehicle_window_position(window_id={window_id}, <=0%)  [was {pos}%]")
        if state.fan_speed == 0:
            if not (_batch_has_min(tool_calls, "set_fan_speed", "speed", 1) or
                    _batch_has_min(tool_calls, "set_fan_speed", "level", 1)):
                missing.append("set_fan_speed(level=1)  [fan was at 0]")
        if missing:
            out.append(PolicyViolation(
                tc["id"], tc["function"]["name"],
                "Policy violation (011): When activating AC, the following co-actions are "
                f"required but missing: {', '.join(missing)}.",
                policy_id="011",
            ))
    return out


def _check_013(tool_calls, state, ctx):
    out = []
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
            out.append(PolicyViolation(
                tc["id"], tc["function"]["name"],
                "Policy violation (013): When activating fog lights the following co-actions "
                f"are required but missing: {', '.join(missing)}.",
                policy_id="013",
            ))
    return out


def _check_014(tool_calls, state, ctx):
    out = []
    for tc in tool_calls:
        if tc["function"]["name"] != "set_head_lights_high_beams":
            continue
        if _arg(tc, "on") is True and state.fog_lights_on is True:
            out.append(PolicyViolation(
                tc["id"], tc["function"]["name"],
                "Policy violation (014): High beam headlights cannot be activated when fog "
                "lights are already on. Deactivate fog lights first.",
                policy_id="014",
            ))
    return out


def _check_016(tool_calls, state, ctx):
    out = []
    if state.current_location is None:
        return out
    current_id = state.current_location.get("id")
    if not current_id:
        return out
    for tc in tool_calls:
        if tc["function"]["name"] != "set_new_navigation":
            continue
        route_ids = _arg(tc, "route_ids") or []
        if not route_ids:
            continue
        start_id = _find_route_start(route_ids[0], state.last_route_options)
        if start_id is not None and start_id != current_id:
            out.append(PolicyViolation(
                tc["id"], "set_new_navigation",
                f"Policy violation (016): Start of the route must be the current car "
                f"location (id={current_id}), but the first route's start is {start_id}. "
                "Re-fetch routes with start=current location.",
                policy_id="016",
            ))
    return out


def _check_017(tool_calls, state, ctx):
    out = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        if name in _NAV_EDITING_TOOLS and state.nav_active is False:
            out.append(PolicyViolation(
                tc["id"], name,
                f"Policy violation (017): {name} can only be used when navigation is already "
                "active. Currently navigation is inactive; use set_new_navigation to start "
                "navigation first.",
                policy_id="017",
            ))
    return out


def _check_018(tool_calls, state, ctx):
    out = []
    for tc in tool_calls:
        if tc["function"]["name"] == "set_new_navigation" and state.nav_active is True:
            out.append(PolicyViolation(
                tc["id"], "set_new_navigation",
                "Policy violation (018): set_new_navigation cannot be called when navigation is "
                "already active. Use the appropriate editing tool instead: "
                "navigation_replace_final_destination, navigation_replace_one_waypoint, "
                "navigation_add_one_waypoint, navigation_delete_waypoint, "
                "navigation_delete_destination.",
                policy_id="018",
            ))
    return out


def _check_019(tool_calls, state, ctx):
    out = []
    for tc in tool_calls:
        if tc["function"]["name"] != "navigation_delete_destination":
            continue
        if len(state.nav_waypoints) <= 2:
            out.append(PolicyViolation(
                tc["id"], "navigation_delete_destination",
                "Policy violation (019): Cannot delete the destination; the route has no "
                "intermediate waypoints (only start + destination).",
                policy_id="019",
            ))
    return out


_ROUTE_SELECTION_EXPLICIT = re.compile(
    r"\b(fastest|shortest|first|second|third|route\s+option|via\s+[A-Z0-9])\b",
    re.I,
)


def _check_single_segment_route_selection(tool_calls, state, ctx):
    out = []
    if state.route_selection_preference_present:
        return out
    user_msg = ctx.get("last_user_msg", "") or ""
    if _ROUTE_SELECTION_EXPLICIT.search(user_msg):
        return out

    for tc in tool_calls:
        name = tc["function"]["name"]
        single_segment_edit = (
            name == "navigation_replace_final_destination" and len(state.nav_waypoints) <= 2
        ) or (
            name == "navigation_delete_waypoint" and len(state.nav_waypoints) == 3
        )
        if not single_segment_edit:
            continue
        out.append(PolicyViolation(
            tc["id"], name,
            "Ambiguity violation: This navigation edit leaves one direct route segment and "
            "the user has not selected a route option. Present the fastest and shortest "
            "routes and ask which route to use before editing navigation.",
            policy_id="AMB_SINGLE_SEGMENT_ROUTE",
        ))
    return out


def _check_reading_light_all(tool_calls, state, ctx):
    out = []
    user_msg = ctx.get("last_user_msg", "") or ""
    explicit_all = re.search(r"\b(all|both|everyone|everybody|all\s+reading\s+lights)\b", user_msg, re.I)
    for tc in tool_calls:
        if tc["function"]["name"] != "set_reading_light":
            continue
        if str(_arg(tc, "position") or "").upper() != "ALL":
            continue
        if explicit_all:
            continue
        out.append(PolicyViolation(
            tc["id"], "set_reading_light",
            "Ambiguity violation: The user did not explicitly ask for all reading lights. "
            "Ask which reading light position they want, or use the specific position if it "
            "is explicitly resolved.",
            policy_id="AMB_READING_LIGHT",
        ))
    return out


def _check_023(tool_calls, state, ctx):
    if state.current_datetime is None:
        return []
    current_month = state.current_datetime.get("month")
    current_day = state.current_datetime.get("day")
    out = []
    for tc in tool_calls:
        if tc["function"]["name"] != "get_entries_from_calendar":
            continue
        month, day = _arg(tc, "month"), _arg(tc, "day")
        if (month, day) != (current_month, current_day):
            out.append(PolicyViolation(
                tc["id"], "get_entries_from_calendar",
                f"Policy violation (023): Calendar entries can only be requested for the "
                f"current day (month={current_month}, day={current_day}); got month={month}, day={day}.",
                policy_id="023",
            ))
    return out


def _check_024(tool_calls, state, ctx):
    if state.current_datetime is None:
        return []
    current_month = state.current_datetime.get("month")
    current_day = state.current_datetime.get("day")
    out = []
    for tc in tool_calls:
        if tc["function"]["name"] != "get_weather":
            continue
        month, day = _arg(tc, "month"), _arg(tc, "day")
        if (month, day) != (current_month, current_day):
            out.append(PolicyViolation(
                tc["id"], "get_weather",
                f"Policy violation (024): Weather can only be requested for the current day "
                f"(month={current_month}, day={current_day}); got month={month}, day={day}.",
                policy_id="024",
            ))
        elif _arg(tc, "time_hour_24hformat") is None:
            out.append(PolicyViolation(
                tc["id"], "get_weather",
                "Policy violation (024): Weather request must specify time_hour_24hformat.",
                policy_id="024",
            ))
    return out


# ---------------------------------------------------------------------------
# Policy registries
# ---------------------------------------------------------------------------

_CONFIRMATION_AFFIRMATIVE = re.compile(
    r"\b(yes|yep|yeah|sure|ok|okay|confirm|go\s+ahead|proceed|please\s+do|do\s+it)\b",
    re.I,
)

_SAFE_WEATHER_REGISTRY = {
    "sunroof_opening": {"sunny", "cloudy", "partly_cloudy"},
    "fog_light_activation": {"cloudy_and_thunderstorm", "cloudy_and_hail"},
}

POLICY_CAPABILITY_REGISTRY: list[PolicyCapability] = [
    PolicyCapability("005", "Sunroof needs sunshade open", _check_005),
    PolicyCapability("009", "Weather confirmation: sunroof/fog", _check_009),
    PolicyCapability("010", "Window defrost co-actions", _check_010),
    PolicyCapability("011", "AC-on co-actions", _check_011),
    PolicyCapability("013", "Fog-light co-actions", _check_013),
    PolicyCapability("014", "High beam vs fog conflict", _check_014),
    PolicyCapability("016", "Route start = current location", _check_016),
    PolicyCapability("017", "Editing tools require active nav", _check_017),
    PolicyCapability("018", "Use editing tools when nav active", _check_018),
    PolicyCapability("019", "Cannot delete sole destination", _check_019),
    PolicyCapability("AMB_SINGLE_SEGMENT_ROUTE", "Single-segment edits need route choice", _check_single_segment_route_selection),
    PolicyCapability("AMB_READING_LIGHT", "Reading light ALL requires explicit all", _check_reading_light_all),
    PolicyCapability("023", "Calendar restricted to current day", _check_023),
    PolicyCapability("024", "Weather restricted to current day", _check_024),
]

"""Per-context working memory for the purple agent.

Each conversation context owns one :class:`StateCache`. The cache is
updated after every tool result via :meth:`StateCache.update`, turning
raw tool observations (climate settings, window positions, light
status, navigation state, sunshade position) into a structured
snapshot of the world. The policy and ambiguity guards consult that
snapshot before the next LLM turn so they can enforce vehicle-policy
invariants and the P1–P5 disambiguation cascade against an
authoritative view of the vehicle, not against the LLM's own claims.
"""
from __future__ import annotations

import json
import re


def _parse_bool(value) -> bool | None:
    """Coerce raw tool-result fields (bool / int / common string spellings) into ``bool | None``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "on", "yes", "1", "active")
    if isinstance(value, (int, float)):
        return bool(value)
    return None


class StateCache:
    """Working memory of vehicle observations for one conversation context.

    The cache is populated from tool results so the policy guard and the
    ambiguity guard can verify their preconditions against an
    authoritative snapshot — never against the LLM's own claims about
    the world.
    """

    def __init__(self):
        self.nav_active: bool | None = None
        self.nav_waypoints: list = []
        self.fog_lights_on: bool | None = None
        self.low_beam_on: bool | None = None
        self.high_beam_on: bool | None = None
        self.ac_on: bool | None = None
        self.fan_speed: int | None = None
        self.fan_direction: str | None = None
        self.window_positions: dict | None = None
        self.sunshade_position: int | None = None
        self.last_route_options: list[dict] = []
        self.current_location: dict | None = None
        self.current_datetime: dict | None = None
        self.weather_by_location: dict[str, dict] = {}

    def parse_system_prompt(self, system_text: str) -> None:
        if not system_text:
            return
        for label, attr in (("CURRENT_LOCATION", "current_location"), ("DATETIME", "current_datetime")):
            match = re.search(rf"{label}\s*=\s*(\{{.*?\}})", system_text, re.DOTALL)
            if not match:
                continue
            try:
                setattr(self, attr, json.loads(match.group(1)))
            except Exception:
                pass

    def update(self, tool_name: str, args: dict, result_content: str) -> None:
        """Fold a single tool observation into the working-memory snapshot."""
        try:
            result = json.loads(result_content) if result_content else {}
        except Exception:
            result = {}
        if isinstance(result, dict) and "result" in result and isinstance(result["result"], dict):
            result = result["result"]

        if tool_name == "get_current_navigation_state":
            nav_active = result.get("navigation_active")
            if nav_active is not None:
                self.nav_active = bool(nav_active)
            if "waypoints_id" in result:
                self.nav_waypoints = result["waypoints_id"] or []
        elif tool_name == "get_exterior_lights_status":
            lights = result.get("lights", result)
            if "fog_lights" in lights:
                self.fog_lights_on = _parse_bool(lights["fog_lights"])
            if "low_beam_headlights" in lights:
                self.low_beam_on = _parse_bool(lights["low_beam_headlights"])
            if "high_beam_headlights" in lights:
                self.high_beam_on = _parse_bool(lights["high_beam_headlights"])
        elif tool_name == "get_climate_settings":
            if "air_conditioning" in result:
                self.ac_on = _parse_bool(result["air_conditioning"])
            if "fan_speed" in result:
                v = result["fan_speed"]
                if isinstance(v, (int, float)):
                    self.fan_speed = int(v)
            if "fan_airflow_direction" in result:
                self.fan_direction = result["fan_airflow_direction"]
        elif tool_name == "get_vehicle_window_positions":
            windows = result.get("windows", result)
            if isinstance(windows, list):
                parsed = {}
                for w in windows:
                    if isinstance(w, dict) and w.get("window_id") is not None:
                        try:
                            parsed[w["window_id"]] = int(w.get("position", 0))
                        except (TypeError, ValueError):
                            parsed[w["window_id"]] = 0
                self.window_positions = parsed
            elif isinstance(windows, dict):
                parsed = {}
                for k, v in windows.items():
                    try:
                        parsed[k] = int(v)
                    except (TypeError, ValueError):
                        parsed[k] = 0
                self.window_positions = parsed
        elif tool_name == "get_sunroof_and_sunshade_position":
            if "sunshade_position" in result:
                v = result["sunshade_position"]
                if isinstance(v, (int, float)):
                    self.sunshade_position = int(v)
        elif tool_name == "open_close_sunshade" and "percentage" in args:
            self.sunshade_position = int(args["percentage"])
        elif tool_name == "set_fog_lights" and "on" in args:
            self.fog_lights_on = bool(args["on"])
        elif tool_name == "set_head_lights_high_beams" and "on" in args:
            self.high_beam_on = bool(args["on"])
        elif tool_name == "set_head_lights_low_beams" and "on" in args:
            self.low_beam_on = bool(args["on"])
        elif tool_name == "set_air_conditioning" and "on" in args:
            self.ac_on = bool(args["on"])
        elif tool_name == "set_fan_speed":
            v = args.get("speed") if "speed" in args else args.get("level")
            if v is not None:
                try:
                    self.fan_speed = int(v)
                except (TypeError, ValueError):
                    pass
        elif tool_name == "set_fan_airflow_direction" and "direction" in args:
            self.fan_direction = str(args["direction"])
        elif tool_name == "set_new_navigation":
            self.nav_active = True
            waypoints = args.get("waypoints", [])
            if isinstance(waypoints, list):
                self.nav_waypoints = [
                    w.get("id") or w.get("waypoint_id") or w.get("location_id")
                    for w in waypoints if isinstance(w, dict)
                ]
        elif tool_name == "get_routes_from_start_to_destination":
            routes = result.get("routes", [])
            if isinstance(routes, list) and routes:
                self.last_route_options = routes
        elif tool_name == "get_weather":
            location_id = args.get("location_or_poi_id")
            if location_id and isinstance(result, dict):
                self.weather_by_location[location_id] = result
        elif tool_name in _NAV_EDITING_TOOLS:
            self.nav_active = True


# ──────────────────────────────────────────────────────────────

# Tool names that mutate an active navigation route. PolicyChecker uses
# this to enforce Policy 017 (editing tools require active nav) and
# StateCache uses it to mark navigation active after a successful edit.
_NAV_EDITING_TOOLS = frozenset({
    "navigation_add_one_waypoint",
    "navigation_delete_waypoint",
    "navigation_replace_one_waypoint",
    "navigation_replace_final_destination",
    "navigation_delete_destination",
})

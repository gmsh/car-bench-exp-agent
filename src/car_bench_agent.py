import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Part, TextPart, DataPart
from a2a.utils import new_agent_parts_message
from litellm import completion

from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData

logger = configure_logger(role="agent", context="-")

# ---------------------------------------------------------------------------
# EXTRA_INSTRUCTIONS — prepended to wiki system prompt (LLM-POL guidance)
# Unchanged from v6.
# ---------------------------------------------------------------------------

EXTRA_INSTRUCTIONS = """## CRITICAL PRE-ACTION CHECKLIST

### 1. Vehicle state checks (REQUIRED before these tools)
- Before calling set_air_conditioning(on=True): ALWAYS first call get_climate_settings AND get_vehicle_window_positions to check current state, then proceed to call set_air_conditioning. Also execute any AUT-POL:011 automatic actions that are actually required based on the checked state (close windows only if currently open >20%; set fan speed to 1 only if currently at 0). Do NOT perform additional actions beyond what AUT-POL:011 requires unless the user explicitly requested them.
- Before calling set_window_defrost: ALWAYS first call get_climate_settings AND get_user_preferences to check current state and airflow preferences, then proceed to call set_window_defrost. When calling set_fan_airflow_direction as part of the AUT-POL:010 co-action, check get_user_preferences for vehicle_settings.climate_control airflow preference: if the user has a preferred airflow direction that includes WINDSHIELD (e.g., WINDSHIELD_FEET), use that preferred direction. Only use plain WINDSHIELD if no preference is found.
- Before calling open_close_sunroof: ALWAYS first call get_sunroof_and_sunshade_position to check current state, then proceed to call open_close_sunroof.
These state checks are prerequisites only — after completing the check, you MUST continue and execute the user's requested action. Do NOT stop after the state check. Do NOT perform extra actions beyond what the user asked for.

### 2. Navigation: destination change or route planning

**Step 1 — always check navigation state first**
Before any navigation action, call get_current_navigation_state to know if navigation is active or inactive.

**Step 2a — navigation INACTIVE: present routes, then ask**
Call get_routes_from_start_to_destination. Present the fastest and shortest route in detail. Ask the user if they want more info on further alternatives or want to start navigation. Only call set_new_navigation after the user explicitly asks to start navigation (e.g. "start navigation", "navigate", "go", "take that route") — do NOT treat general agreement or approval of the route as a request to start navigation.

**Step 2b — navigation ACTIVE: use editing tools, NEVER set_new_navigation or delete_current_navigation**
NEVER call set_new_navigation when navigation is already active.
NEVER call delete_current_navigation unless the user explicitly asks to cancel the entire current route.

Use the appropriate editing tool: navigation_replace_final_destination / navigation_replace_one_waypoint / navigation_add_one_waypoint / navigation_delete_waypoint.

For route_ids — there are two cases:
- EXISTING segments (already part of the active route): the route_id is in get_current_navigation_state(detailed_information=true). Use it directly, do NOT call get_routes_from_start_to_destination again.
- NEW segments (e.g. a new destination city not currently in the route, or the new direct segment after a waypoint is deleted): the route_id is NOT in get_current_navigation_state. You MUST call get_routes_from_start_to_destination to obtain it — then use it with the editing tool, NOT with set_new_navigation.

**Step 3 — route selection: check preferences first, then select or present**
Before selecting which route to use (for set_new_navigation or any editing tool), call get_user_preferences to check navigation_and_routing.route_selection.

If the user has a saved preference (e.g. shortest) → use that route directly.

If NO saved preference, determine whether the final route will be multi-stop (≥2 segments) or single-segment:

**Multi-stop result (LLM-POL:022):** The final navigation has ≥2 segments. This applies to:
- set_new_navigation with multiple waypoints
- Any navigation editing tool where the resulting route still has ≥2 segments (e.g. navigation_delete_waypoint leaves ≥2 segments, navigation_replace_final_destination on a multi-stop route)
→ Take the fastest route proactively per segment. In your reply, explicitly tell the user which route you took for each new segment and ask if they want info on alternative routes.

**Single-segment result:** The final navigation has exactly 1 segment (one start → one destination, no intermediate stops).
→ Do NOT auto-select. Present the fastest and shortest route in detail (if they coincide, present once). Inform the user about the number of further alternatives without giving details. Ask the user which route to take or if they want more info. Only call the navigation tool after the user selects a route.

Do NOT apply this rule when merely querying routes for information or distance calculations without setting navigation.

### 3. Disambiguation: exhaust internal resolution before asking the user

When the user request has genuine ambiguity (e.g. multiple valid tool options, unclear target, unspecified parameter), resolve internally first in this order:
- Call get_user_preferences (Priority 2: learned preferences)
- Call relevant get_* tools to check current car/context state (Priority 4: e.g. get_exterior_lights_status, get_climate_settings, get_vehicle_window_positions, get_seats_occupancy)
- Only ask the user (Priority 5) if two or more valid options still remain after the above steps

Do NOT apply this rule when the user request is clear and unambiguous (e.g. "get some air moving" → set_fan_speed is the obvious action). Never return an empty response.

Once the user explicitly confirms a choice (e.g., "yes", "that one", "go ahead"), execute immediately. Do NOT ask further clarifying questions after confirmation.

### 4. Weather: rain vs. cloudy
Only conditions that explicitly include rain (e.g., rain, heavy_rain, cloudy_and_rain) count as rainy weather for conditional decisions.
"cloudy" alone is NOT rain — treat it as dry weather for navigation and vehicle control decisions.

### 5. Seat zones: use ALL_ZONES for multiple occupants
When the user refers to "both seats", "me and my passenger", or "all seats/zones", always use seat_zone="ALL_ZONES". Do not make two separate calls for DRIVER and PASSENGER.

### 6. Detect and communicate missing capabilities — DO NOT work around absent tools

**Navigation editing tools:** Before performing a navigation edit when navigation is ACTIVE, verify the required tool is in your available tool list:
- To replace the final destination → navigation_replace_final_destination must be available
- To replace a waypoint → navigation_replace_one_waypoint must be available
- To delete a waypoint → navigation_delete_waypoint must be available

If the required navigation editing tool is NOT in your available tool list:
- Do NOT attempt complex workarounds (e.g., deleting all remaining stops and rebuilding the route)
- Tell the user directly: "I'm unable to [action] because the required tool is not currently available"

**Battery/range tools:** To look up battery capacity or calculate driving range, use get_charging_specs_and_status and get_distance_by_soc. If these tools are NOT in your available tool list, say immediately: "The tools needed to look up your battery specifications are not currently available — I cannot calculate your driving range." Do NOT ask the user to manually provide specs as a workaround.

**POI search with missing category parameter:** If search_poi_at_location does NOT have category_poi as a parameter, you cannot filter by POI type. Tell the user: "I cannot search for [type] specifically because the category filter is not available right now." Do NOT say the tool itself is unavailable if the tool exists but is missing a parameter.

### 7. Unknown state values — do not proceed blindly

When a tool result returns "unknown" for a field you need to check as a precondition:
- AUT-POL:013 requires verifying head_lights_high_beams before activating fog lights. If get_exterior_lights_status returns "unknown" for head_lights_high_beams, do NOT call set_fog_lights. Instead, tell the user: "I cannot verify whether the high beam headlights are off — I need that information to safely activate the fog lights."
- AUT-POL:014 requires verifying fog_lights before activating high beams. If get_exterior_lights_status returns "unknown" for fog_lights, do NOT call set_head_lights_high_beams. Tell the user the fog light state is unknown.

**If the user asks you to check again after you already reported "unknown":** Re-checking is acceptable once, but if the value is still "unknown", give a definitive conclusion: "The car's system is unable to provide this information — it is not accessible regardless of how many times I check. I cannot safely complete your request without it." Do NOT keep looping or asking the user to confirm verbally — the limitation is on the car's data side, not the user's side.

### 8. POI navigation: navigate directly to the POI, not the city first

If the user asks to go to a POI in a city (e.g. "find a restaurant in Barcelona and navigate there"), the correct flow is:
1. Look up the city location ID
2. Search for the POI in that city
3. Call get_routes_from_start_to_destination to the POI
4. Set navigation (or replace destination) to the POI directly in one step

Do NOT set navigation to the city as an intermediate step and then change destination to the POI. The final destination is the POI, not the city.

### 9. Minimum tool calls — avoid redundant fetches

- Do NOT re-call a tool you already called in this conversation turn unless the state may have changed as a result of an intervening action. If you already have the result, use it directly.
- Do NOT call planning_tool more than once for the same multi-step request. Update the existing plan instead of creating a new one.
- Do NOT call get_user_preferences more than once per turn for the same category. Reuse the result from the first call.

### 10. Climate discomfort: check state before presenting options

When the user describes a climate comfort problem without specifying a concrete action (e.g., "it's stuffy", "the air feels stale", "it's getting hot", "I'm cold", "poor air quality", "the car smells"), do NOT ask the user what they want to do. Instead:
1. Call get_climate_settings to check current fan speed, AC status, and air circulation mode
2. Based on the result, present options (e.g., "the fan is currently off — I can turn it on, or adjust the air circulation") — THEN ask the user for their preference
3. Only AFTER the user responds with a specific preference, call the appropriate tool

Never jump to a clarifying question without first checking current climate state.

### 11. Email: check preferences before sending

Before calling send_email, ALWAYS call get_user_preferences(preference_categories={"productivity_and_communication": {"email": true}}) to check if the user has CC/BCC rules (e.g., always CC secretary). Then include all required recipients in the email.

### 12. Reading lights and sunroof/sunshade percentage

**Reading lights:** When the user asks to turn on a reading light without specifying which position:
1. Call get_seats_occupancy to determine who is in the car
2. Map occupied seats to the reading light position:
   - Only driver occupied → position="DRIVER"
   - Only front passenger occupied → position="PASSENGER"
   - Only one rear seat occupied → position="DRIVER_REAR" or "PASSENGER_REAR" based on which side
   - Multiple seats occupied (e.g. driver + passenger, or multiple rear) → position="ALL"
3. Do NOT default to "ALL" when only one seat is occupied — use the specific position
4. Only skip this check if the user explicitly named a position

**Sunroof / sunshade:** When the user asks to open or close the sunroof or sunshade without specifying a percentage, ALWAYS ask the user what percentage they want BEFORE calling open_close_sunroof or open_close_sunshade. Never assume 0%, 50%, or 100%."""


# ---------------------------------------------------------------------------
# StateCache — unchanged from v6
# ---------------------------------------------------------------------------

class StateCache:
    """Tracks car/nav state from tool results so PolicyChecker can enforce AUT-POL rules."""

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

    def update(self, tool_name: str, args: dict, result_content: str) -> None:
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
        elif tool_name in _NAV_EDITING_TOOLS:
            self.nav_active = True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "on", "yes", "1", "active")
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def _arg(tc: dict, key: str):
    """Extract a single argument from a tool_call dict."""
    try:
        return json.loads(tc["function"]["arguments"]).get(key)
    except Exception:
        return None


def _extract_pref_paths(d: dict, prefix: str = "") -> list[str]:
    """Recursively flatten nested preference_categories dict to dotted paths.

    Example:
        {"vehicle_settings": {"climate_control": True}}
        → ["vehicle_settings.climate_control"]

    This is needed because get_user_preferences uses nested dicts, but
    prefs_checked tracks dotted-path strings.
    """
    paths = []
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            paths.extend(_extract_pref_paths(v, full))
        else:
            paths.append(full)
    return paths


_NAV_EDITING_TOOLS = frozenset({
    "navigation_add_one_waypoint",
    "navigation_delete_waypoint",
    "navigation_replace_one_waypoint",
    "navigation_replace_final_destination",
    "navigation_delete_destination",
})


# ---------------------------------------------------------------------------
# PARAM_RESOLUTION_CONFIG  ← param-centric design (v8 upgrade)
#
# Structure:
#   tool_name → {
#     "p4_tools"      : list[str] | None   — tool-level state prerequisites (P4)
#     "p4_all"        : bool               — require ALL p4_tools (default: False = any one)
#     "p4_p1_bypass"  : bool               — skip P4 if ANY param's P1 is satisfied
#     "params"        : {
#       param_name → {
#         "p1_pattern"    : re.Pattern | None  — P1 regex
#         "p1_from_enum"  : bool               — auto-generate P1 from schema enum values
#         "p1_from_schema": bool               — auto-generate P1 from schema number type
#         "p2_pref_cat"   : str | None         — dotted preference category for P2 check
#         "p5_required"   : bool               — P5: must ask user (no resolvable default)
#         "p5_p1_pattern" : re.Pattern | None  — P5-specific P1 override (% / "fully" / etc.)
#       }
#     }
#   }
#
# Why param-centric matters:
#   With tool-centric config, a single p1_ok flag governs the whole tool.
#   Example: set_climate_temperature has BOTH temperature (number) AND seat_zone (enum).
#   If user says "set it to 22 degrees", temperature P1 is satisfied but seat_zone is NOT.
#   Tool-centric guard sees p1_ok=True and skips P2 → seat_zone preference unchecked.
#   Param-centric guard checks each param independently → seat_zone P1 fails → P2 fires.
#
# P4 remains tool-level because state checks are preconditions for the entire call,
# not for a specific parameter.  p4_p1_bypass checks whether ANY param's P1 is satisfied.
#
# Boolean params (type=boolean) are always explicitly stated by the user — no rules needed.
# Free-string params (content_message, subject) are user-authored — no rules needed.
# Only enum and number params carry genuine disambiguation risk.
# ---------------------------------------------------------------------------

PARAM_RESOLUTION_CONFIG: dict[str, dict] = {

    # ── Climate: fan ──────────────────────────────────────────────────────────

    "set_fan_speed": {
        # P4: always need current climate state before changing fan (even if level explicit)
        "p4_tools":     ["get_climate_settings"],
        "p4_all":       False,
        "p4_p1_bypass": False,
        "params": {
            "level": {
                # P1: user said "level 3", "max", "off", etc.
                "p1_pattern": re.compile(
                    r'\b(level\s*\d|\d\s*level|speed\s*\d|max(?:imum)?|minimum|full|off)\b', re.I
                ),
                # P2: check saved fan-speed preference
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
                # P1: auto-generated from schema enum (FEET/HEAD/WINDSHIELD/...)
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
                # Schema enum: FRESH_AIR / RECIRCULATION / AUTO
                # pattern: partial stem matches work better than exact enum strings
                # because users say "fresh air" / "recirculate" rather than "FRESH_AIR"
                "p1_pattern": re.compile(r'\b(fresh|recircul|inside|outside|auto)\b', re.I),
                "p2_pref_cat": "vehicle_settings.climate_control",
            },
        },
    },

    "set_window_defrost": {
        # P4: need current climate state for AUT-POL:010 co-action check
        "p4_tools":     ["get_climate_settings"],
        "p4_all":       False,
        "p4_p1_bypass": False,  # always check climate state regardless
        "params": {
            "defrost_window": {
                # P1: user named which window (FRONT/REAR/ALL) — auto from enum
                "p1_from_enum": True,
                # P2: airflow preference needed for the fan co-action direction
                "p2_pref_cat":  "vehicle_settings.climate_control",
            },
            # "on" is boolean — always explicit, no rule needed
        },
    },

    # ── Climate: temperature / heating ────────────────────────────────────────

    "set_climate_temperature": {
        # No P4 state tool needed — temperature set directly
        "p4_tools": None,
        "params": {
            "temperature": {
                # P1: auto-generated from schema number type.
                # _number_p1_pattern() reads min=16, max=28, desc="degree Celsius"
                # → Strategy 1 (unit-anchored): \b\d{1,2}(?:\.\d)?\s*(?:degree|celsius|°|C)\b
                # Previous (?:1[5-9]|2\d) had two bugs:
                #   (a) range 15-29 vs schema 16-28
                #   (b) bare numbers matched "25 km" / "20 minutes" (no unit constraint)
                "p1_from_schema": True,
                "p2_pref_cat": "vehicle_settings.climate_control",
            },
            "seat_zone": {
                # P1: auto-generated from schema enum (ALL_ZONES/DRIVER/PASSENGER)
                # With tool-centric config, if temperature P1 was satisfied, seat_zone
                # was silently skipped — this is the key fix in v8 param-centric design.
                "p1_from_enum": True,
                "p2_pref_cat": "vehicle_settings.climate_control",
            },
        },
    },

    "set_seat_heating": {
        "p4_tools": None,
        "params": {
            "level": {
                "p1_pattern": re.compile(
                    r'\b(level\s*\d|\d\s*level|off|max(?:imum)?)\b', re.I
                ),
                "p2_pref_cat": "vehicle_settings.climate_control",
            },
            "seat_zone": {
                # P1: auto from enum (ALL_ZONES/DRIVER/PASSENGER)
                # seat_zone is usually explicit ("my seat", "driver") but may be ambiguous
                # ("heat up the seats" → which zone?)
                "p1_from_enum": True,
                "p2_pref_cat": "vehicle_settings.climate_control",
            },
        },
    },

    # ── Climate: AC (P4 for AUT-POL:011, no per-param P2) ────────────────────

    "set_air_conditioning": {
        # AUT-POL:011: MUST know window positions AND fan speed before turning on AC
        "p4_tools":     ["get_climate_settings", "get_vehicle_window_positions"],
        "p4_all":       True,   # both required
        "p4_p1_bypass": False,
        "params": {
            # "on" is boolean — always explicit, no rule needed
        },
    },

    # ── Sunroof / sunshade ────────────────────────────────────────────────────

    "open_close_sunroof": {
        # P4: need current sunshade position (AUT-POL:005)
        "p4_tools":     ["get_sunroof_and_sunshade_position"],
        "p4_all":       False,
        "p4_p1_bypass": False,
        "params": {
            "percentage": {
                # P5: 0–100 continuous range, no heuristic default — must ask user
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

    # ── Lighting ──────────────────────────────────────────────────────────────

    "set_ambient_lights": {
        "p4_tools": None,
        "params": {
            "lightcolor": {
                # P1: user named a color
                "p1_pattern": re.compile(
                    r'\b(red|blue|green|white|purple|orange|yellow|pink|cyan)\b', re.I
                ),
                "p2_pref_cat": "vehicle_settings.ambient_light",
            },
            # "on" is boolean — always explicit, no rule needed
        },
    },

    "set_reading_light": {
        # P4: need seat occupancy to determine correct position
        "p4_tools":     ["get_seats_occupancy"],
        "p4_all":       False,
        # Bypass P4 when user explicitly named the position (P1 for "position" satisfied)
        "p4_p1_bypass": True,
        "params": {
            "position": {
                # P1: auto from enum (ALL/DRIVER/PASSENGER/DRIVER_REAR/...)
                # No P2: position resolved from P4 (seat occupancy) or P1 (user named it)
                "p1_from_enum": True,
            },
            # "on" is boolean — always explicit, no rule needed
        },
    },

    # ── Charging ──────────────────────────────────────────────────────────────

    "calculate_charging_time_by_soc": {
        "p4_tools": None,
        "params": {
            "target_state_of_charge": {
                # P1: user stated explicit SOC target like "80%" or "80 percent"
                "p1_pattern": re.compile(r'\b\d{1,3}\s*(?:%|percent)\b', re.I),
                # P2: check saved charging SOC preference (e.g., "always charge to 80%")
                "p2_pref_cat": "points_of_interest.charging_stations",
            },
            # Other params (station_id, plug_id, start_soc) are always explicit — no rules
        },
    },

    # calculate_charging_soc_by_time: charging_time is explicitly user-provided (minutes)
    # → no P2/P5 rules needed; removed from config to avoid false guard triggers

    # ── Email ─────────────────────────────────────────────────────────────────

    "send_email": {
        "p4_tools": None,
        "params": {
            "email_addresses": {
                # re.compile(r'(?!)') never matches → guard ALWAYS fires.
                # Reason: CC/BCC rules (e.g., "always CC secretary") cannot be inferred
                # from the user message — must check get_user_preferences first.
                "p1_pattern":  re.compile(r'(?!)', re.I),
                "p2_pref_cat": "productivity_and_communication.email",
            },
            # "content_message" is free_string — user-authored content, no rule needed
        },
    },
}


# ---------------------------------------------------------------------------
# Schema Analysis Utilities  ← NEW in v8
#
# These functions provide a uniform, schema-driven classification of tool
# parameters.  They allow UniversalAmbiguityGuard to reason about ANY tool in the schema.
#
# The key insight: the JSON Schema type of a parameter tells us what kind of
# ambiguity it can carry and therefore which P-level should resolve it:
#
#   boolean     → always explicit from user (P1); no guard needed
#   enum        → multiple discrete choices → P1 literal match; else P2/P4/P5
#   number/int  → continuous range (e.g. 0–100%) → P5 candidate; P1 via unit
#   free_string → content is user-specified (e.g. message body) → no guard
#   other       → array, object, unknown → handled explicitly in config
# ---------------------------------------------------------------------------

# Parameter type labels — use these constants everywhere to avoid typos.
PARAM_BOOLEAN     = "boolean"       # {"type": "boolean"}
PARAM_ENUM        = "enum"          # {"type": "string", "enum": [...]}
PARAM_NUMBER      = "number"        # {"type": "number"} or {"type": "integer"}
PARAM_FREE_STRING = "free_string"   # {"type": "string"} without enum
PARAM_OTHER       = "other"         # array, object, or unknown


def _classify_param(param_schema: dict) -> str:
    """
    Classify a single parameter's JSON Schema into one of the PARAM_* labels.

    Disambiguation implication:
      PARAM_BOOLEAN     — user always says "on"/"off" → P1 by definition, no guard
      PARAM_ENUM        — finite choices → P1 if user names a value; else P2/P4/P5
      PARAM_NUMBER      — continuous range → P5 candidate; P1 via unit-anchored pattern
      PARAM_FREE_STRING — user-specified content (body, subject) → no auto-guard
      PARAM_OTHER       — arrays / objects → must be handled explicitly in config
    """
    ptype = param_schema.get("type", "")
    if ptype == "boolean":
        return PARAM_BOOLEAN
    if ptype == "string":
        return PARAM_ENUM if "enum" in param_schema else PARAM_FREE_STRING
    if ptype in ("number", "integer"):
        return PARAM_NUMBER
    return PARAM_OTHER


def _analyze_tool_params(tool_schema: dict) -> dict:
    """
    Analyze ALL parameters of a tool from its JSON Schema definition.

    Return structure:
      kind            : "no_arg" | "single_arg" | "multi_arg"
      count           : int — total parameter count
      required        : list[str] — required parameter names
      params          : dict[str, str] — {param_name: PARAM_* label}
      has_enum        : bool
      has_number      : bool
      has_boolean     : bool
      has_free_string : bool

    Examples:
      set_climate_temperature → {kind:"single_arg"… but actually has 2 params}
        params: {"temperature": "number", "seat_zone": "enum"}
        → has_number=True, has_enum=True, kind="multi_arg"

      open_close_sunshade → params: {"percentage": "number"}
        → has_number=True, kind="single_arg"

      get_climate_settings → params: {}
        → kind="no_arg", count=0, all has_* = False
    """
    fn = tool_schema.get("function", tool_schema)
    params_schema = fn.get("parameters", {})
    properties = params_schema.get("properties", {})
    required = list(params_schema.get("required", []))

    params: dict[str, str] = {
        pname: _classify_param(pschema)
        for pname, pschema in properties.items()
    }
    count = len(params)

    return {
        "kind":            "no_arg" if count == 0 else ("single_arg" if count == 1 else "multi_arg"),
        "count":           count,
        "required":        required,
        "params":          params,
        "has_enum":        any(t == PARAM_ENUM        for t in params.values()),
        "has_number":      any(t == PARAM_NUMBER      for t in params.values()),
        "has_boolean":     any(t == PARAM_BOOLEAN     for t in params.values()),
        "has_free_string": any(t == PARAM_FREE_STRING for t in params.values()),
    }


def _number_p1_pattern(param_name: str, param_schema: dict) -> "re.Pattern | None":
    """
    Auto-generate a P1 regex for a PARAM_NUMBER parameter.

    Strategy (in priority order):

    1. Unit-anchored (highest precision):
       If the description mentions a physical unit keyword (celsius, percent…),
       generate a pattern that REQUIRES that unit suffix.
       → "set to 22°C" matches; "25 km away" does NOT → no false positives.

    2. Range-constrained with negative lookahead (fallback):
       If min/max are defined and the span is ≤ 50, enumerate all integer values
       and add a negative lookahead that rejects common non-matching unit suffixes
       (km, m, min, sec, mph, kph, litre, day, hour, %, px).
       → Catches "set fan to level 3" style without units.

    3. Return None if neither strategy applies safely
       (e.g. open-ended range without unit hint → don't guess).
    """
    desc = param_schema.get("description", "").lower()
    minimum = param_schema.get("minimum")
    maximum = param_schema.get("maximum")

    # ── Strategy 1: unit-anchored ─────────────────────────────────────────────
    if any(kw in desc for kw in ("celsius", "degree")):
        # Temperature unit required — never matches "20 km" or "25 minutes"
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

    return None  # can't auto-derive safely — leave P1 unsatisfied (triggers P2/P4)


# ---------------------------------------------------------------------------
# ActionClaimGuard constants  — unchanged from v6
# ---------------------------------------------------------------------------

_COMPLETION_CLAIM_PATTERNS: list[re.Pattern] = [
    re.compile(r'\bI (?:set|turned on|turned off|activated|deactivated|opened|closed|enabled|disabled|changed|adjusted|increased|decreased)\b', re.I),
    re.compile(r'\bDone[,.].*?\b(?:on|off|set|open|closed|level|active|inactive)\b', re.I),
    re.compile(r'\bis now (?:on|off|set|at(?: level)?|open|closed|active)\b', re.I),
    re.compile(r'\bhas been (?:set|turned on|turned off|activated|deactivated)\b', re.I),
]


def _has_completion_claim(text: str) -> bool:
    return any(p.search(text) for p in _COMPLETION_CLAIM_PATTERNS)


# ---------------------------------------------------------------------------
# PolicyViolation — shared by PolicyChecker and UniversalAmbiguityGuard
# ---------------------------------------------------------------------------

class PolicyViolation:
    def __init__(self, tool_call_id: str, tool_name: str, message: str):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.message = message


# ---------------------------------------------------------------------------
# PolicyChecker — unchanged from v6 (AUT-POL hard rules, not ambiguity)
# ---------------------------------------------------------------------------

class PolicyChecker:
    """
    Enforces AUT-POL rules that are unconditional co-action or ordering
    constraints (not ambiguity resolution).  These are completely separate
    from the P1–P5 disambiguation chain.
    """

    def check(self, tool_calls: list[dict], state: StateCache) -> list[PolicyViolation]:
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
                    "AUT-POL:018 violation: set_new_navigation cannot be called when navigation is "
                    "already active. Use the appropriate editing tool instead: "
                    "navigation_replace_final_destination, navigation_replace_one_waypoint, "
                    "navigation_add_one_waypoint, navigation_delete_waypoint, "
                    "navigation_delete_destination."
                ))
            elif name in _NAV_EDITING_TOOLS and state.nav_active is False:
                violations.append(PolicyViolation(
                    tc["id"], name,
                    f"AUT-POL:017 violation: {name} can only be used when navigation is already "
                    "active. Currently navigation is inactive — use set_new_navigation to start "
                    "navigation first."
                ))
            elif name == "navigation_delete_destination":
                if len(state.nav_waypoints) <= 2:
                    violations.append(PolicyViolation(
                        tc["id"], name,
                        "AUT-POL:019 violation: Cannot delete the destination — the route has no "
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
                        "AUT-POL:014 violation: High beam headlights cannot be activated when fog "
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
                    "AUT-POL:013 violation: When activating fog lights the following co-actions "
                    f"are required but missing: {', '.join(missing)}."
                ))
        return violations

    def _check_climate(self, tool_calls, state):
        violations = []
        # AUT-POL:010 — window defrost co-actions (front/all only)
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
                    "AUT-POL:010 violation: When activating window defrost (front/all), the "
                    f"following co-actions are required but missing: {', '.join(missing)}."
                ))
        # AUT-POL:011 — AC on co-actions
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
                    "AUT-POL:011 violation: When activating AC, the following co-actions are "
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
                        f"AUT-POL:005 violation: Sunroof cannot be opened (percentage={percentage}%) "
                        f"because the sunshade is not fully open "
                        f"(current sunshade position: {state.sunshade_position}%). "
                        "Open the sunshade fully first, or open both in parallel."
                    ))
        return violations


# ---------------------------------------------------------------------------
# Batch-check helpers — unchanged from v6
# ---------------------------------------------------------------------------

def _batch_has(tool_calls, tool_name, key, value):
    for tc in tool_calls:
        if tc["function"]["name"] == tool_name and _arg(tc, key) == value:
            return True
    return False


def _batch_has_min(tool_calls, tool_name, key, minimum):
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
    for tc in tool_calls:
        if tc["function"]["name"] == tool_name:
            v = _arg(tc, key)
            if v is not None and substr.upper() in str(v).upper():
                return True
    return False


def _batch_has_window_close(tool_calls, window_id):
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


# ---------------------------------------------------------------------------
# UniversalAmbiguityGuard  ← NEW, replaces DisambiguationGuard +
#                                         StateCheckGuard +
#                                         P5ParamGuard
# ---------------------------------------------------------------------------

class UniversalAmbiguityGuard:
    """
    Single guard that enforces the P1→P2→P4→P5 disambiguation chain for all
    tools listed in PARAM_RESOLUTION_CONFIG.

    Lifecycle:
      1. Instantiate once in CARBenchAgentExecutor.__init__.
      2. Call init_from_schemas(tools) when tool schemas are first received.
         This auto-generates P1 patterns for tools marked p1_from_enum=True.
      3. Call check(...) on every LLM turn before dispatching tool calls.

    Resolution logic per tool call:
      P1  — if user's message already contains the parameter value explicitly,
             skip P2 (and P4 if p4_p1_bypass=True).
      P2  — if a preference category is configured and has NOT been fetched yet,
             emit a "check preferences first" violation.
      P4  — if required state tools have NOT been called yet, emit a
             "check state first" violation.  Only reached if P2 passes (or
             there is no P2 for this tool).
      P5  — if the tool has a continuous-range param (p5_param) AND the user
             did NOT specify a value → emit "must ask user" violation.
             Only reached after P2 and P4 pass.

    On any violation the executor injects a synthetic tool error and retries
    the LLM once (same pattern as PolicyChecker).
    """

    def __init__(self) -> None:
        # Nested: {tool_name: {param_name: compiled_P1_regex}}
        # Built from p1_from_enum and p1_from_schema entries in PARAM_RESOLUTION_CONFIG.
        # Replaces the old flat _enum_p1_patterns + _number_p1_patterns dicts.
        self._param_p1_patterns: dict[str, dict[str, re.Pattern]] = {}
        # Maps tool_name → full _analyze_tool_params result (for future use)
        self._tool_param_analysis: dict[str, dict] = {}
        self._schemas_initialized = False

    def init_from_schemas(self, tool_schemas: list[dict]) -> None:
        """
        Analyze all tool schemas and auto-generate per-param P1 patterns.

        For each tool in PARAM_RESOLUTION_CONFIG:
          1. Run _analyze_tool_params and cache in self._tool_param_analysis.
          2. For each param in cfg["params"]:
             - p1_from_enum=True  → collect enum values for THIS param only
               (not all params — fixes Bug 2: mixed-semantic patterns on multi-enum tools)
             - p1_from_schema=True → call _number_p1_pattern() for this param's schema
               (unit-anchored or range-constrained; no cross-param contamination)
          Store results in self._param_p1_patterns[tool_name][param_name].

        Called once when tool schemas arrive.  Subsequent calls are no-ops.
        """
        if self._schemas_initialized:
            return

        for tool in tool_schemas:
            fn = tool.get("function", {})
            name = fn.get("name", "")

            # Always analyze and cache the tool's parameter structure
            analysis = _analyze_tool_params(tool)
            self._tool_param_analysis[name] = analysis

            cfg = PARAM_RESOLUTION_CONFIG.get(name)
            if not cfg:
                continue

            props = fn.get("parameters", {}).get("properties", {})
            param_cfgs = cfg.get("params", {})

            for param_name, param_cfg in param_cfgs.items():
                pschema = props.get(param_name, {})

                if param_cfg.get("p1_from_enum"):
                    # Build regex from THIS param's enum values only
                    enum_values = [str(v) for v in pschema.get("enum", [])]
                    if enum_values:
                        terms = [re.escape(v.lower()) for v in enum_values]
                        self._param_p1_patterns.setdefault(name, {})[param_name] = re.compile(
                            r'\b(' + '|'.join(terms) + r')\b', re.I
                        )

                elif param_cfg.get("p1_from_schema"):
                    # Build number P1 regex from THIS param's schema (unit/range)
                    if analysis["params"].get(param_name) == PARAM_NUMBER:
                        pattern = _number_p1_pattern(param_name, pschema)
                        if pattern is not None:
                            self._param_p1_patterns.setdefault(name, {})[param_name] = pattern

        self._schemas_initialized = True

    def _p1_satisfied(
        self, tool_name: str, param_name: str, param_cfg: dict, user_msg: str
    ) -> bool:
        """
        Return True if the user message contains an explicit value for this parameter (P1).

        Priority:
          1. p1_pattern in param_cfg
          2. Schema-derived pattern in self._param_p1_patterns (from p1_from_enum or
             p1_from_schema, built by init_from_schemas)
        """
        pattern = param_cfg.get("p1_pattern")
        if pattern is None and (param_cfg.get("p1_from_enum") or param_cfg.get("p1_from_schema")):
            pattern = self._param_p1_patterns.get(tool_name, {}).get(param_name)
        if pattern is None:
            return False
        return bool(pattern.search(user_msg or ""))

    def check(
        self,
        tool_calls: list[dict],
        prefs_checked: set[str],
        tools_called: set[str],
        user_msg: str,
    ) -> list[PolicyViolation]:
        """
        Check each tool call against PARAM_RESOLUTION_CONFIG — per-parameter.

        For each tool call:
          1. P4 (tool-level): if required state tools not called → violation, skip params.
             p4_p1_bypass=True: skip P4 if ANY param's P1 is satisfied.
          2. Per-param P2: if param P1 not satisfied AND preference not yet fetched → violation.
             Multiple params sharing the same p2_pref_cat report only ONE P2 violation
             (fetching preferences once resolves all params in that category).
          3. Per-param P5: if param has no resolvable default AND user didn't specify →
             violation ("must ask user").

        Violations are reported highest-priority-first per tool.  Once a violation is
        found for a tool, lower-priority levels are skipped — fix the top issue first,
        then re-check next turn.
        """
        violations: list[PolicyViolation] = []

        for tc in tool_calls:
            name = tc["function"]["name"]
            cfg = PARAM_RESOLUTION_CONFIG.get(name)
            if cfg is None:
                continue  # tool not in config — no ambiguity rules apply

            param_cfgs: dict = cfg.get("params", {})

            # ── P4: tool-level state check ────────────────────────────────────
            p4_tools = cfg.get("p4_tools") or []
            if p4_tools:
                # p4_p1_bypass=True: skip P4 when user already gave us the value explicitly
                # (check ANY param's P1 — if any is satisfied, bypass is granted)
                p4_bypass = cfg.get("p4_p1_bypass", False) and any(
                    self._p1_satisfied(name, pn, pc, user_msg)
                    for pn, pc in param_cfgs.items()
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
                        continue  # don't check per-param rules if P4 not satisfied

            # ── Per-param P2 / P5 ─────────────────────────────────────────────
            # Track which P2 categories have already been reported this tool call,
            # so params sharing a category generate only ONE violation (not N violations
            # for N unresolved params — one prefs fetch resolves all of them).
            reported_p2_cats: set[str] = set()

            for param_name, param_cfg in param_cfgs.items():
                p1_ok = self._p1_satisfied(name, param_name, param_cfg, user_msg)

                # ── P2: preference check ──────────────────────────────────────
                p2_cat = param_cfg.get("p2_pref_cat")
                if p2_cat and not p1_ok and p2_cat not in prefs_checked:
                    if p2_cat not in reported_p2_cats:
                        # Pre-compute split parts outside f-string (Python 3.11 restriction)
                        p2_top = p2_cat.split(".")[0]
                        p2_leaf = p2_cat.split(".")[-1]
                        violations.append(PolicyViolation(
                            tc["id"], name,
                            f"DISAMBIGUATION_REQUIRED: Before calling {name}, parameter "
                            f"'{param_name}' is not resolved. You MUST first call "
                            f"get_user_preferences(preference_categories="
                            f"{{'{p2_top}': {{'{p2_leaf}': true}}}}) "
                            f"to check if the user has a saved preference. "
                            f"Only proceed with {name} after retrieving preferences. "
                            f"If no preference is found, ask the user to specify the value."
                        ))
                        reported_p2_cats.add(p2_cat)
                    continue  # don't also report P5 for this param — fix P2 first

                # ── P5: continuous-range param — must ask user ────────────────
                if param_cfg.get("p5_required") and not p1_ok:
                    p5_p1 = param_cfg.get("p5_p1_pattern")
                    if not (p5_p1 and p5_p1.search(user_msg or "")):
                        violations.append(PolicyViolation(
                            tc["id"], name,
                            f"PARAMETER_UNRESOLVED: '{param_name}' for {name} was not explicitly "
                            f"specified by the user and has no heuristic default "
                            f"(it is a 0–100 continuous range). "
                            f"You MUST ask the user to specify the {param_name} value "
                            f"before calling {name}. Do NOT guess or assume any value."
                        ))

        return violations


# ---------------------------------------------------------------------------
# CARBenchAgentExecutor
# ---------------------------------------------------------------------------

class CARBenchAgentExecutor(AgentExecutor):
    """Purple agent — AUT-POL enforcement + schema-driven P1–P5 ambiguity guard."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        interleaved_thinking: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.interleaved_thinking = interleaved_thinking

        # Per-context state
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}
        self.ctx_id_to_state: dict[str, StateCache] = {}
        self.ctx_id_to_prev_response_id: dict[str, str] = {}
        self.ctx_id_to_prefs_checked: dict[str, set[str]] = {}  # dotted pref paths retrieved
        self.ctx_id_to_tools_called: dict[str, set[str]] = {}   # tools called this user turn
        self.ctx_id_to_user_msg: dict[str, str] = {}             # latest user message text

        # Guards
        self._policy_checker = PolicyChecker()
        # Single universal guard replaces DisambiguationGuard + StateCheckGuard + P5ParamGuard
        self._universal_guard = UniversalAmbiguityGuard()

    # ── model helpers (unchanged) ─────────────────────────────────────────────

    def _is_claude_model(self) -> bool:
        m = (self.model or "").lower()
        return "claude" in m or m.startswith("anthropic/")

    def _is_openai_model(self) -> bool:
        m = (self.model or "").lower()
        return m.startswith("gpt-") or m.startswith("openai/")

    def _is_openai_responses_model(self) -> bool:
        return (self.model or "").lower().startswith("openai/responses/")

    def _is_lm_studio_model(self) -> bool:
        return (self.model or "").lower().startswith("lm_studio/")

    def _build_completion_kwargs(self) -> dict:
        kwargs: dict = {"model": self.model}
        if not self._is_openai_model():
            kwargs["temperature"] = self.temperature
        if self._is_lm_studio_model():
            kwargs["api_base"] = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
            kwargs["api_key"] = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        if self._is_openai_model() and not self._is_lm_studio_model():
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        if self.thinking:
            m = (self.model or "").lower()
            if self._is_claude_model():
                if "claude-opus-4-6" in m:
                    kwargs["thinking"] = {"type": "adaptive"}
                else:
                    if self.reasoning_effort in ("none", "disable", "low", "medium", "high"):
                        kwargs["reasoning_effort"] = self.reasoning_effort
                    else:
                        try:
                            kwargs["thinking"] = {
                                "type": "enabled",
                                "budget_tokens": int(self.reasoning_effort),
                            }
                        except ValueError:
                            raise ValueError(
                                "reasoning_effort must be 'none','disable','low','medium','high',"
                                " or an integer token budget"
                            )
                    if self.interleaved_thinking:
                        kwargs["extra_headers"] = {
                            "anthropic-beta": "interleaved-thinking-2025-05-14"
                        }
            elif self._is_openai_model():
                if self.reasoning_effort in ("none", "low", "medium", "high", "xhigh"):
                    kwargs["reasoning_effort"] = self.reasoning_effort
                else:
                    raise ValueError(
                        "For OpenAI, reasoning_effort must be 'none','low','medium','high','xhigh'"
                    )
        return kwargs

    # ── guard helpers ─────────────────────────────────────────────────────────

    def _inject_violations_and_retry(
        self,
        messages: list,
        tool_calls: list,
        assistant_content: dict,
        completion_kwargs: dict,
        violations: list[PolicyViolation],
        cancelled_msg: str,
        ctx_logger,
        guard_name: str,
    ) -> tuple[dict, list]:
        """
        Shared injection+retry logic used by both _apply_policy_guard and
        _apply_universal_guard.

        Appends the violating assistant message, injects synthetic tool errors
        for violated calls, injects NOT_EXECUTED for non-violated calls, then
        calls the LLM once and returns the retry result.
        """
        messages.append({
            "role": "assistant",
            "content": assistant_content.get("content"),
            "tool_calls": assistant_content["tool_calls"],
        })
        for v in violations:
            messages.append({
                "role": "tool",
                "tool_call_id": v.tool_call_id,
                "content": json.dumps({"status": "ERROR", "message": v.message}),
            })
        violated_ids = {v.tool_call_id for v in violations}
        for tc in tool_calls:
            if tc["id"] not in violated_ids:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps({"status": "NOT_EXECUTED", "message": cancelled_msg}),
                })

        retry_response = completion(messages=messages, **completion_kwargs)
        retry_content = retry_response.choices[0].message.model_dump(exclude_unset=True)
        retry_tool_calls = retry_content.get("tool_calls")
        ctx_logger.info(
            f"{guard_name} retry complete",
            retry_tool_calls=(
                [tc["function"]["name"] for tc in retry_tool_calls]
                if retry_tool_calls else []
            ),
        )
        return retry_content, retry_tool_calls

    def _apply_policy_guard(
        self,
        messages, tool_calls, assistant_content, completion_kwargs, state, ctx_logger,
    ) -> tuple[dict, list]:
        """Enforce AUT-POL hard rules (PolicyChecker).  One retry on violation."""
        violations = self._policy_checker.check(tool_calls, state)
        if not violations:
            return assistant_content, tool_calls

        ctx_logger.warning(
            "PolicyGuard: AUT-POL violation, retrying LLM",
            violated_tools=[v.tool_name for v in violations],
        )
        return self._inject_violations_and_retry(
            messages, tool_calls, assistant_content, completion_kwargs, violations,
            "Batch cancelled due to AUT-POL violation. Please resubmit after fixing.",
            ctx_logger, "PolicyGuard",
        )

    def _apply_universal_guard(
        self,
        messages, tool_calls, assistant_content, completion_kwargs,
        prefs_checked, tools_called, user_msg, ctx_logger,
    ) -> tuple[dict, list]:
        """
        Enforce P1–P5 disambiguation chain (UniversalAmbiguityGuard).

        Covers P2 (preference check), P4 (state check), and P5 (ask user)
        in a single pass.  Reports only the highest-priority violation per
        tool so the LLM resolves issues one level at a time.
        One retry on violation.
        """
        violations = self._universal_guard.check(
            tool_calls, prefs_checked, tools_called, user_msg
        )
        if not violations:
            return assistant_content, tool_calls

        ctx_logger.warning(
            "UniversalAmbiguityGuard: disambiguation required, retrying LLM",
            violated_tools=[v.tool_name for v in violations],
            violation_messages=[v.message[:60] for v in violations],
        )
        return self._inject_violations_and_retry(
            messages, tool_calls, assistant_content, completion_kwargs, violations,
            "Batch cancelled due to ambiguity guard violation. Resolve the flagged issue first.",
            ctx_logger, "UniversalAmbiguityGuard",
        )

    def _apply_action_claim_guard(
        self, messages, assistant_content, completion_kwargs, ctx_logger,
    ) -> dict:
        """
        Detect text responses that claim a state change without a tool call.
        Injects a corrective message and retries once.
        """
        content = assistant_content.get("content") or ""
        if not _has_completion_claim(content):
            return assistant_content

        ctx_logger.warning(
            "ActionClaimGuard: completion claim without tool call, retrying",
            claim_preview=content[:100],
        )
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content":
            "[SYSTEM CORRECTION] You claimed to have completed an action but made no tool call. "
            "State changes MUST be executed via tool calls — verbal claims do not change car state. "
            "Please retry and call the appropriate tool to actually perform the action."
        })
        retry_response = completion(messages=messages, **completion_kwargs)
        retry_content = retry_response.choices[0].message.model_dump(exclude_unset=True)
        ctx_logger.info(
            "ActionClaimGuard retry complete",
            has_tool_calls=bool(retry_content.get("tool_calls")),
        )
        return retry_content

    # ── main execute ──────────────────────────────────────────────────────────

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        inbound_message = context.message
        ctx_logger = logger.bind(role="agent", context=f"ctx:{context.context_id[:8]}")

        # ── per-context initialisation ────────────────────────────────────────
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []
        if context.context_id not in self.ctx_id_to_state:
            self.ctx_id_to_state[context.context_id] = StateCache()

        messages = self.ctx_id_to_messages[context.context_id]
        tools = self.ctx_id_to_tools.get(context.context_id, [])
        state = self.ctx_id_to_state[context.context_id]

        # ── parse incoming A2A message ─────────────────────────────────────────
        user_message_text = None
        incoming_tool_results = None

        try:
            for part in inbound_message.parts:
                if isinstance(part.root, TextPart):
                    text = part.root.text
                    if "System:" in text and "\n\nUser:" in text:
                        parts_split = text.split("\n\nUser:", 1)
                        system_prompt = parts_split[0].replace("System:", "").strip()
                        user_message_text = parts_split[1].strip()
                        if not messages:
                            messages.append({
                                "role": "system",
                                "content": EXTRA_INSTRUCTIONS + "\n\n" + system_prompt,
                            })
                    else:
                        user_message_text = text
                elif isinstance(part.root, DataPart):
                    data = part.root.data
                    if "tools" in data:
                        tools = data["tools"]
                        self.ctx_id_to_tools[context.context_id] = tools
                        # Initialise enum P1 patterns from schema (no-op after first call)
                        self._universal_guard.init_from_schemas(tools)
                    elif "tool_results" in data:
                        incoming_tool_results = data["tool_results"]

            if not user_message_text and not incoming_tool_results:
                user_message_text = context.get_user_input()

            ctx_logger.info(
                "Received message",
                turn=len(messages) + 1,
                preview=(
                    user_message_text[:100] if user_message_text else
                    f"[{len(incoming_tool_results)} tool results]" if incoming_tool_results else ""
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to parse message parts: {e}, using fallback")
            user_message_text = context.get_user_input()

        # Store latest user message for P1 detection in guards (updated every turn)
        if user_message_text:
            self.ctx_id_to_user_msg[context.context_id] = user_message_text

        # ── process tool results ──────────────────────────────────────────────
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
            prev_tool_calls = messages[-1]["tool_calls"]

            if incoming_tool_results:
                tool_call_by_name: dict[str, list] = {}
                for tc in prev_tool_calls:
                    tool_call_by_name.setdefault(tc["function"]["name"], []).append(tc)

                tool_results = []
                for tr in incoming_tool_results:
                    tr_name = tr.get("tool_name", "")
                    tr_content = tr.get("content", "")
                    matching_calls = tool_call_by_name.get(tr_name, [])
                    if matching_calls:
                        matched_tc = matching_calls.pop(0)
                        try:
                            matched_args = json.loads(matched_tc["function"]["arguments"])
                        except Exception:
                            matched_args = {}
                        # Update car state cache
                        state.update(tr_name, matched_args, tr_content)
                        # Track every tool called (for P4 state-check guard)
                        self.ctx_id_to_tools_called \
                            .setdefault(context.context_id, set()).add(tr_name)
                        # Track fetched preference categories (for P2 guard)
                        if tr_name == "get_user_preferences":
                            try:
                                for cat in _extract_pref_paths(
                                    matched_args.get("preference_categories", {})
                                ):
                                    self.ctx_id_to_prefs_checked \
                                        .setdefault(context.context_id, set()).add(cat)
                            except Exception:
                                pass
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": matched_tc["id"],
                            "content": tr_content,
                        })
                    else:
                        ctx_logger.warning("No matching tool_call for result", tool_name=tr_name)
                        self.ctx_id_to_tools_called \
                            .setdefault(context.context_id, set()).add(tr_name)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id", f"unknown_{tr_name}"),
                            "content": tr_content,
                        })
            else:
                tool_results = [
                    {"role": "tool", "tool_call_id": tc["id"], "content": user_message_text or ""}
                    for tc in prev_tool_calls
                ]

            messages.extend(tool_results)
        else:
            # New user message: append and reset tools_called for this turn
            messages.append({"role": "user", "content": user_message_text})
            self.ctx_id_to_tools_called[context.context_id] = set()

        # ── call LLM ─────────────────────────────────────────────────────────
        try:
            if self._is_claude_model():
                if tools:
                    tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                if messages:
                    messages[0]["cache_control"] = {"type": "ephemeral"}

            completion_kwargs = self._build_completion_kwargs()
            completion_kwargs["tools"] = tools if tools else None

            if self._is_openai_responses_model():
                prev_id = self.ctx_id_to_prev_response_id.get(context.context_id)
                if prev_id:
                    completion_kwargs["extra_body"] = {"previous_response_id": prev_id}

            response = completion(messages=messages, **completion_kwargs)

            if self._is_openai_responses_model():
                resp_id = getattr(response, "id", None) or getattr(response, "response_id", None)
                if resp_id and str(resp_id).startswith("resp"):
                    self.ctx_id_to_prev_response_id[context.context_id] = resp_id

            llm_message = response.choices[0].message
            assistant_content = llm_message.model_dump(exclude_unset=True)
            tool_calls = assistant_content.get("tool_calls")

            ctx_logger.info(
                "LLM response",
                has_tool_calls=bool(tool_calls),
                num_tool_calls=len(tool_calls) if tool_calls else 0,
                has_content=bool(assistant_content.get("content")),
            )

            # ── guard chain ───────────────────────────────────────────────────
            # 1. AUT-POL hard rules (PolicyChecker)
            if tool_calls:
                assistant_content, tool_calls = self._apply_policy_guard(
                    messages, tool_calls, assistant_content, completion_kwargs,
                    state, ctx_logger,
                )

            # 2. P1–P5 disambiguation chain (UniversalAmbiguityGuard)
            #    Replaces DisambiguationGuard + StateCheckGuard + P5ParamGuard
            if tool_calls:
                assistant_content, tool_calls = self._apply_universal_guard(
                    messages, tool_calls, assistant_content, completion_kwargs,
                    prefs_checked=self.ctx_id_to_prefs_checked.get(context.context_id, set()),
                    tools_called=self.ctx_id_to_tools_called.get(context.context_id, set()),
                    user_msg=self.ctx_id_to_user_msg.get(context.context_id, ""),
                    ctx_logger=ctx_logger,
                )

            # 3. ActionClaimGuard — text claims state change with no tool call
            if not tool_calls and assistant_content.get("content"):
                assistant_content = self._apply_action_claim_guard(
                    messages, assistant_content, completion_kwargs, ctx_logger,
                )
                tool_calls = assistant_content.get("tool_calls")

            # ── build response parts ──────────────────────────────────────────
            parts = []
            if assistant_content.get("content"):
                parts.append(Part(root=TextPart(kind="text", text=assistant_content["content"])))
            if assistant_content.get("tool_calls"):
                tool_calls_list = [
                    ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in assistant_content["tool_calls"]
                ]
                parts.append(Part(root=DataPart(
                    kind="data",
                    data=ToolCallsData(tool_calls=tool_calls_list).model_dump(),
                )))
            if assistant_content.get("reasoning_content"):
                parts.append(Part(root=DataPart(
                    kind="data",
                    data={"reasoning_content": assistant_content["reasoning_content"]},
                )))
            if not parts:
                parts.append(Part(root=TextPart(kind="text", text=assistant_content.get("content", ""))))

        except Exception as e:
            logger.error(f"LLM error: {e}")
            parts = [Part(root=TextPart(kind="text", text=f"Error processing request: {str(e)}"))]
            assistant_content = {"content": f"Error processing request: {str(e)}"}

        # ── persist assistant message to history ──────────────────────────────
        assistant_message_for_history: dict = {
            "role": "assistant",
            "content": assistant_content.get("content"),
        }
        if assistant_content.get("tool_calls"):
            assistant_message_for_history["tool_calls"] = assistant_content["tool_calls"]
        if assistant_content.get("thinking_blocks"):
            assistant_message_for_history["thinking_blocks"] = assistant_content["thinking_blocks"]
        if assistant_content.get("reasoning_content"):
            assistant_message_for_history["reasoning_content"] = assistant_content["reasoning_content"]
        messages.append(assistant_message_for_history)

        response_message = new_agent_parts_message(parts=parts, context_id=context.context_id)
        await event_queue.enqueue_event(response_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        ctx_logger = logger.bind(role="agent", context=f"ctx:{context.context_id[:8]}")
        ctx_logger.info("Canceling context")
        self.ctx_id_to_messages.pop(context.context_id, None)
        self.ctx_id_to_tools.pop(context.context_id, None)
        self.ctx_id_to_state.pop(context.context_id, None)
        self.ctx_id_to_prev_response_id.pop(context.context_id, None)
        self.ctx_id_to_prefs_checked.pop(context.context_id, None)
        self.ctx_id_to_tools_called.pop(context.context_id, None)
        self.ctx_id_to_user_msg.pop(context.context_id, None)

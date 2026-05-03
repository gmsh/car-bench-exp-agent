"""CARBenchAgentExecutor — orchestrator for the purple agent.

The executor is the public entrypoint imported as
``CARBenchAgentExecutor`` from ``car_bench_agent`` (a 3-line shim).
It coordinates one conversation turn end-to-end:

- per-context working memory: a :class:`~agent.state.StateCache`
  snapshot of vehicle observations, the set of preference categories
  already observed via ``get_user_preferences``, and the set of
  tools called this turn
- the guard chain that screens every LLM proposal before dispatch
  (MissingTool → ParamSchema → Policy → UniversalAmbiguity →
  ActionClaim)
- the model-specific completion-kwargs builder for the underlying LLM
- the a2a ``execute()`` / ``cancel()`` contract

Two private helpers fold in here because they are used only by the
executor: ``_extract_preference_paths`` (flattens the nested
preference_categories observation) and ``_has_completion_claim``
(ActionClaimGuard).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Part, TextPart, DataPart
from a2a.utils import new_agent_parts_message
from litellm import completion

# tool_call_types and logging_utils live next to the agent package, not
# inside it; add the parent of our package directory to sys.path so the
# bare imports resolve regardless of how the executor is launched.
_PKG_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PKG_DIR)
try:
    from logging_utils import configure_logger
    from tool_call_types import ToolCall, ToolCallsData
finally:
    try:
        sys.path.remove(_PKG_DIR)
    except ValueError:
        pass

from .prompts import EXTRA_INSTRUCTIONS
from .state import StateCache
from .policy import PolicyViolation, PolicyChecker
from .guards import (
    MissingToolGuard,
    ParamSchemaGuard,
    UniversalAmbiguityGuard,
)


logger = configure_logger(role="agent", context="-")


def _extract_preference_paths(d: dict, prefix: str = "") -> list[str]:
    """Flatten the nested preference_categories observation into dotted paths.

    The agent records observed preference categories as a flat set of
    dotted paths even though ``get_user_preferences`` exchanges them as
    a nested dict, so the cascade can answer "did we already observe
    this category?" with a single set membership test.
    """
    paths = []
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            paths.extend(_extract_preference_paths(v, full))
        else:
            paths.append(full)
    return paths


def _has_completion_claim(text: str) -> bool:
    """True if ``text`` reads like a verbal claim that a state change occurred without a tool call."""
    return any(p.search(text) for p in _COMPLETION_CLAIM_PATTERNS)


# ---------------------------------------------------------------------------
# CARBenchAgentExecutor
# ---------------------------------------------------------------------------

class CARBenchAgentExecutor(AgentExecutor):
    """Purple agent — vehicle-policy invariant enforcement + schema-driven P1–P5 ambiguity guard."""

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

        # Per-context working memory
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}
        self.ctx_id_to_state: dict[str, StateCache] = {}
        self.ctx_id_to_prev_response_id: dict[str, str] = {}
        # dotted preference category paths the agent has already observed via get_user_preferences
        self.ctx_id_to_preference_categories_observed: dict[str, set[str]] = {}
        # tools called within the current user turn (cleared on each new user message)
        self.ctx_id_to_tools_called: dict[str, set[str]] = {}
        # latest user message text (used by the cascade for P1 detection)
        self.ctx_id_to_user_msg: dict[str, str] = {}

        # Guards
        self._missing_tool_guard = MissingToolGuard()       # step 0: tool name in schema?
        self._param_schema_guard = ParamSchemaGuard()       # step 0.5: parameter names in schema?
        self._policy_checker = PolicyChecker()
        self._universal_guard = UniversalAmbiguityGuard()   # step 2: P1–P5 ambiguity cascade

    # ── model helpers ─────────────────────────────────────────────────────────

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
        Shared injection-and-retry logic used by every guard wrapper.

        Appends the violating assistant message, injects synthetic tool
        errors for violated calls, marks the rest as NOT_EXECUTED, then
        calls the LLM once for the retry.
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

    def _apply_missing_tool_guard(
        self,
        messages, tool_calls, assistant_content, completion_kwargs,
        tools, ctx_logger,
    ) -> tuple[dict, list]:
        """
        Step 0: verify all LLM-generated tool names exist in the live schema.
        One retry on violation. Double-failure fallback strips tool_calls entirely.
        """
        available = {t["function"]["name"] for t in tools} if tools else set()
        if not available:
            return assistant_content, tool_calls  # schema not yet received — skip

        violations = self._missing_tool_guard.check(tool_calls, available)
        if not violations:
            return assistant_content, tool_calls

        ctx_logger.warning(
            "MissingToolGuard: tool not in schema, retrying LLM",
            missing_tools=[v.tool_name for v in violations],
        )
        retry_content, retry_tool_calls = self._inject_violations_and_retry(
            messages, tool_calls, assistant_content, completion_kwargs, violations,
            "Batch cancelled: one or more tools are not available in this session.",
            ctx_logger, "MissingToolGuard",
        )

        # Double-failure fallback: retry still calls a missing tool → strip tool_calls
        if retry_tool_calls:
            still_missing = [
                tc["function"]["name"]
                for tc in retry_tool_calls
                if tc["function"]["name"] not in available
            ]
            if still_missing:
                ctx_logger.warning(
                    "MissingToolGuard: retry still calls missing tools, stripping",
                    still_missing=still_missing,
                )
                retry_content.pop("tool_calls", None)
                if not retry_content.get("content"):
                    retry_content["content"] = (
                        "I'm unable to complete this request because the required tool "
                        "is not currently available in this session."
                    )
                return retry_content, None

        return retry_content, retry_tool_calls

    def _apply_param_schema_guard(
        self,
        messages, tool_calls, assistant_content, completion_kwargs, ctx_logger,
    ) -> tuple[dict, list]:
        """
        Step 0.5: verify all generated parameter keys exist in the live tool schema.
        One retry on violation. Double-failure fallback strips tool_calls entirely.
        """
        violations = self._param_schema_guard.check(tool_calls)
        if not violations:
            return assistant_content, tool_calls

        ctx_logger.warning(
            "ParamSchemaGuard: parameter not in schema, retrying LLM",
            violated_tools=[v.tool_name for v in violations],
            invalid_params=[v.message.split("'")[1] for v in violations],
        )
        retry_content, retry_tool_calls = self._inject_violations_and_retry(
            messages, tool_calls, assistant_content, completion_kwargs, violations,
            "Batch cancelled: tool call includes parameter(s) not in current schema.",
            ctx_logger, "ParamSchemaGuard",
        )

        # Double-failure fallback: retry still contains invalid params → strip tool_calls
        if retry_tool_calls:
            still_invalid = []
            for tc in retry_tool_calls:
                name = tc["function"]["name"]
                valid = self._param_schema_guard._schema_props.get(name)
                if valid is None:
                    continue
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    continue
                still_invalid.extend(k for k in args if k not in valid)
            if still_invalid:
                ctx_logger.warning(
                    "ParamSchemaGuard: retry still has invalid params, stripping",
                    still_invalid=still_invalid,
                )
                retry_content.pop("tool_calls", None)
                if not retry_content.get("content"):
                    retry_content["content"] = (
                        "I'm unable to complete this request because the required "
                        "parameter is not available in this session's tool schema."
                    )
                return retry_content, None

        return retry_content, retry_tool_calls

    def _apply_policy_guard(
        self,
        messages, tool_calls, assistant_content, completion_kwargs, state, ctx_logger,
    ) -> tuple[dict, list]:
        """Enforce vehicle-policy invariants via PolicyChecker. One retry on violation."""
        violations = self._policy_checker.check(tool_calls, state)
        if not violations:
            return assistant_content, tool_calls

        ctx_logger.warning(
            "PolicyGuard: policy violation, retrying LLM",
            violated_tools=[v.tool_name for v in violations],
        )
        return self._inject_violations_and_retry(
            messages, tool_calls, assistant_content, completion_kwargs, violations,
            "Batch cancelled due to vehicle-policy violation. Please resubmit after fixing.",
            ctx_logger, "PolicyGuard",
        )

    def _apply_universal_guard(
        self,
        messages, tool_calls, assistant_content, completion_kwargs,
        preference_categories_observed, tools_called, user_msg, ctx_logger,
    ) -> tuple[dict, list]:
        """
        Enforce the P1–P5 disambiguation cascade (UniversalAmbiguityGuard).

        Covers P2 (preference observation), P4 (state observation), and P5
        (ask user) in a single pass. Reports only the highest-priority
        violation per parameter so the LLM repairs issues one level at a time.
        One retry on violation.
        """
        violations = self._universal_guard.check(
            tool_calls, preference_categories_observed, tools_called, user_msg
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
        Injects a corrective observation and retries once.
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
                        # Initialise per-parameter P1 patterns from schema (no-op after first call)
                        self._universal_guard.init_from_schemas(tools)
                        # Initialise the parameter-schema index (no-op after first call)
                        self._param_schema_guard.init_from_schemas(tools)
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

        # Store latest user message for P1 detection in the cascade (updated every turn)
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
                        # Fold this observation into per-context working memory
                        state.update(tr_name, matched_args, tr_content)
                        # Track every tool called this turn (for the P4 state-check cascade)
                        self.ctx_id_to_tools_called \
                            .setdefault(context.context_id, set()).add(tr_name)
                        # Track preference categories observed this turn (for the P2 cascade)
                        if tr_name == "get_user_preferences":
                            try:
                                for category in _extract_preference_paths(
                                    matched_args.get("preference_categories", {})
                                ):
                                    self.ctx_id_to_preference_categories_observed \
                                        .setdefault(context.context_id, set()).add(category)
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
            # 0. MissingToolGuard — tool name must exist in live schema
            if tool_calls:
                assistant_content, tool_calls = self._apply_missing_tool_guard(
                    messages, tool_calls, assistant_content, completion_kwargs,
                    tools=tools, ctx_logger=ctx_logger,
                )

            # 0.5 ParamSchemaGuard — parameter keys must exist in live schema
            if tool_calls:
                assistant_content, tool_calls = self._apply_param_schema_guard(
                    messages, tool_calls, assistant_content, completion_kwargs,
                    ctx_logger=ctx_logger,
                )

            # 1. Vehicle-policy invariants (PolicyChecker)
            if tool_calls:
                assistant_content, tool_calls = self._apply_policy_guard(
                    messages, tool_calls, assistant_content, completion_kwargs,
                    state, ctx_logger,
                )

            # 2. P1–P5 disambiguation cascade (UniversalAmbiguityGuard)
            if tool_calls:
                assistant_content, tool_calls = self._apply_universal_guard(
                    messages, tool_calls, assistant_content, completion_kwargs,
                    preference_categories_observed=self.ctx_id_to_preference_categories_observed.get(
                        context.context_id, set()
                    ),
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
        self.ctx_id_to_preference_categories_observed.pop(context.context_id, None)
        self.ctx_id_to_tools_called.pop(context.context_id, None)
        self.ctx_id_to_user_msg.pop(context.context_id, None)


# ──────────────────────────────────────────────────────────────

# ActionClaimGuard regex patterns — match LLM "I just did X" assertions that
# claim a state change without a corresponding tool call.
_COMPLETION_CLAIM_PATTERNS: list[re.Pattern] = [
    re.compile(r'\bI (?:set|turned on|turned off|activated|deactivated|opened|closed|enabled|disabled|changed|adjusted|increased|decreased)\b', re.I),
    re.compile(r'\bDone[,.].*?\b(?:on|off|set|open|closed|level|active|inactive)\b', re.I),
    re.compile(r'\bis now (?:on|off|set|at(?: level)?|open|closed|active)\b', re.I),
    re.compile(r'\bhas been (?:set|turned on|turned off|activated|deactivated)\b', re.I),
]

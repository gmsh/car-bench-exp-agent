# CAReful: A Reliable In-Car Assistant Agent

> **Competition submission** for the [AgentBeats Phase 2 Sprint 2 — Computer Use & Web Agent Track](https://rdi.berkeley.edu/agentx-agentbeats)  
> Benchmark Leaderboard: [CAR-bench](https://agentbeats.dev/agentbeater/car-bench) · Our agent page: [CAReful: A Reliable In-Car Assistant Agent](https://agentbeats.dev/gmsh/careful-a-reliable-in-car-assistant-agent)

 Drivers speak ambiguously, cannot be interrupted often, and expect correct actions — making hallucination avoidance, careful disambiguation, and strict policy compliance essential. To address this, we ground the LLM in live vehicle state to prevent false assumptions, enforce a priority-ordered disambiguation policy that minimizes driver interruption, and add a deterministic feedback step that catches policy violations before they reach the car.

---

## Key Contributions

Building a reliable in-car voice assistant is harder than it looks. Users speak in ambiguous, conversational language while driving — they cannot be interrupted often, and a wrong action (activating the wrong lights, misreading navigation state) can have real consequences. General-purpose LLM agents struggle here: they hallucinate vehicle state they have not checked, guess at ambiguous parameters rather than resolving them properly, and ignore safety constraints when those constraints conflict with their best-guess interpretation of the request. We address these three failure modes directly.

### 1. Grounding the LLM in User Context and Vehicle State

LLMs hallucinate in three distinct ways in this domain: they invent vehicle state they have not checked, they call tools that do not exist in the current session, and they fabricate parameter names that are not in the live schema. We address all three. First, we maintain a continuously updated record of conversation history and vehicle state across every turn — car state is extracted from tool results as they arrive, so every subsequent LLM call is grounded in what the car actually reported. Second, we add schema-grounded guards that verify every tool name and every parameter key against the live tool schemas before any call is dispatched — preventing the LLM from invoking tools that do not exist or passing parameters it imagined rather than ones the tool actually accepts.

### 2. Minimizing Driver Interruption Through Structured Disambiguation

Asking the driver to clarify is costly and potentially unsafe. At the same time, silently guessing is worse. We enforce a **structured disambiguation priority policy** that maximizes autonomous resolution before ever surfacing a question to the driver:

- **Highest priority** — hard safety and policy rules are checked first and are non-negotiable
- **Explicit user request** — if the user's utterance makes the intent unambiguous, it is honored directly
- **Learned preferences** — personal user preferences (retrieved via the preferences tool) are consulted before falling back on defaults
- **Contextual vehicle state** — current car state (window positions, active navigation, lighting, etc.) is used to resolve remaining ambiguity
- **User clarification (last resort)** — only when the above sources leave genuine ambiguity does the agent ask the driver

This priority ordering is enforced programmatically at the parameter level, not left to the LLM's discretion, ensuring consistent behavior across models.

### 3. Immediate Policy Feedback to Reduce Constraint Violations

Policy rules in a prompt can be forgotten or overridden by the LLM's own reasoning. We add a **deterministic feedback step** after each LLM response: a rule checker inspects the proposed tool calls against current vehicle state and flags any policy violation before it reaches the environment. When a violation is detected, a structured error is injected into the conversation and the LLM is given a chance to correct itself immediately. This closes the feedback loop at the earliest possible point — within the same turn — rather than relying on the LLM to self-enforce rules it may have deprioritized.

---

## Role in the Evaluation System

This repo is the **purple agent** — the agent *under evaluation*. It does not run the benchmark itself.

The benchmark is run by the **green agent** (harness), which plays both the simulated user and the simulated car:

```
┌─────────────────────────────────────────────────────────┐
│  Green Agent (harness, separate repo)                   │
│                                                         │
│  ┌──────────────┐    ┌─────────────────────────────┐    │
│  │  Simulated   │    │  CAR-bench Environment      │    │
│  │  User (LLM)  │    │  (real tool simulator)      │    │
│  └──────┬───────┘    └──────────────┬──────────────┘    │
│         │  user turns               │ tool results      │
└─────────┼───────────────────────────┼──────────────────-┘
          │ A2A                       │ A2A
          ▼                           │
┌─────────────────────────────────────────────────────────┐
│  Purple Agent  ◄── THIS REPO                            │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  CARBenchAgentExecutor                           │   │
│  │    1. Build message history + system prompt      │   │
│  │    2. Call LLM (LiteLLM, any model)              │   │
│  │    3. Guard chain:                               │   │
│  │         a. MissingToolGuard (tool hallucination) │   │
│  │         b. ParamSchemaGuard (param hallucination)│   │
│  │         c. PolicyChecker    (AUT-POL hard rules) │   │
│  │         d. UniversalAmbiguityGuard (P1→P5)       │   │
│  │         e. ActionClaimGuard (no fake claims)     │   │
│  │    4. Return tool calls + text ──► green agent   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Flow per turn:**
1. Green agent sends a user message to the purple agent (A2A TextPart) along with available tool schemas and the system/wiki prompt (A2A DataPart).
2. Purple agent calls the LLM, runs the guard chain, and returns tool calls back to the green agent (A2A DataPart).
3. Green agent executes those tool calls against the simulated car environment and sends results back (A2A DataPart with `tool_results`).
4. Steps 2–3 repeat until the purple agent produces a final text response.
5. Green agent scores the conversation against the ground truth.

---

## Project Structure

```
src/
  car_bench_agent.py   # All agent logic (guards, state cache, executor)
  server.py            # HTTP server entry point (Starlette + uvicorn)
  tool_call_types.py   # Pydantic models for A2A DataPart tool calls
  logging_utils.py     # Structured logger helper
amber/
  amber-manifest-purple.json5  # Amber deployment manifest
```

---

## Running the Server

```bash
# Basic startup
python src/server.py --host 127.0.0.1 --port 8080 --agent-llm claude-sonnet-4-6

# With thinking enabled (Claude extended thinking)
python src/server.py --agent-llm claude-opus-4-6 --thinking --reasoning-effort medium

# With OpenAI model
python src/server.py --agent-llm gpt-4o --temperature 0.0

# Environment variables (alternative to CLI flags)
AGENT_LLM=claude-sonnet-4-6 AGENT_TEMPERATURE=0.0 python src/server.py
```

**Supported model families** (via LiteLLM):
- `claude-*` / `anthropic/*` — Claude models with optional thinking / interleaved thinking
- `gpt-*` / `openai/*` — OpenAI models with optional `reasoning_effort`
- `openai/responses/*` — OpenAI Responses API (stateful, uses `previous_response_id`)
- `lm_studio/*` — Local LM Studio server

---

## Resources

- [AgentBeats Competition — Phase 2 Sprint 2](https://rdi.berkeley.edu/agentx-agentbeats) — the competition this agent is submitted to
- [CAR-bench Leaderboard](https://agentbeats.dev/agentbeater/car-bench) — benchmark leaderboard on AgentBeats
- [CAR-bench Paper (arXiv:2601.22027)](https://arxiv.org/abs/2601.22027) — Kirmayr et al., 2026
- [Our Agent's Page](https://agentbeats.dev/gmsh/careful-a-reliable-in-car-assistant-agent) — submission page on AgentBeats
- [Our Code Repository](https://github.com/gmsh/car-bench-exp-agent)

```bibtex
@misc{kirmayr2026carbenchevaluatingconsistencylimitawareness,
      title={CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty}, 
      author={Johannes Kirmayr and Lukas Stappen and Elisabeth André},
      year={2026},
      eprint={2601.22027},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.22027}, 
}
```

---

## Acknowledgements

Parts of this codebase build on the [official CAR-bench agent starter code](https://github.com/CAR-bench/car-bench-agentbeats) released by the original paper authors, used under the [MIT License](https://github.com/CAR-bench/car-bench-agentbeats/blob/main/LICENSE). We thank the authors for their work on the benchmark and for making the reference implementation available.

This project was developed with the assistance of [Claude](https://claude.ai) (Anthropic) and [Codex](https://openai.com/codex) (OpenAI).

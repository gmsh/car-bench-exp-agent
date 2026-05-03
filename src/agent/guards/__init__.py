"""Guard chain for the purple agent.

Each guard inspects an LLM-proposed batch of tool calls before dispatch
and returns a list of :class:`~agent.policy.PolicyViolation` instances.
The executor injects synthetic tool errors for any violation and gives
the LLM one retry opportunity to repair the batch.
"""
from .missing_tool import MissingToolGuard
from .param_schema import ParamSchemaGuard
from .universal_ambiguity import UniversalAmbiguityGuard

__all__ = ["MissingToolGuard", "ParamSchemaGuard", "UniversalAmbiguityGuard"]

"""Pure routing decisions for the evidence-driven diagnosis graph.

The functions in this module do not mutate state.  Keeping routing separate from
LangGraph assembly makes termination behaviour straightforward to unit test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sized
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from app.config import Settings, config

EXECUTE: Final = "execute"
REPLAN: Final = "replan"
RANK: Final = "rank"

EvaluationRoute: TypeAlias = Literal["execute", "replan", "rank"]
PostReplanRoute: TypeAlias = Literal["execute", "rank"]


@dataclass(frozen=True, slots=True)
class RoutingLimits:
    """Hard limits used by the routing functions."""

    max_steps: int
    max_replans: int
    max_tool_calls: int

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if self.max_replans < 0:
            raise ValueError("max_replans must be non-negative")
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be greater than zero")

    @classmethod
    def from_settings(cls, settings: Settings) -> RoutingLimits:
        """Build immutable routing limits from application settings."""

        return cls(
            max_steps=settings.aiops_max_steps,
            max_replans=settings.aiops_max_replans,
            max_tool_calls=settings.aiops_max_tool_calls,
        )


DEFAULT_ROUTING_LIMITS = RoutingLimits.from_settings(config)


def _get_field(value: object, name: str, default: Any = None) -> Any:
    """Read a field from a JSON-compatible mapping or a model-like object."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_non_negative_int(value: object, default: int = 0) -> int:
    """Return a defensive non-negative counter value."""

    if not isinstance(value, (bool, int, float, str, bytes, bytearray)):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _item_count(value: object) -> int:
    """Count state collections without consuming iterators."""

    if isinstance(value, Sized) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _counter(
    state: Mapping[str, Any],
    name: str,
    *,
    fallback_collection: str | None = None,
) -> int:
    value = state.get(name)
    if value is not None:
        return _as_non_negative_int(value)
    if fallback_collection is not None:
        return _item_count(state.get(fallback_collection))
    return 0


def _has_items(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, Sized):
        return len(value) > 0
    return bool(value)


def _execution_budget_exhausted(
    state: Mapping[str, Any],
    evaluation: object,
    limits: RoutingLimits,
) -> bool:
    """Check budgets that prohibit any further tool-execution step."""

    if bool(_get_field(evaluation, "budget_exhausted", False)):
        return True

    remaining_budget = state.get("remaining_budget")
    if remaining_budget is not None and _as_non_negative_int(remaining_budget) == 0:
        return True

    step_count = _counter(state, "step_count", fallback_collection="execution_history")
    tool_call_count = _counter(
        state,
        "tool_call_count",
        fallback_collection="raw_tool_results",
    )
    return step_count >= limits.max_steps or tool_call_count >= limits.max_tool_calls


def _needs_replan(state: Mapping[str, Any], evaluation: object) -> bool:
    """Recognize evaluator signals that require a different investigation plan."""

    explicit_decision = _get_field(evaluation, "need_replan", None)
    if explicit_decision is not None:
        return bool(explicit_decision)

    for field_name in ("missing_evidence", "evidence_conflicts", "tool_failures"):
        if _has_items(_get_field(evaluation, field_name)):
            return True

    # Compatibility for an execution failure routed before an evaluator exists.
    return evaluation is None and _has_items(state.get("tool_errors"))


def route_after_evaluation(
    state: Mapping[str, Any],
    limits: RoutingLimits = DEFAULT_ROUTING_LIMITS,
) -> EvaluationRoute:
    """Route an evaluated diagnosis to execution, replanning, or ranking.

    Precedence is intentional: exhausted budgets and sufficient evidence always
    terminate investigation; missing/conflicting/unavailable evidence replans
    only while the replan budget remains; an otherwise valid remaining plan is
    executed.  With no executable or replannable work, ranking produces a
    partial diagnosis.
    """

    evaluation = state.get("evaluation")

    if _execution_budget_exhausted(state, evaluation, limits):
        return RANK

    if bool(_get_field(evaluation, "can_finish", False)):
        return RANK

    replan_count = _counter(state, "replan_count")
    can_replan = replan_count < limits.max_replans

    if _needs_replan(state, evaluation):
        return REPLAN if can_replan else RANK

    if _has_items(state.get("plan")):
        return EXECUTE

    # A completed plan with an explicit inconclusive evaluation may investigate
    # again, but only up to max_replans.  Missing evaluation cannot justify more
    # calls and therefore falls through to a partial ranking.
    if evaluation is not None and can_replan:
        return REPLAN

    return RANK


def route_after_replan(
    state: Mapping[str, Any],
    limits: RoutingLimits = DEFAULT_ROUTING_LIMITS,
) -> PostReplanRoute:
    """Execute a newly produced plan or terminate with a partial ranking.

    Reaching ``max_replans`` does not discard the final legal replan: that plan
    may still execute within step/tool budgets.  The next evaluation cannot
    request another replan because :func:`route_after_evaluation` uses a strict
    ``replan_count < max_replans`` check.
    """

    evaluation = state.get("evaluation")
    if _execution_budget_exhausted(state, evaluation, limits):
        return RANK

    if _counter(state, "replan_count") > limits.max_replans:
        return RANK

    return EXECUTE if _has_items(state.get("plan")) else RANK


__all__ = [
    "DEFAULT_ROUTING_LIMITS",
    "EXECUTE",
    "EvaluationRoute",
    "PostReplanRoute",
    "RANK",
    "REPLAN",
    "RoutingLimits",
    "route_after_evaluation",
    "route_after_replan",
]

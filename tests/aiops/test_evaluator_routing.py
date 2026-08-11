"""Evidence sufficiency and bounded routing tests."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.aiops.evidence import evaluate_evidence
from app.agent.aiops.routing import (
    EXECUTE,
    RANK,
    REPLAN,
    RoutingLimits,
    route_after_evaluation,
    route_after_replan,
)


def test_evaluator_finishes_with_two_reliable_online_sources(
    hypothesis_factory: Any,
    evidence_factory: Any,
) -> None:
    hypothesis = hypothesis_factory()
    evidence = [
        evidence_factory("E-monitor", source="monitor", supports=["H1"]),
        evidence_factory("E-logs", source="logs", supports=["H1"]),
    ]

    evaluation = evaluate_evidence(
        [hypothesis],
        evidence,
        remaining_budget=3,
    )

    assert evaluation.supported_hypotheses == ["H1"]
    assert evaluation.diagnosis_confidence >= 0.70
    assert evaluation.can_finish is True
    assert evaluation.need_replan is False
    assert evaluation.missing_evidence == []


def test_evaluator_requests_replan_when_key_evidence_is_missing(
    hypothesis_factory: Any,
) -> None:
    evaluation = evaluate_evidence(
        [hypothesis_factory()],
        [],
        remaining_budget=3,
    )

    assert evaluation.can_finish is False
    assert evaluation.need_replan is True
    assert evaluation.uncertain_hypotheses == ["H1"]
    assert len(evaluation.missing_evidence) == 2


def test_evaluator_keeps_executing_an_existing_plan_before_replanning(
    hypothesis_factory: Any,
) -> None:
    evaluation = evaluate_evidence(
        [hypothesis_factory()],
        [],
        remaining_budget=3,
        remaining_plan=[
            {
                "id": "S-next",
                "hypothesis_id": "H1",
                "goal": "Collect an online CPU metric",
                "rationale": "The initial bounded plan still has useful work.",
                "tool_hint": ["query_cpu_metrics"],
                "expected_evidence": ["online CPU metric"],
            }
        ],
    )

    assert evaluation.can_finish is False
    assert evaluation.need_replan is False
    assert evaluation.missing_evidence
    assert (
        route_after_evaluation(
            {
                "remaining_budget": 3,
                "replan_count": 0,
                "plan": [{"id": "S-next"}],
                "evaluation": evaluation.model_dump(mode="json"),
            },
            LIMITS,
        )
        == EXECUTE
    )


def test_neutral_online_observations_do_not_consume_expected_evidence_slots(
    hypothesis_factory: Any,
    evidence_factory: Any,
) -> None:
    neutral = [
        evidence_factory("E-neutral-1", source="generic"),
        evidence_factory("E-neutral-2", source="generic"),
    ]

    evaluation = evaluate_evidence(
        [hypothesis_factory()],
        neutral,
        remaining_budget=3,
    )

    assert evaluation.need_replan is True
    assert evaluation.missing_evidence == [
        "H1: CPU above threshold",
        "H1: matching runtime errors",
    ]


def test_evaluator_detects_conflicting_online_evidence(
    hypothesis_factory: Any,
    evidence_factory: Any,
) -> None:
    evidence = [
        evidence_factory("E-support", source="monitor", supports=["H1"]),
        evidence_factory("E-refute", source="logs", contradicts=["H1"]),
    ]

    evaluation = evaluate_evidence(
        [hypothesis_factory()],
        evidence,
        remaining_budget=3,
    )

    assert evaluation.can_finish is False
    assert evaluation.need_replan is True
    assert evaluation.uncertain_hypotheses == ["H1"]
    assert len(evaluation.evidence_conflicts) == 1
    assert "E-support" in evaluation.evidence_conflicts[0]
    assert "E-refute" in evaluation.evidence_conflicts[0]


def test_knowledge_only_can_never_complete_diagnosis(
    hypothesis_factory: Any,
    evidence_factory: Any,
) -> None:
    knowledge = evidence_factory(
        "KE1",
        source="knowledge_base",
        supports=["H1"],
        observation="Runbook says CPU saturation is worth investigating",
    )

    evaluation = evaluate_evidence(
        [hypothesis_factory(expected_evidence=["online CPU metric"])],
        [knowledge],
        remaining_budget=3,
    )

    assert evaluation.supported_hypotheses == []
    assert evaluation.can_finish is False
    assert evaluation.need_replan is True
    assert evaluation.diagnosis_confidence == 0.0
    assert evaluation.missing_evidence == ["H1: online CPU metric"]


LIMITS = RoutingLimits(max_steps=4, max_replans=2, max_tool_calls=5)


@pytest.mark.parametrize(
    ("state", "expected_route"),
    [
        (
            {
                "remaining_budget": 4,
                "replan_count": 0,
                "evaluation": {
                    "need_replan": True,
                    "can_finish": False,
                    "missing_evidence": ["H1: CPU metric"],
                },
            },
            REPLAN,
        ),
        (
            {
                "remaining_budget": 4,
                "replan_count": 0,
                "evaluation": {"need_replan": False, "can_finish": True},
            },
            RANK,
        ),
        (
            {
                "remaining_budget": 4,
                "replan_count": 0,
                "evaluation": {
                    "need_replan": True,
                    "can_finish": False,
                    "tool_failures": ["TE1: monitor timeout"],
                },
            },
            REPLAN,
        ),
    ],
    ids=["missing-to-replan", "sufficient-to-rank", "failure-to-replan"],
)
def test_evaluation_routes(state: dict[str, Any], expected_route: str) -> None:
    assert route_after_evaluation(state, LIMITS) == expected_route


@pytest.mark.parametrize(
    "state",
    [
        {
            "remaining_budget": 0,
            "evaluation": {"need_replan": True, "can_finish": False},
        },
        {
            "remaining_budget": 4,
            "step_count": 4,
            "evaluation": {"need_replan": True, "can_finish": False},
        },
        {
            "remaining_budget": 4,
            "tool_call_count": 5,
            "evaluation": {"need_replan": True, "can_finish": False},
        },
        {
            "remaining_budget": 4,
            "replan_count": 2,
            "evaluation": {"need_replan": True, "can_finish": False},
        },
    ],
    ids=["remaining-budget", "step-budget", "tool-budget", "replan-budget"],
)
def test_all_budget_limits_force_partial_ranking(state: dict[str, Any]) -> None:
    assert route_after_evaluation(state, LIMITS) == RANK


def test_pure_state_machine_always_reaches_rank_within_bounds(step_factory: Any) -> None:
    limits = RoutingLimits(max_steps=3, max_replans=2, max_tool_calls=3)
    state: dict[str, Any] = {
        "remaining_budget": 10,
        "step_count": 0,
        "tool_call_count": 0,
        "replan_count": 0,
        "plan": [],
        "evaluation": {
            "need_replan": True,
            "can_finish": False,
            "missing_evidence": ["H1: runtime signal"],
        },
    }
    transitions: list[str] = []

    for _ in range(20):
        route = route_after_evaluation(state, limits)
        transitions.append(route)
        if route == RANK:
            break

        if route == REPLAN:
            state["replan_count"] += 1
            state["plan"] = [step_factory(expected_evidence=["runtime signal"])]
            post_replan = route_after_replan(state, limits)
            transitions.append(post_replan)
            if post_replan == RANK:
                break
            assert post_replan == EXECUTE
            route = EXECUTE

        if route == EXECUTE:
            state["step_count"] += 1
            state["tool_call_count"] += 1
            state["remaining_budget"] -= 1
            state["plan"] = []
            # The environment remains inconclusive forever; hard limits must stop it.
            state["evaluation"] = {
                "need_replan": True,
                "can_finish": False,
                "missing_evidence": ["H1: runtime signal"],
            }
    else:  # pragma: no cover - assertion message is clearer than silent loop exhaustion
        pytest.fail("bounded diagnosis state machine did not terminate")

    assert transitions[-1] == RANK
    assert state["replan_count"] <= limits.max_replans
    assert state["step_count"] <= limits.max_steps
    assert state["tool_call_count"] <= limits.max_tool_calls
    assert len(transitions) <= 2 * limits.max_replans + 1

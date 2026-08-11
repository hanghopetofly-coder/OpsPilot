"""Structured model validation tests for the diagnosis domain."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.agent.aiops.models import (
    DiagnosisEvidence,
    DiagnosisHypothesis,
    DiagnosisPlan,
    DiagnosisStep,
)


def test_diagnosis_plan_is_strict_json_structured_output(
    hypothesis_factory: Any,
    step_factory: Any,
) -> None:
    plan = DiagnosisPlan.model_validate(
        {
            "hypotheses": [hypothesis_factory()],
            "steps": [step_factory()],
        }
    )

    payload = plan.model_dump(mode="json")

    assert payload["hypotheses"][0]["id"] == "H1"
    assert payload["steps"][0]["hypothesis_id"] == "H1"
    assert json.loads(json.dumps(payload))["steps"][0]["id"] == "S1"


@pytest.mark.parametrize(
    "hypotheses,steps,error_pattern",
    [
        (
            [{"id": "H1", "cause": "CPU", "expected_evidence": ["metric"]}],
            [
                {
                    "id": "S1",
                    "hypothesis_id": "H-missing",
                    "goal": "query",
                    "rationale": "verify",
                    "expected_evidence": ["metric"],
                }
            ],
            "unknown hypothesis",
        ),
        (
            [
                {"id": "H1", "cause": "CPU", "expected_evidence": ["metric"]},
                {"id": "H1", "cause": "Memory", "expected_evidence": ["metric"]},
            ],
            [
                {
                    "id": "S1",
                    "hypothesis_id": "H1",
                    "goal": "query",
                    "rationale": "verify",
                    "expected_evidence": ["metric"],
                }
            ],
            "hypothesis ids must not contain duplicates",
        ),
        (
            [{"id": "H1", "cause": "CPU", "expected_evidence": []}],
            [
                {
                    "id": "S1",
                    "hypothesis_id": "H1",
                    "goal": "query",
                    "rationale": "verify",
                    "expected_evidence": ["metric"],
                }
            ],
            "at least 1 item",
        ),
    ],
)
def test_diagnosis_plan_rejects_invalid_structure(
    hypotheses: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    error_pattern: str,
) -> None:
    with pytest.raises(ValidationError, match=error_pattern):
        DiagnosisPlan.model_validate({"hypotheses": hypotheses, "steps": steps})


def test_hypothesis_rejects_overlapping_evidence_ids() -> None:
    with pytest.raises(ValidationError, match="both support and contradict"):
        DiagnosisHypothesis(
            id="H1",
            cause="CPU saturation",
            expected_evidence=["CPU metric"],
            supporting_evidence_ids=["E1"],
            contradicting_evidence_ids=["E1"],
        )


def test_model_default_lists_are_isolated_between_instances() -> None:
    first = DiagnosisHypothesis(
        id="H1",
        cause="CPU saturation",
        expected_evidence=["CPU metric"],
    )
    second = DiagnosisHypothesis(
        id="H2",
        cause="Memory pressure",
        expected_evidence=["Memory metric"],
    )
    first_step = DiagnosisStep(
        id="S1",
        hypothesis_id="H1",
        goal="Inspect CPU",
        rationale="Validate H1",
        expected_evidence=["CPU metric"],
    )
    second_step = DiagnosisStep(
        id="S2",
        hypothesis_id="H2",
        goal="Inspect memory",
        rationale="Validate H2",
        expected_evidence=["Memory metric"],
    )

    first.supporting_evidence_ids.append("E1")
    first_step.tool_hint.append("query_cpu_metrics")

    assert second.supporting_evidence_ids == []
    assert second_step.tool_hint == []


def test_evidence_binding_is_strict_and_defaults_are_isolated() -> None:
    first = DiagnosisEvidence(
        id="E1",
        source="monitor",
        tool_name="query_cpu_metrics",
        hypothesis_ids=["H1"],
        observation="CPU max is 99%",
    )
    second = DiagnosisEvidence(
        id="E2",
        source="logs",
        tool_name="search_log",
        hypothesis_ids=["H1"],
        observation="No timeout logs",
    )
    first.supports.append("H1")

    assert second.supports == []
    with pytest.raises(ValidationError, match="bound hypothesis_ids"):
        DiagnosisEvidence(
            id="E3",
            source="monitor",
            tool_name="query_cpu_metrics",
            hypothesis_ids=["H1"],
            observation="CPU max is 99%",
            supports=["H2"],
        )

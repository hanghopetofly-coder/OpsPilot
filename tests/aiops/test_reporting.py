"""Evidence-whitelist tests for deterministic grounded reports."""

from __future__ import annotations

from typing import Any

from app.agent.aiops.reporting import (
    build_grounded_report,
    validate_report_grounding,
)


def _evaluation() -> dict[str, Any]:
    return {
        "supported_hypotheses": ["H1"],
        "contradicted_hypotheses": [],
        "uncertain_hypotheses": [],
        "missing_evidence": [],
        "scheduled_evidence": [],
        "evidence_conflicts": [],
        "tool_failures": [],
        "diagnosis_confidence": 0.91,
        "need_replan": False,
        "can_finish": True,
        "budget_exhausted": False,
        "reason": "Two independent online sources support H1.",
    }


def test_generated_report_references_only_state_evidence_ids(
    hypothesis_factory: Any,
    evidence_factory: Any,
) -> None:
    evidence = [
        evidence_factory(
            "E-monitor",
            source="monitor",
            supports=["H1"],
            observation="CPU max was 98% during the incident window",
        ),
        evidence_factory(
            "E-logs",
            source="logs",
            supports=["H1"],
            observation="Repeated worker starvation errors occurred in the same window",
        ),
    ]
    state = {
        "hypotheses": [hypothesis_factory()],
        "evidence": evidence,
        "evaluation": _evaluation(),
        # This unsupported text must never be copied into the deterministic report.
        "input": "Invented incident fact with fake citation E-fake-input",
        "raw_tool_results": [{"content": "fake observation E-fake-raw"}],
    }

    report = build_grounded_report(state)

    assert "[E-monitor]" in report
    assert "[E-logs]" in report
    assert "E-fake-input" not in report
    assert "E-fake-raw" not in report
    assert validate_report_grounding(report, evidence) is True


def test_grounding_validator_rejects_unknown_evidence_id(evidence_factory: Any) -> None:
    evidence = [evidence_factory("E1", source="monitor", supports=["H1"])]

    assert validate_report_grounding("Supported by [E1].", evidence) is True
    assert validate_report_grounding("Supported by [E1] and [E999].", evidence) is False


def test_application_error_code_is_not_mistaken_for_evidence_id(
    hypothesis_factory: Any,
    evidence_factory: Any,
) -> None:
    evidence = [
        evidence_factory(
            "E-logs",
            source="logs",
            supports=["H1"],
            observation="Application returned E1234 and literal [E5678] during the incident",
        )
    ]
    evaluation = _evaluation()
    evaluation["diagnosis_confidence"] = 0.6

    report = build_grounded_report(
        {
            "hypotheses": [hypothesis_factory()],
            "evidence": evidence,
            "evaluation": evaluation,
        }
    )

    assert "E1234" in report
    assert "&#91;E5678&#93;" in report
    assert validate_report_grounding(report, evidence) is True


def test_knowledge_only_report_remains_explicitly_uncertain(
    hypothesis_factory: Any,
    evidence_factory: Any,
) -> None:
    knowledge = evidence_factory(
        "KE1",
        source="knowledge_base",
        supports=["H1"],
        observation="Runbook recommends checking CPU saturation",
    )
    evaluation = _evaluation()
    evaluation.update(
        {
            "supported_hypotheses": [],
            "uncertain_hypotheses": ["H1"],
            "missing_evidence": ["H1: online CPU metric"],
            "diagnosis_confidence": 0.0,
            "reason": "Knowledge alone is insufficient.",
        }
    )

    report = build_grounded_report(
        {
            "hypotheses": [hypothesis_factory(expected_evidence=["online CPU metric"])],
            "evidence": [knowledge],
            "evaluation": evaluation,
        }
    )

    assert "尚未收集到可验证的在线观测" in report
    assert "不能确认其为线上根因" in report
    assert validate_report_grounding(report, [knowledge]) is True

"""Replanner merge-boundary tests."""

from __future__ import annotations

from app.agent.aiops.models import (
    DiagnosisHypothesis,
    DiagnosisStep,
    ReplanDecision,
)
from app.agent.aiops.replanner import _merge_replan


def test_colliding_revised_hypothesis_gets_fresh_id_and_clean_evidence_links() -> None:
    state = {
        "hypotheses": [
            DiagnosisHypothesis(
                id="H1",
                cause="CPU saturation",
                expected_evidence=["CPU metric"],
                supporting_evidence_ids=["E-existing"],
                confidence=0.9,
                status="supported",
            ).model_dump(mode="json")
        ],
        "plan": [],
        "execution_history": [],
        "step_count": 1,
        "tool_call_count": 1,
        "replan_count": 0,
    }
    decision = ReplanDecision(
        new_hypotheses=[
            DiagnosisHypothesis(
                id="H1",
                cause="Downstream timeout",
                expected_evidence=["downstream timeout logs"],
                supporting_evidence_ids=["E-invented"],
                confidence=1.0,
                status="supported",
            )
        ],
        steps=[
            DiagnosisStep(
                id="RP1S1",
                hypothesis_id="H1",
                goal="Verify downstream timeout",
                rationale="Conflict introduced an alternative cause",
                tool_hint=["search_log"],
                expected_evidence=["downstream timeout logs"],
            )
        ],
        reason="Revise the conflicting hypothesis",
    )

    hypotheses, steps = _merge_replan(state, decision, {"search_log"})

    by_id = {item["id"]: item for item in hypotheses}
    assert by_id["H1"]["cause"] == "CPU saturation"
    assert by_id["H1"]["supporting_evidence_ids"] == ["E-existing"]
    assert by_id["H2"]["cause"] == "Downstream timeout"
    assert by_id["H2"]["supporting_evidence_ids"] == []
    assert by_id["H2"]["contradicting_evidence_ids"] == []
    assert by_id["H2"]["confidence"] == 0.0
    assert by_id["H2"]["status"] == "pending"
    assert steps[0]["hypothesis_id"] == "H2"

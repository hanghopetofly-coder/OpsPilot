"""Deterministic execution and metric aggregation for offline AIOps cases."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from app.agent.aiops.evidence import (
    apply_evaluation_to_hypotheses,
    evaluate_evidence,
    extract_evidence,
    tool_error_from_raw_result,
)
from app.agent.aiops.models import (
    DiagnosisEvidence,
    RootCauseCandidate,
    ToolError,
)
from app.agent.aiops.reporting import (
    build_grounded_report,
    rank_root_causes,
    validate_report_grounding,
)

from .cases import CASES, EvaluationCase

OFFLINE_TOOL_BUDGET = 10


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Serializable outcome for one offline diagnosis case."""

    case_id: str
    scenario: str
    expected_root_cause: str | None
    predicted_root_cause: str | None
    top_candidates: tuple[dict[str, Any], ...]
    tool_calls: int
    replans: int
    steps: int
    success: bool
    tool_failure: bool
    tool_failure_recovered: bool
    unsupported_conclusion: bool
    report_grounded: bool
    termination_reason: str
    evaluation: dict[str, Any]
    baseline: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["top_candidates"] = list(self.top_candidates)
        return value


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregate metrics with explicit numerators and denominators."""

    cases: int
    root_cause_cases: int
    top_1_hits: int
    top_3_hits: int
    successful_diagnoses: int
    tool_failure_cases: int
    recovered_tool_failures: int
    top_1_accuracy: float
    top_3_recall: float
    average_tool_calls: float
    average_replans: float
    average_steps: float
    successful_diagnosis_rate: float
    tool_failure_recovery_rate: float
    unsupported_conclusion_count: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Full offline evaluation result, including a future baseline label."""

    metrics: EvaluationMetrics
    case_results: tuple[CaseResult, ...]
    baseline: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "metrics": self.metrics.to_dict(),
            "cases": [item.to_dict() for item in self.case_results],
        }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _is_confirmed(
    candidate: RootCauseCandidate,
    evidence_by_id: dict[str, DiagnosisEvidence],
) -> bool:
    supporting = [
        evidence_by_id[evidence_id]
        for evidence_id in candidate.supporting_evidence_ids
        if evidence_id in evidence_by_id and evidence_by_id[evidence_id].source != "knowledge_base"
    ]
    return (
        candidate.confidence >= 0.50
        and len(supporting) >= 2
        and len({item.source for item in supporting}) >= 2
        and not candidate.contradicting_evidence_ids
    )


def _unsupported_candidate(
    candidate: RootCauseCandidate,
    evidence_by_id: dict[str, DiagnosisEvidence],
) -> bool:
    """Check the evidence relationships used by a confirmed conclusion."""

    if not _is_confirmed(candidate, evidence_by_id):
        return False
    for evidence_id in candidate.supporting_evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None or candidate.hypothesis_id not in evidence.supports:
            return True
    for evidence_id in candidate.contradicting_evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None or candidate.hypothesis_id not in evidence.contradicts:
            return True
    return False


def _termination_reason(evaluation: dict[str, Any], tool_failure: bool) -> str:
    if evaluation.get("budget_exhausted"):
        return "investigation_budget_exhausted"
    if evaluation.get("can_finish") and evaluation.get("supported_hypotheses"):
        return "evidence_sufficient"
    if evaluation.get("evidence_conflicts"):
        return "evidence_conflict_unresolved"
    if tool_failure:
        return "tool_unavailable_partial_diagnosis"
    return "inconclusive_partial_diagnosis"


def evaluate_case(case: EvaluationCase, *, baseline: str | None = None) -> CaseResult:
    """Run one case through the production extractor/evaluator/ranker/reporter."""

    evidence: list[DiagnosisEvidence] = []
    errors: list[ToolError] = []
    first_failure_index: int | None = None
    later_success = False
    for index, raw in enumerate(case.raw_results):
        if raw.is_error:
            first_failure_index = index if first_failure_index is None else first_failure_index
            errors.append(tool_error_from_raw_result(raw))
            continue
        if first_failure_index is not None and index > first_failure_index:
            later_success = True
        evidence.extend(extract_evidence(raw))

    # The catalog represents a completed fallback attempt by placing successful
    # alternative-source calls after the failed call.  Mark the original error
    # handled before deterministic evaluation, mirroring Replanner behaviour.
    if later_success:
        errors = [item.model_copy(update={"handled": True}) for item in errors]

    updated_hypotheses = apply_evaluation_to_hypotheses(case.hypotheses, evidence)
    evaluation = evaluate_evidence(
        updated_hypotheses,
        evidence,
        errors,
        remaining_budget=max(0, OFFLINE_TOOL_BUDGET - case.tool_calls),
    )
    candidates = rank_root_causes(updated_hypotheses, evidence, evaluation)
    evidence_by_id = {item.id: item for item in evidence}
    confirmed = [item for item in candidates if _is_confirmed(item, evidence_by_id)]
    predicted_root_cause = confirmed[0].cause if confirmed else None

    evaluation_payload = evaluation.model_dump(mode="json")
    termination_reason = _termination_reason(evaluation_payload, bool(errors))
    state = {
        "hypotheses": [item.model_dump(mode="json") for item in updated_hypotheses],
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "evaluation": evaluation_payload,
        "termination_reason": termination_reason,
    }
    report = build_grounded_report(state)
    grounded = validate_report_grounding(report, evidence)
    unsupported = (not grounded) or any(
        _unsupported_candidate(item, evidence_by_id) for item in confirmed
    )
    expected_match = (
        predicted_root_cause == case.expected_root_cause
        if case.expected_root_cause is not None
        else predicted_root_cause is None
    )
    success = expected_match and grounded and not unsupported
    recovered = bool(errors) and later_success and predicted_root_cause is not None

    top_candidates = tuple(
        {
            "hypothesis_id": item.hypothesis_id,
            "cause": item.cause,
            "confidence": item.confidence,
            "supporting_evidence_ids": list(item.supporting_evidence_ids),
            "contradicting_evidence_ids": list(item.contradicting_evidence_ids),
            "confirmed": _is_confirmed(item, evidence_by_id),
        }
        for item in candidates
    )
    return CaseResult(
        case_id=case.id,
        scenario=case.scenario,
        expected_root_cause=case.expected_root_cause,
        predicted_root_cause=predicted_root_cause,
        top_candidates=top_candidates,
        tool_calls=case.tool_calls,
        replans=case.replans,
        steps=case.steps,
        success=success,
        tool_failure=bool(errors),
        tool_failure_recovered=recovered,
        unsupported_conclusion=unsupported,
        report_grounded=grounded,
        termination_reason=termination_reason,
        evaluation=evaluation_payload,
        baseline=case.baseline or baseline,
    )


def aggregate_results(results: Sequence[CaseResult]) -> EvaluationMetrics:
    """Calculate all published metrics from case-level records."""

    case_count = len(results)
    scored = [item for item in results if item.expected_root_cause is not None]
    top_1_hits = sum(
        bool(item.top_candidates) and item.top_candidates[0]["cause"] == item.expected_root_cause
        for item in scored
    )
    top_3_hits = sum(
        item.expected_root_cause in {candidate["cause"] for candidate in item.top_candidates[:3]}
        for item in scored
    )
    failures = [item for item in results if item.tool_failure]
    recovered = sum(item.tool_failure_recovered for item in failures)
    divisor = case_count or 1
    return EvaluationMetrics(
        cases=case_count,
        root_cause_cases=len(scored),
        top_1_hits=top_1_hits,
        top_3_hits=top_3_hits,
        successful_diagnoses=sum(item.success for item in results),
        tool_failure_cases=len(failures),
        recovered_tool_failures=recovered,
        top_1_accuracy=_safe_ratio(top_1_hits, len(scored)),
        top_3_recall=_safe_ratio(top_3_hits, len(scored)),
        average_tool_calls=sum(item.tool_calls for item in results) / divisor,
        average_replans=sum(item.replans for item in results) / divisor,
        average_steps=sum(item.steps for item in results) / divisor,
        successful_diagnosis_rate=_safe_ratio(sum(item.success for item in results), case_count),
        tool_failure_recovery_rate=_safe_ratio(recovered, len(failures)),
        unsupported_conclusion_count=sum(item.unsupported_conclusion for item in results),
    )


def run_evaluation(
    cases: Sequence[EvaluationCase] = CASES,
    *,
    baseline: str | None = None,
) -> EvaluationSummary:
    """Evaluate a case collection without using an LLM, MCP or the network."""

    results = tuple(evaluate_case(case, baseline=baseline) for case in cases)
    return EvaluationSummary(
        metrics=aggregate_results(results),
        case_results=results,
        baseline=baseline,
    )


__all__ = [
    "CaseResult",
    "EvaluationMetrics",
    "EvaluationSummary",
    "aggregate_results",
    "evaluate_case",
    "run_evaluation",
]

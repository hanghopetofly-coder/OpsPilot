"""Offline, deterministic evaluation for the evidence-grounded AIOps workflow."""

from .cases import CASES, EvaluationCase
from .runner import (
    CaseResult,
    EvaluationMetrics,
    EvaluationSummary,
    aggregate_results,
    evaluate_case,
    run_evaluation,
)

__all__ = [
    "CASES",
    "CaseResult",
    "EvaluationCase",
    "EvaluationMetrics",
    "EvaluationSummary",
    "aggregate_results",
    "evaluate_case",
    "run_evaluation",
]

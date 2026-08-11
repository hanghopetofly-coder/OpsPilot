"""Contract and metric tests for the deterministic offline evaluation."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from evaluation.cases import CASES
from evaluation.run import main
from evaluation.runner import aggregate_results, run_evaluation


def test_catalog_contains_exactly_twenty_fixed_cases_with_required_coverage() -> None:
    assert len(CASES) == 20
    assert len({item.id for item in CASES}) == 20
    assert Counter(item.scenario for item in CASES) == {
        "cpu_saturation": 3,
        "memory_pressure": 3,
        "database_timeout": 3,
        "downstream_timeout": 3,
        "normal": 2,
        "conflicting_evidence": 3,
        "tool_unavailable": 3,
    }
    assert all(item.tool_calls >= 1 for item in CASES)
    assert all(item.steps >= 1 for item in CASES)


def test_metrics_are_derived_from_case_records_and_reports_remain_grounded() -> None:
    summary = run_evaluation()
    results = summary.case_results
    metrics = summary.metrics
    scored = [item for item in results if item.expected_root_cause is not None]
    failures = [item for item in results if item.tool_failure]

    expected_top_1 = sum(
        item.top_candidates[0]["cause"] == item.expected_root_cause for item in scored
    )
    expected_top_3 = sum(
        item.expected_root_cause in {candidate["cause"] for candidate in item.top_candidates[:3]}
        for item in scored
    )
    assert metrics.cases == len(results) == 20
    assert metrics.root_cause_cases == len(scored)
    assert metrics.top_1_hits == expected_top_1
    assert metrics.top_3_hits == expected_top_3
    assert math.isclose(metrics.top_1_accuracy, expected_top_1 / len(scored))
    assert math.isclose(metrics.top_3_recall, expected_top_3 / len(scored))
    assert math.isclose(
        metrics.average_tool_calls,
        sum(item.tool_calls for item in results) / len(results),
    )
    assert math.isclose(
        metrics.average_replans,
        sum(item.replans for item in results) / len(results),
    )
    assert math.isclose(
        metrics.average_steps,
        sum(item.steps for item in results) / len(results),
    )
    assert math.isclose(
        metrics.successful_diagnosis_rate,
        sum(item.success for item in results) / len(results),
    )
    assert math.isclose(
        metrics.tool_failure_recovery_rate,
        sum(item.tool_failure_recovered for item in failures) / len(failures),
    )
    assert metrics.tool_failure_cases == 3
    assert metrics.recovered_tool_failures == 2
    assert metrics.unsupported_conclusion_count == 0
    assert all(item.report_grounded for item in results)
    assert all(not item.unsupported_conclusion for item in results)


def test_normal_conflict_and_unavailable_cases_do_not_force_a_root_cause() -> None:
    results = run_evaluation().case_results
    uncertain = [
        item
        for item in results
        if item.expected_root_cause is None
        and item.scenario in {"normal", "conflicting_evidence", "tool_unavailable"}
    ]

    assert len(uncertain) == 6
    assert all(item.predicted_root_cause is None for item in uncertain)
    assert all(item.success for item in uncertain)
    assert all(
        not any(candidate["confirmed"] for candidate in item.top_candidates) for item in uncertain
    )


def test_aggregate_results_handles_an_empty_baseline() -> None:
    metrics = aggregate_results([])

    assert metrics.cases == 0
    assert metrics.top_1_accuracy == 0.0
    assert metrics.top_3_recall == 0.0
    assert metrics.successful_diagnosis_rate == 0.0
    assert metrics.tool_failure_recovery_rate == 0.0
    assert metrics.unsupported_conclusion_count == 0


def test_cli_emits_required_text_and_machine_readable_json(
    capsys: Any,
) -> None:
    assert main([]) == 0
    text_output = capsys.readouterr().out
    for label in (
        "Cases: 20",
        "Top-1 Accuracy:",
        "Top-3 Recall:",
        "Average Tool Calls:",
        "Average Replans:",
        "Average Steps:",
        "Successful Diagnosis Rate:",
        "Tool Failure Recovery Rate:",
        "Unsupported Conclusion Count: 0",
    ):
        assert label in text_output

    assert main(["--json", "--baseline", "candidate-v1"]) == 0
    json_output = capsys.readouterr().out
    payload = json.loads(json_output)
    assert payload["baseline"] == "candidate-v1"
    assert payload["metrics"]["cases"] == 20
    assert payload["metrics"]["unsupported_conclusion_count"] == 0
    assert len(payload["cases"]) == 20
    assert all(item["baseline"] == "candidate-v1" for item in payload["cases"])
    required_case_fields = {
        "expected_root_cause",
        "top_candidates",
        "tool_calls",
        "replans",
        "steps",
        "success",
        "tool_failure",
        "tool_failure_recovered",
        "unsupported_conclusion",
    }
    assert all(required_case_fields <= set(item) for item in payload["cases"])

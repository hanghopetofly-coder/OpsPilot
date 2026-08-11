"""Scenario outputs must create a real same-hypothesis Evidence conflict."""

from __future__ import annotations

from app.agent.aiops.evidence import evaluate_evidence, extract_evidence
from app.agent.aiops.models import RawToolResult
from mcp_servers.cls_server import search_log
from mcp_servers.monitor_server import query_cpu_metrics

START = "2026-02-14 10:00:00"
END = "2026-02-14 11:00:00"
START_MS = 1_771_063_200_000
END_MS = 1_771_066_800_000


def test_conflicting_scenario_extracts_support_and_contradiction_then_replans() -> None:
    metric_payload = query_cpu_metrics(
        "checkout-service",
        START,
        END,
        "5m",
        "conflicting_evidence",
    )
    log_payload = search_log(
        "topic-001",
        START_MS,
        END_MS,
        None,
        100,
        "checkout-service",
        "conflicting_evidence",
    )
    metric_raw = RawToolResult(
        id="R-metric",
        step_id="S-metric",
        hypothesis_ids=["H-CPU"],
        tool_name="query_cpu_metrics",
        source="monitor",
        diagnostic_goal="Check whether CPU saturation is present",
        expected_evidence=["CPU usage"],
        arguments={"service_name": "checkout-service"},
        payload=metric_payload,
    )
    log_raw = RawToolResult(
        id="R-logs",
        step_id="S-logs",
        hypothesis_ids=["H-CPU"],
        tool_name="search_log",
        source="logs",
        diagnostic_goal="Check logs for CPU saturation",
        expected_evidence=["CPU saturation errors"],
        arguments={"service_name": "checkout-service"},
        payload=log_payload,
    )

    metric_evidence = extract_evidence(metric_raw)
    log_evidence = extract_evidence(log_raw)

    assert metric_evidence[0].supports == []
    assert metric_evidence[0].contradicts == ["H-CPU"]
    assert log_evidence[0].supports == ["H-CPU"]
    assert log_evidence[0].contradicts == []

    evaluation = evaluate_evidence(
        [
            {
                "id": "H-CPU",
                "cause": "CPU saturation",
                "description": "Worker capacity is exhausted by CPU load",
                "expected_evidence": ["CPU usage", "CPU saturation errors"],
            }
        ],
        [*metric_evidence, *log_evidence],
        remaining_budget=3,
    )

    assert evaluation.evidence_conflicts
    assert "H-CPU" in evaluation.evidence_conflicts[0]
    assert evaluation.need_replan is True
    assert evaluation.can_finish is False

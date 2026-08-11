"""Deterministic extraction and tool-error normalization tests."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.aiops.evidence import (
    evaluate_evidence,
    evidence_extractor,
    extract_evidence,
    tool_error_from_raw_result,
)


@pytest.mark.parametrize(
    ("tool_name", "metric_name", "statistics", "threshold"),
    [
        (
            "query_cpu_metrics",
            "cpu_usage_percent",
            {"avg": 87.5, "max": 98.0, "min": 42.0, "p95": 96.0},
            80.0,
        ),
        (
            "query_memory_metrics",
            "memory_usage_percent",
            {"avg": 84.0, "max": 97.0, "min": 61.0, "p95": 95.5},
            70.0,
        ),
    ],
)
def test_cpu_and_memory_metrics_become_bounded_monitor_evidence(
    raw_result_factory: Any,
    tool_name: str,
    metric_name: str,
    statistics: dict[str, float],
    threshold: float,
) -> None:
    raw = raw_result_factory(
        tool_name=tool_name,
        source="monitor",
        diagnostic_goal=f"Verify {metric_name} saturation",
        expected_evidence=[f"{metric_name} above threshold"],
        payload={
            "service_name": "checkout-service",
            "metric_name": metric_name,
            "interval": "10m",
            "statistics": statistics,
            "alert_info": {
                "triggered": True,
                "threshold": threshold,
                "message": "threshold exceeded",
            },
        },
    )

    result = extract_evidence(raw, max_chars=300)

    assert len(result) == 1
    assert result[0].source == "monitor"
    assert result[0].supports == ["H1"]
    assert result[0].contradicts == []
    assert result[0].raw_result_ref == "R1"
    assert metric_name in result[0].observation
    assert len(result[0].observation) <= 300


def test_logs_are_aggregated_limited_and_keep_raw_reference(raw_result_factory: Any) -> None:
    logs = [
        {
            "timestamp": f"2026-08-09 10:{index:02d}:00",
            "level": "ERROR",
            "message": f"connection timeout request_id={1000 + index}",
        }
        for index in range(12)
    ]
    raw = raw_result_factory(
        tool_name="search_log",
        source="logs",
        payload={"total": 126, "logs": logs, "query": "level:ERROR"},
        diagnostic_goal="Verify repeated connection timeouts",
        expected_evidence=["connection timeout errors"],
    )

    result = extract_evidence(raw, max_chars=400, max_log_records=4)

    assert len(result) == 1
    evidence = result[0]
    assert evidence.source == "logs"
    assert evidence.supports == ["H1"]
    assert evidence.raw_result_ref == "R1"
    assert "total=126, sampled=4" in evidence.observation
    assert "x4" in evidence.observation
    assert "8 in-payload records omitted" in evidence.observation
    assert len(evidence.observation) <= 400


def test_prometheus_alerts_honor_item_bound(raw_result_factory: Any) -> None:
    alerts = [
        {
            "labels": {
                "alertname": f"HighLatency{index}",
                "service": "checkout-service",
                "severity": "critical",
            },
            "annotations": {"summary": "P95 latency above SLO"},
            "state": "firing",
        }
        for index in range(3)
    ]
    raw = raw_result_factory(
        tool_name="query_prometheus_alerts",
        source="prometheus",
        payload={"alerts": alerts},
        diagnostic_goal="Verify a latency alert",
        expected_evidence=["P95 latency alert"],
    )

    result = extract_evidence(raw, max_items=2, max_chars=280)

    assert len(result) == 2
    assert all(item.source == "prometheus" for item in result)
    assert all(item.supports == ["H1"] for item in result)
    assert all(item.raw_result_ref == "R1" for item in result)
    assert "1 additional alerts omitted" in result[-1].observation


def test_unrelated_active_alert_and_generic_error_do_not_support_hypothesis(
    raw_result_factory: Any,
) -> None:
    alert_raw = raw_result_factory(
        tool_name="query_prometheus_alerts",
        source="prometheus",
        diagnostic_goal="Verify database connection-pool exhaustion",
        expected_evidence=["database connection-pool saturation"],
        payload={
            "alerts": [
                {
                    "labels": {"alertname": "HighCPU", "service": "checkout"},
                    "annotations": {"summary": "CPU above threshold"},
                    "state": "firing",
                }
            ]
        },
    )
    log_raw = raw_result_factory(
        raw_id="R2",
        tool_name="search_log",
        source="logs",
        diagnostic_goal="Verify database connection-pool exhaustion",
        expected_evidence=["database connection-pool saturation"],
        payload={
            "total": 1,
            "logs": [
                {
                    "timestamp": "2026-08-09 10:00:00",
                    "level": "ERROR",
                    "message": "unclassified application error",
                }
            ],
        },
    )

    alert = extract_evidence(alert_raw)[0]
    log = extract_evidence(log_raw)[0]

    assert alert.supports == []
    assert alert.contradicts == []
    assert log.supports == []
    assert log.contradicts == []


def test_resolved_alert_and_unrelated_error_cannot_be_combined_into_support(
    raw_result_factory: Any,
    hypothesis_factory: Any,
) -> None:
    alert_raw = raw_result_factory(
        tool_name="query_prometheus_alerts",
        source="prometheus",
        diagnostic_goal="Verify database connection-pool exhaustion",
        expected_evidence=["database connection-pool health alert"],
        payload={
            "alerts": [
                {
                    "labels": {
                        "alertname": "DatabaseConnectionPoolHealthy",
                        "service": "checkout",
                    },
                    "annotations": {"summary": "database connection pool is healthy"},
                    "state": "resolved",
                }
            ]
        },
    )
    log_raw = raw_result_factory(
        raw_id="R2",
        tool_name="search_log",
        source="logs",
        diagnostic_goal="Verify database connection-pool exhaustion",
        expected_evidence=["database connection-pool logs"],
        payload={
            "total": 2,
            "logs": [
                {
                    "timestamp": "2026-08-09 10:00:00",
                    "level": "INFO",
                    "message": "database connection pool healthy",
                },
                {
                    "timestamp": "2026-08-09 10:00:01",
                    "level": "ERROR",
                    "message": "disk full",
                },
            ],
        },
    )

    evidence = [*extract_evidence(alert_raw), *extract_evidence(log_raw)]
    evaluation = evaluate_evidence(
        [
            hypothesis_factory(
                cause="Database connection-pool exhaustion",
                expected_evidence=[
                    "database connection-pool health alert",
                    "database connection-pool logs",
                ],
            )
        ],
        evidence,
        remaining_budget=4,
    )

    assert all(item.supports == [] for item in evidence)
    assert all(item.contradicts == ["H1"] for item in evidence)
    assert evaluation.supported_hypotheses == []
    assert evaluation.can_finish is False


def test_unrelated_normal_cpu_metric_is_neutral(raw_result_factory: Any) -> None:
    raw = raw_result_factory(
        tool_name="query_cpu_metrics",
        source="monitor",
        diagnostic_goal="Verify database connection-pool exhaustion",
        expected_evidence=["database pool utilization"],
        payload={
            "service_name": "checkout",
            "metric_name": "cpu_usage_percent",
            "statistics": {"avg": 25.0, "max": 30.0, "min": 20.0, "p95": 29.0},
            "alert_info": {"triggered": False, "threshold": 80.0},
        },
    )

    evidence = extract_evidence(raw)[0]

    assert evidence.supports == []
    assert evidence.contradicts == []


def test_broad_database_word_does_not_confirm_connection_pool_hypothesis(
    raw_result_factory: Any,
) -> None:
    common = {
        "diagnostic_goal": "Verify database connection-pool exhaustion",
        "expected_evidence": ["database connection-pool saturation"],
    }
    alert_raw = raw_result_factory(
        tool_name="query_prometheus_alerts",
        source="prometheus",
        payload={
            "alerts": [
                {
                    "labels": {"alertname": "DatabaseBackupFailed"},
                    "annotations": {"summary": "database snapshot backup failed"},
                    "state": "firing",
                }
            ]
        },
        **common,
    )
    log_raw = raw_result_factory(
        raw_id="R2",
        tool_name="search_log",
        source="logs",
        payload={
            "logs": [
                {
                    "level": "ERROR",
                    "message": "database schema migration failed",
                }
            ]
        },
        **common,
    )

    alert = extract_evidence(alert_raw)[0]
    log = extract_evidence(log_raw)[0]

    assert alert.supports == []
    assert log.supports == []


def test_knowledge_guides_but_never_supports_online_hypothesis(
    raw_result_factory: Any,
) -> None:
    raw = raw_result_factory(
        tool_name="retrieve_knowledge",
        source="knowledge_base",
        payload={"context": "CPU saturation runbook: " + "inspect threads " * 100},
    )

    result = extract_evidence(raw, max_chars=180)

    assert len(result) == 1
    evidence = result[0]
    assert evidence.source == "knowledge_base"
    assert evidence.supports == []
    assert evidence.contradicts == []
    assert evidence.raw_result_ref == "R1"
    assert "not an online observation" in evidence.observation
    assert evidence.observation.endswith("…[truncated]")
    assert len(evidence.observation) <= 180


def test_generic_evidence_is_neutral_truncated_and_traceable(raw_result_factory: Any) -> None:
    raw = raw_result_factory(
        tool_name="custom_inventory_tool",
        source="generic",
        payload={"details": "x" * 1_000},
    )

    result = extract_evidence(raw, max_chars=120)

    assert len(result) == 1
    evidence = result[0]
    assert evidence.source == "generic"
    assert evidence.supports == []
    assert evidence.contradicts == []
    assert evidence.raw_result_ref == "R1"
    assert evidence.observation.endswith("…[truncated]")
    assert len(evidence.observation) <= 120


@pytest.mark.parametrize(
    ("message", "expected_category", "retryable"),
    [
        ("Monitor MCP timed out after 10 seconds", "timeout", True),
        ("connection refused by CLS endpoint", "connection", True),
        ("invalid result: missing required field", "invalid_result", False),
    ],
)
def test_tool_errors_are_classified_deterministically(
    raw_result_factory: Any,
    message: str,
    expected_category: str,
    retryable: bool,
) -> None:
    raw = raw_result_factory(
        content=message,
        is_error=True,
        error_type=None,
    )

    error = tool_error_from_raw_result(raw)

    assert error.category == expected_category
    assert error.retryable is retryable
    assert error.id == "TE-R1"
    assert error.step_id == "S1"
    assert error.hypothesis_id == "H1"


def test_extractor_converts_timeout_connection_and_malformed_raw_without_crashing(
    raw_result_factory: Any,
) -> None:
    timeout = raw_result_factory(
        "R-timeout",
        content="request timed out",
        is_error=True,
        error_type="timeout",
    )
    connection = raw_result_factory(
        "R-connection",
        tool_name="search_log",
        source="logs",
        content="connection refused",
        is_error=True,
        error_type="connection",
    )
    malformed = {
        "id": "R-invalid",
        "step_id": "S-invalid",
        "tool_name": "invalid_payload_tool",
        # Missing required hypothesis_ids is intentionally invalid.
        "payload": {"statistics": "not-a-mapping"},
    }

    result = evidence_extractor(
        {
            "raw_tool_results": [timeout, connection, malformed],
            "evidence": [],
            "tool_errors": [],
        }
    )

    assert result["evidence"] == []
    assert {item["category"] for item in result["tool_errors"]} == {
        "timeout",
        "connection",
        "invalid_result",
    }
    assert all(item.get("evidence_extracted") is True for item in result["raw_tool_results"])

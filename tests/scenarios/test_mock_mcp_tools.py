"""Direct, offline tests of the existing FastMCP tool functions."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from mcp_servers.cls_server import (
    get_current_timestamp,
    search_log,
    search_topic_by_service_name,
)
from mcp_servers.monitor_server import query_cpu_metrics, query_memory_metrics
from mcp_servers.scenarios import ScenarioToolUnavailable, expected_root_cause

START = "2026-02-14 10:00:00"
END = "2026-02-14 11:00:00"
START_MS = 1_771_063_200_000
END_MS = 1_771_066_800_000


def _metric_pair(scenario: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    cpu = query_cpu_metrics("checkout-service", START, END, "5m", scenario)  # type: ignore[arg-type]
    memory = query_memory_metrics("checkout-service", START, END, "5m", scenario)  # type: ignore[arg-type]
    return cpu, memory


def _logs(scenario: str | None, *, topic_id: str = "topic-001") -> dict[str, Any]:
    return search_log(
        topic_id,
        START_MS,
        END_MS,
        None,
        100,
        "checkout-service",
        scenario,  # type: ignore[arg-type]
    )


def test_fastmcp_tool_signatures_expose_scenario_without_breaking_existing_arguments() -> None:
    cpu_parameters = list(inspect.signature(query_cpu_metrics).parameters)
    memory_parameters = list(inspect.signature(query_memory_metrics).parameters)
    log_parameters = list(inspect.signature(search_log).parameters)

    assert cpu_parameters == [
        "service_name",
        "start_time",
        "end_time",
        "interval",
        "scenario",
    ]
    assert memory_parameters == cpu_parameters
    assert log_parameters[:5] == ["topic_id", "start_time", "end_time", "query", "limit"]
    assert log_parameters[-1] == "scenario"


def test_default_normal_scenario_is_fixed_repeatable_and_does_not_false_alarm() -> None:
    first_cpu, first_memory = _metric_pair(None)
    second_cpu, second_memory = _metric_pair(None)

    assert (first_cpu, first_memory) == (second_cpu, second_memory)
    assert first_cpu["scenario"] == first_memory["scenario"] == "normal"
    assert first_cpu["start_time"] == START
    assert first_cpu["end_time"] == END
    assert first_cpu["statistics"]["max"] < first_cpu["threshold"]
    assert first_memory["statistics"]["max"] < first_memory["threshold"]
    assert first_cpu["alert_info"]["triggered"] is False
    assert first_memory["alert_info"]["triggered"] is False
    assert first_cpu["anomalous_intervals"] == []
    assert first_memory["anomalous_intervals"] == []
    assert {item["level"] for item in _logs(None)["logs"]} == {"INFO"}


def test_cpu_saturation_has_high_cpu_normal_memory_and_related_logs() -> None:
    cpu, memory = _metric_pair("cpu_saturation")
    logs = _logs("cpu_saturation")

    assert cpu["statistics"]["max"] >= 95.0
    assert cpu["statistics"]["p95"] >= 95.0
    assert cpu["alert_info"]["triggered"] is True
    assert memory["statistics"]["max"] < memory["threshold"]
    assert any("CPU" in item["message"] for item in logs["logs"])


def test_memory_pressure_has_high_memory_normal_cpu_and_oom_gc_logs() -> None:
    cpu, memory = _metric_pair("memory_pressure")
    logs = _logs("memory_pressure")

    assert cpu["statistics"]["max"] < cpu["threshold"]
    assert memory["statistics"]["max"] >= 95.0
    assert memory["statistics"]["p95"] >= 95.0
    assert memory["alert_info"]["triggered"] is True
    messages = " ".join(item["message"] for item in logs["logs"])
    assert "OutOfMemoryError" in messages
    assert "GC" in messages


@pytest.mark.parametrize(
    ("scenario", "log_fragment"),
    [
        ("database_timeout", "database query timeout"),
        ("downstream_timeout", "downstream payment-service timeout"),
    ],
)
def test_timeout_scenarios_keep_local_resources_normal_and_explain_logs(
    scenario: str,
    log_fragment: str,
) -> None:
    cpu, memory = _metric_pair(scenario)
    logs = _logs(scenario)

    assert cpu["statistics"]["max"] < cpu["threshold"]
    assert memory["statistics"]["max"] < memory["threshold"]
    assert any(log_fragment in item["message"] for item in logs["logs"])


def test_conflicting_evidence_has_normal_cpu_metrics_but_cpu_saturation_logs() -> None:
    cpu, memory = _metric_pair("conflicting_evidence")
    logs = _logs("conflicting_evidence")

    assert cpu["alert_info"]["triggered"] is False
    assert memory["alert_info"]["triggered"] is False
    assert any("CPU saturation" in item["message"] for item in logs["logs"])
    assert expected_root_cause("conflicting_evidence") == (
        "Unresolved: CPU metrics are normal while logs report CPU saturation"
    )
    assert "expected_root_cause" not in cpu
    assert "expected_root_cause" not in logs


@pytest.mark.parametrize("tool", [query_cpu_metrics, query_memory_metrics])
def test_tool_unavailable_is_a_classified_monitor_connection_failure(tool: Any) -> None:
    with pytest.raises(ScenarioToolUnavailable) as captured:
        tool("checkout-service", START, END, "5m", "tool_unavailable")

    assert captured.value.category == "connection"
    assert captured.value.code == "MOCK_TOOL_UNAVAILABLE"
    assert captured.value.retryable is True


def test_tool_unavailable_keeps_cls_available_for_fallback() -> None:
    logs = _logs("tool_unavailable")

    assert logs["success"] is True
    assert logs["scenario"] == "tool_unavailable"
    assert logs["logs"]
    assert {item["level"] for item in logs["logs"]} == {"INFO"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval": "0m"},
        {"interval": "-1m"},
        {"start_time": END, "end_time": START},
        {
            "start_time": "2026-01-01 00:00:00",
            "end_time": "2026-02-14 11:00:00",
            "interval": "1m",
        },
    ],
)
def test_monitor_rejects_invalid_or_unbounded_windows(kwargs: dict[str, Any]) -> None:
    call_kwargs = {
        "service_name": "checkout-service",
        "start_time": START,
        "end_time": END,
        "interval": "5m",
        "scenario": "normal",
        **kwargs,
    }
    with pytest.raises(ValueError):
        query_cpu_metrics(**call_kwargs)


@pytest.mark.parametrize("limit", [0, 101])
def test_cls_limit_is_strictly_bounded(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between"):
        search_log("topic-001", START_MS, END_MS, limit=limit)


def test_cls_rejects_reversed_window() -> None:
    with pytest.raises(ValueError, match="less than or equal"):
        search_log("topic-001", END_MS, START_MS)


def test_cls_response_retains_service_topic_time_query_and_limit() -> None:
    result = search_log(
        "topic-002",
        START_MS,
        END_MS,
        "level:ERROR",
        10,
        "data-sync-service",
        "database_timeout",
    )

    assert result["service_name"] == "data-sync-service"
    assert result["topic_id"] == "topic-002"
    assert result["start_time"] == START_MS
    assert result["end_time"] == END_MS
    assert result["query"] == "level:ERROR"
    assert result["limit"] == 10
    assert result["returned"] == len(result["logs"]) == 1
    assert result["total"] == 27


def test_every_discovered_topic_can_be_queried() -> None:
    discovered = search_topic_by_service_name("service")

    assert discovered["topics"]
    for topic in discovered["topics"]:
        result = search_log(
            topic["topic_id"],
            START_MS,
            END_MS,
            service_name=topic["service_name"],
        )
        assert result["success"] is True


def test_current_timestamp_is_fixed_instead_of_reading_wall_clock() -> None:
    assert get_current_timestamp() == END_MS

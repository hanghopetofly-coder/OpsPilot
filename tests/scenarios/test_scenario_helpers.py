"""Pure deterministic scenario builder tests."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from mcp_servers.scenarios import (
    DEFAULT_REFERENCE_TIME,
    EXPECTED_ROOT_CAUSES,
    MAX_METRIC_POINTS,
    SUPPORTED_SCENARIOS,
    build_log_result,
    build_metric_result,
    expected_root_cause,
    normalize_scenario,
    parse_interval_minutes,
    resolve_metric_window,
)


def test_manifest_contains_every_required_scenario_and_expected_diagnosis() -> None:
    assert set(SUPPORTED_SCENARIOS) == {
        "cpu_saturation",
        "memory_pressure",
        "database_timeout",
        "downstream_timeout",
        "conflicting_evidence",
        "tool_unavailable",
        "normal",
    }
    assert set(EXPECTED_ROOT_CAUSES) == set(SUPPORTED_SCENARIOS)
    assert expected_root_cause("cpu_saturation") == ("CPU saturation / exhausted worker capacity")
    assert expected_root_cause("normal") is None


@pytest.mark.parametrize("value", [None, "", "normal", " NORMAL "])
def test_omitted_or_blank_scenario_is_deterministic_normal(value: str | None) -> None:
    assert normalize_scenario(value) == "normal"


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported scenario"):
        normalize_scenario("random-overload")


@pytest.mark.parametrize(
    ("value", "expected_minutes"),
    [("1m", 1), ("5m", 5), ("1h", 60), ("24h", 1440)],
)
def test_positive_metric_intervals_are_parsed(value: str, expected_minutes: int) -> None:
    assert parse_interval_minutes(value) == expected_minutes


@pytest.mark.parametrize("value", ["0m", "-1m", "1s", "1.5m", "25h", "invalid"])
def test_non_positive_or_unreasonable_intervals_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="interval"):
        parse_interval_minutes(value)


def test_default_metric_window_uses_fixed_reference_time() -> None:
    window = resolve_metric_window(None, None, "1m")

    assert window.start == datetime(2026, 2, 14, 10, 0, 0)
    assert window.end == DEFAULT_REFERENCE_TIME
    assert window.interval_minutes == 1
    assert window.point_count == 61


def test_caller_metric_window_is_inclusive_and_finite() -> None:
    window = resolve_metric_window(
        "2026-02-14 10:00:00",
        "2026-02-14 10:10:00",
        "5m",
    )

    assert window.point_count == 3


def test_reversed_or_excessive_metric_window_is_rejected_before_generation() -> None:
    with pytest.raises(ValueError, match="less than or equal"):
        resolve_metric_window(
            "2026-02-14 11:00:00",
            "2026-02-14 10:00:00",
            "1m",
        )

    with pytest.raises(ValueError, match=f"maximum is {MAX_METRIC_POINTS}"):
        resolve_metric_window(
            "2026-02-01 00:00:00",
            "2026-02-14 00:00:00",
            "1m",
        )


def test_pure_metric_builder_is_json_safe_and_repeatable() -> None:
    window = resolve_metric_window(
        "2026-02-14 10:00:00",
        "2026-02-14 11:00:00",
        "5m",
    )
    kwargs = {
        "metric": "cpu",
        "scenario": "cpu_saturation",
        "service_name": "checkout-service",
        "window": window,
        "interval": "5m",
    }

    first = build_metric_result(**kwargs)  # type: ignore[arg-type]
    second = build_metric_result(**kwargs)  # type: ignore[arg-type]

    assert first == second
    json.dumps(first, ensure_ascii=False)
    assert first["statistics"]["p95"] >= 95.0
    assert first["anomalous_intervals"]
    assert first["trend"]["direction"] == "rising"


@pytest.mark.parametrize(
    ("metric", "scenario"),
    [("cpu", "cpu_saturation"), ("memory", "memory_pressure")],
)
def test_single_point_fault_window_keeps_peak_signal(metric: str, scenario: str) -> None:
    window = resolve_metric_window(
        "2026-02-14 11:00:00",
        "2026-02-14 11:00:00",
        "1m",
    )

    result = build_metric_result(
        metric=metric,  # type: ignore[arg-type]
        scenario=scenario,
        service_name="checkout-service",
        window=window,
        interval="1m",
    )

    assert window.point_count == 1
    assert len(result["data_points"]) == 1
    assert result["statistics"]["max"] >= 95.0
    assert result["alert_info"]["triggered"] is True


def test_pure_log_builder_samples_repeats_instead_of_returning_duplicate_rows() -> None:
    result = build_log_result(
        scenario="database_timeout",
        service_name="checkout-service",
        topic_id="topic-001",
        start_time=1_771_063_200_000,
        end_time=1_771_066_800_000,
        query=None,
        limit=100,
    )

    assert result["total"] > result["returned"] == len(result["logs"])
    assert result["returned"] <= 12
    repeated = result["statistics"]["repeated_errors"]
    assert repeated == [
        {
            "fingerprint": "database_query_timeout",
            "message": "database query timeout after 3000ms",
            "count": 27,
            "first_seen": "2026-02-14T10:00:00Z",
            "last_seen": "2026-02-14T11:00:00Z",
        }
    ]


def test_log_query_filter_is_applied_before_aggregation() -> None:
    result = build_log_result(
        scenario="memory_pressure",
        service_name="checkout-service",
        topic_id="topic-001",
        start_time=1_771_063_200_000,
        end_time=1_771_066_800_000,
        query="level:ERROR",
        limit=10,
    )

    assert result["statistics"]["level_counts"] == {"ERROR": 14}
    assert {item["level"] for item in result["logs"]} == {"ERROR"}

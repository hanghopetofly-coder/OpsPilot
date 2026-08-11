"""Deterministic data builders shared by the local Monitor and CLS MCP tools.

The helpers in this module are deliberately free of I/O, global randomness and
wall-clock reads.  Supplying the same scenario and time window therefore always
produces byte-for-byte equivalent JSON data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

ScenarioName = Literal[
    "cpu_saturation",
    "memory_pressure",
    "database_timeout",
    "downstream_timeout",
    "conflicting_evidence",
    "tool_unavailable",
    "normal",
]
MetricKind = Literal["cpu", "memory"]

DEFAULT_SCENARIO: ScenarioName = "normal"
SUPPORTED_SCENARIOS: tuple[ScenarioName, ...] = (
    "cpu_saturation",
    "memory_pressure",
    "database_timeout",
    "downstream_timeout",
    "conflicting_evidence",
    "tool_unavailable",
    "normal",
)
DEFAULT_REFERENCE_TIME = datetime(2026, 2, 14, 11, 0, 0)
DEFAULT_WINDOW_MINUTES = 60
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_METRIC_POINTS = 288
MAX_LOG_LIMIT = 100
MAX_LOG_SAMPLES = 12

EXPECTED_ROOT_CAUSES: dict[ScenarioName, str | None] = {
    "cpu_saturation": "CPU saturation / exhausted worker capacity",
    "memory_pressure": "Memory pressure / allocation or GC pressure",
    "database_timeout": "Database query timeout / exhausted connection pool",
    "downstream_timeout": "Downstream service timeout / circuit breaker activation",
    "conflicting_evidence": "Unresolved: CPU metrics are normal while logs report CPU saturation",
    "tool_unavailable": "Unknown: observability source unavailable",
    "normal": None,
}


@dataclass(frozen=True)
class MetricWindow:
    """A validated inclusive metric window."""

    start: datetime
    end: datetime
    interval_minutes: int
    point_count: int


@dataclass(frozen=True)
class MetricProfile:
    """Start/end values used to synthesize a deterministic metric trend."""

    start: float
    end: float
    wobble: float = 0.0


@dataclass(frozen=True)
class LogTemplate:
    """A representative log message plus its deterministic occurrence count."""

    level: str
    message: str
    fingerprint: str
    count: int


class ScenarioToolUnavailable(ConnectionError):
    """Classifiable simulated transport failure for the Monitor MCP tools."""

    category = "connection"
    code = "MOCK_TOOL_UNAVAILABLE"
    retryable = True

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"{self.code}: category={self.category}; retryable=true; "
            f"mock observability source for {tool_name} is unavailable"
        )


_METRIC_PROFILES: dict[ScenarioName, dict[MetricKind, MetricProfile]] = {
    "normal": {
        "cpu": MetricProfile(24.0, 29.0, 1.0),
        "memory": MetricProfile(41.0, 44.0, 0.6),
    },
    "cpu_saturation": {
        "cpu": MetricProfile(68.0, 96.0, 1.2),
        "memory": MetricProfile(44.0, 51.0, 0.7),
    },
    "memory_pressure": {
        "cpu": MetricProfile(29.0, 35.0, 0.8),
        "memory": MetricProfile(63.0, 98.0, 1.0),
    },
    "database_timeout": {
        "cpu": MetricProfile(33.0, 38.0, 0.7),
        "memory": MetricProfile(45.0, 48.0, 0.5),
    },
    "downstream_timeout": {
        "cpu": MetricProfile(28.0, 36.0, 0.9),
        "memory": MetricProfile(41.0, 46.0, 0.5),
    },
    "conflicting_evidence": {
        "cpu": MetricProfile(27.0, 31.0, 0.7),
        "memory": MetricProfile(42.0, 46.0, 0.5),
    },
    # This profile is never emitted; the MCP tools return a classified error.
    "tool_unavailable": {
        "cpu": MetricProfile(0.0, 0.0),
        "memory": MetricProfile(0.0, 0.0),
    },
}

_LOG_TEMPLATES: dict[ScenarioName, tuple[LogTemplate, ...]] = {
    "normal": (
        LogTemplate("INFO", "request completed successfully", "request_ok", 36),
        LogTemplate("INFO", "health check passed", "health_ok", 12),
    ),
    "cpu_saturation": (
        LogTemplate(
            "ERROR",
            "worker pool exhausted while CPU remained saturated",
            "worker_pool_exhausted",
            24,
        ),
        LogTemplate("WARN", "request queue latency exceeded 2000ms", "queue_latency", 11),
        LogTemplate("INFO", "autoscaling evaluation requested", "autoscaling", 3),
    ),
    "memory_pressure": (
        LogTemplate(
            "ERROR",
            "OutOfMemoryError: allocation failed for request buffer",
            "out_of_memory",
            14,
        ),
        LogTemplate("WARN", "full GC pause exceeded 1200ms", "long_gc_pause", 19),
        LogTemplate("INFO", "heap occupancy crossed 85 percent", "heap_high", 7),
    ),
    "database_timeout": (
        LogTemplate(
            "ERROR",
            "database query timeout after 3000ms",
            "database_query_timeout",
            27,
        ),
        LogTemplate(
            "WARN",
            "database connection pool exhausted",
            "database_pool_exhausted",
            13,
        ),
        LogTemplate("INFO", "database retry scheduled", "database_retry", 6),
    ),
    "downstream_timeout": (
        LogTemplate(
            "ERROR",
            "downstream payment-service timeout after 2000ms",
            "payment_service_timeout",
            23,
        ),
        LogTemplate(
            "WARN",
            "circuit breaker opened for payment-service",
            "payment_circuit_open",
            9,
        ),
        LogTemplate("INFO", "fallback response returned", "payment_fallback", 5),
    ),
    "conflicting_evidence": (
        LogTemplate(
            "ERROR",
            "CPU saturation detected while workers experienced starvation",
            "cpu_worker_starvation",
            21,
        ),
        LogTemplate(
            "WARN",
            "CPU saturation caused the worker queue to stall",
            "cpu_queue_stall",
            9,
        ),
    ),
    "tool_unavailable": (),
}


def normalize_scenario(scenario: str | None) -> ScenarioName:
    """Return a canonical supported scenario or raise a precise validation error."""

    normalized = (scenario or DEFAULT_SCENARIO).strip().lower()
    if normalized not in SUPPORTED_SCENARIOS:
        supported = ", ".join(SUPPORTED_SCENARIOS)
        raise ValueError(f"unsupported scenario {scenario!r}; expected one of: {supported}")
    return normalized  # type: ignore[return-value]


def expected_root_cause(scenario: str | None) -> str | None:
    """Return the scenario's expected diagnosis without mutating any state."""

    return EXPECTED_ROOT_CAUSES[normalize_scenario(scenario)]


def parse_interval_minutes(interval: str) -> int:
    """Parse a strictly positive minute/hour interval."""

    if not isinstance(interval, str):
        raise ValueError("interval must be a string such as '1m', '5m', or '1h'")
    match = re.fullmatch(r"([1-9][0-9]*)([mh])", interval.strip().lower())
    if match is None:
        raise ValueError("interval must be a positive value ending in 'm' or 'h'")
    value = int(match.group(1))
    minutes = value if match.group(2) == "m" else value * 60
    if minutes > 24 * 60:
        raise ValueError("interval must not exceed 24h")
    return minutes


def _parse_time(value: str | None, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must use format {TIME_FORMAT}") from exc


def resolve_metric_window(
    start_time: str | None,
    end_time: str | None,
    interval: str,
    *,
    reference_time: datetime = DEFAULT_REFERENCE_TIME,
    max_points: int = MAX_METRIC_POINTS,
) -> MetricWindow:
    """Validate and resolve an inclusive fixed/caller-supplied metric window."""

    if max_points <= 0:
        raise ValueError("max_points must be positive")
    interval_minutes = parse_interval_minutes(interval)
    parsed_end = _parse_time(end_time, field_name="end_time")
    end = parsed_end or reference_time
    parsed_start = _parse_time(start_time, field_name="start_time")
    start = parsed_start or end - timedelta(minutes=DEFAULT_WINDOW_MINUTES)
    if start > end:
        raise ValueError("start_time must be less than or equal to end_time")

    duration_seconds = (end - start).total_seconds()
    point_count = math.floor(duration_seconds / (interval_minutes * 60)) + 1
    if point_count > max_points:
        raise ValueError(
            f"time window would generate {point_count} points; maximum is {max_points}"
        )
    return MetricWindow(
        start=start,
        end=end,
        interval_minutes=interval_minutes,
        point_count=point_count,
    )


def _metric_values(profile: MetricProfile, point_count: int) -> list[float]:
    wobble_pattern = (0.0, 0.7, -0.4, 1.0, -0.6, 0.3)
    values: list[float] = []
    for index in range(point_count):
        # A legal single-point fault window must still represent the requested
        # scenario, so sample the profile's terminal/peak state.
        progress = 1.0 if point_count == 1 else index / (point_count - 1)
        baseline = profile.start + (profile.end - profile.start) * progress
        wobble = wobble_pattern[index % len(wobble_pattern)] * profile.wobble
        values.append(round(max(0.0, min(100.0, baseline + wobble)), 1))
    return values


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return round(ordered[rank - 1], 2)


def _anomalous_intervals(
    timestamps: list[str],
    values: list[float],
    threshold: float,
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    start_index: int | None = None
    for index, value in enumerate([*values, float("-inf")]):
        if index < len(values) and value >= threshold:
            if start_index is None:
                start_index = index
            continue
        if start_index is None:
            continue
        interval_values = values[start_index:index]
        intervals.append(
            {
                "start": timestamps[start_index],
                "end": timestamps[index - 1],
                "points": len(interval_values),
                "max": max(interval_values),
            }
        )
        start_index = None
    return intervals


def build_metric_result(
    *,
    metric: MetricKind,
    scenario: str | None,
    service_name: str,
    window: MetricWindow,
    interval: str,
) -> dict[str, Any]:
    """Build deterministic CPU or memory output for a validated window."""

    scenario_name = normalize_scenario(scenario)
    if scenario_name == "tool_unavailable":
        return unavailable_result(
            tool_name=f"query_{metric}_metrics",
            scenario=scenario_name,
            context={
                "service_name": service_name,
                "metric_name": f"{metric}_usage_percent",
                "interval": interval,
            },
        )

    threshold = 80.0 if metric == "cpu" else 70.0
    profile = _METRIC_PROFILES[scenario_name][metric]
    values = _metric_values(profile, window.point_count)
    timestamps = [
        (window.start + timedelta(minutes=window.interval_minutes * index)).strftime(TIME_FORMAT)
        for index in range(window.point_count)
    ]
    data_points: list[dict[str, Any]] = []
    for timestamp, value in zip(timestamps, values, strict=True):
        point: dict[str, Any] = {"timestamp": timestamp, "value": value}
        if metric == "cpu":
            point["process_id"] = "pid-12345"
        else:
            point.update(used_gb=round(value * 8.0 / 100.0, 2), total_gb=8.0)
        data_points.append(point)

    change = round(values[-1] - values[0], 2)
    if change > 5.0:
        direction = "rising"
    elif change < -5.0:
        direction = "falling"
    else:
        direction = "stable"
    anomalies = _anomalous_intervals(timestamps, values, threshold)
    alert_triggered = bool(anomalies)
    statistics: dict[str, Any] = {
        "avg": round(sum(values) / len(values), 2),
        "max": max(values),
        "min": min(values),
        "p95": _nearest_rank_percentile(values, 0.95),
    }
    statistics["spike_detected" if metric == "cpu" else "memory_pressure"] = alert_triggered
    return {
        "success": True,
        "scenario": scenario_name,
        "service_name": service_name,
        "metric_name": f"{metric}_usage_percent",
        "start_time": window.start.strftime(TIME_FORMAT),
        "end_time": window.end.strftime(TIME_FORMAT),
        "interval": interval,
        "data_points": data_points,
        "statistics": statistics,
        "anomalous_intervals": anomalies,
        "trend": {
            "direction": direction,
            "change_percent_points": change,
            "first": values[0],
            "last": values[-1],
        },
        "threshold": threshold,
        "alert_info": {
            "triggered": alert_triggered,
            "threshold": threshold,
            "message": (
                f"{metric.upper()} usage crossed the {threshold:.0f}% threshold"
                if alert_triggered
                else f"{metric.upper()} usage remained below the {threshold:.0f}% threshold"
            ),
        },
    }


def validate_log_query(start_time: int, end_time: int, limit: int) -> None:
    """Validate the finite CLS query boundary without generating any rows."""

    if isinstance(start_time, bool) or not isinstance(start_time, int):
        raise ValueError("start_time must be an integer Unix timestamp in milliseconds")
    if isinstance(end_time, bool) or not isinstance(end_time, int):
        raise ValueError("end_time must be an integer Unix timestamp in milliseconds")
    if start_time > end_time:
        raise ValueError("start_time must be less than or equal to end_time")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LOG_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LOG_LIMIT}")


def _format_millis(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _matches_query(template: LogTemplate, query: str | None) -> bool:
    if not query or not query.strip():
        return True
    normalized = query.strip().lower()
    if normalized.startswith("level:"):
        return template.level.lower() == normalized.partition(":")[2].strip()
    if normalized.startswith("message:"):
        needle = normalized.partition(":")[2].strip()
        return needle in template.message.lower()
    return normalized in f"{template.level} {template.message} {template.fingerprint}".lower()


def build_log_result(
    *,
    scenario: str | None,
    service_name: str,
    topic_id: str,
    start_time: int,
    end_time: int,
    query: str | None,
    limit: int,
) -> dict[str, Any]:
    """Build sampled deterministic logs and aggregate repeated errors."""

    validate_log_query(start_time, end_time, limit)
    scenario_name = normalize_scenario(scenario)
    context = {
        "service_name": service_name,
        "topic_id": topic_id,
        "start_time": start_time,
        "end_time": end_time,
        "query": query,
        "limit": limit,
    }
    # ``tool_unavailable`` models a Monitor outage.  CLS remains available as
    # the alternative evidence source and emits a small, normal log sample.
    template_scenario = "normal" if scenario_name == "tool_unavailable" else scenario_name
    matching = [item for item in _LOG_TEMPLATES[template_scenario] if _matches_query(item, query)]
    sample_limit = min(limit, MAX_LOG_SAMPLES)
    sampled_templates = matching[:sample_limit]
    duration = max(0, end_time - start_time)
    logs: list[dict[str, Any]] = []
    for index, item in enumerate(sampled_templates):
        timestamp = start_time + duration * (index + 1) // (len(sampled_templates) + 1)
        logs.append(
            {
                "timestamp": _format_millis(timestamp),
                "level": item.level,
                "message": item.message,
                "fingerprint": item.fingerprint,
                "occurrence_count": item.count,
            }
        )

    level_counts: dict[str, int] = {}
    for item in matching:
        level_counts[item.level] = level_counts.get(item.level, 0) + item.count
    repeated_errors = [
        {
            "fingerprint": item.fingerprint,
            "message": item.message,
            "count": item.count,
            "first_seen": _format_millis(start_time),
            "last_seen": _format_millis(end_time),
        }
        for item in matching
        if item.level == "ERROR" and item.count > 1
    ]
    total_occurrences = sum(item.count for item in matching)
    return {
        "success": True,
        "scenario": scenario_name,
        **context,
        "total": total_occurrences,
        "returned": len(logs),
        "sampled": total_occurrences > len(logs),
        "logs": logs,
        "statistics": {
            "level_counts": level_counts,
            "repeated_errors": repeated_errors,
        },
        "took_ms": 1,
        "message": (
            f"sampled {len(logs)} representative logs from {total_occurrences} occurrences"
        ),
    }


def unavailable_result(
    *,
    tool_name: str,
    scenario: str | None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable protocol payload that downstream code can classify."""

    scenario_name = normalize_scenario(scenario)
    return {
        "success": False,
        "scenario": scenario_name,
        **(context or {}),
        "error": {
            "type": "tool_unavailable",
            "category": "connection",
            "code": "MOCK_TOOL_UNAVAILABLE",
            "tool_name": tool_name,
            "retryable": True,
            "message": f"mock observability source for {tool_name} is unavailable",
        },
    }


__all__ = [
    "DEFAULT_REFERENCE_TIME",
    "DEFAULT_SCENARIO",
    "EXPECTED_ROOT_CAUSES",
    "MAX_LOG_LIMIT",
    "MAX_LOG_SAMPLES",
    "MAX_METRIC_POINTS",
    "MetricWindow",
    "ScenarioToolUnavailable",
    "SUPPORTED_SCENARIOS",
    "build_log_result",
    "build_metric_result",
    "expected_root_cause",
    "normalize_scenario",
    "parse_interval_minutes",
    "resolve_metric_window",
    "unavailable_result",
    "validate_log_query",
]

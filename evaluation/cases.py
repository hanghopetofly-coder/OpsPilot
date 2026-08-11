"""Fixed offline diagnosis cases used by :mod:`evaluation.run`.

The catalog intentionally contains no random values and performs no I/O.  Raw
tool results mimic the bounded records produced by the Executor, allowing the
evaluation runner to exercise the real extractor, evaluator and reporter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.agent.aiops.models import DiagnosisHypothesis, RawToolResult

Scenario = Literal[
    "cpu_saturation",
    "memory_pressure",
    "database_timeout",
    "downstream_timeout",
    "normal",
    "conflicting_evidence",
    "tool_unavailable",
]
RootScenario = Literal[
    "cpu_saturation",
    "memory_pressure",
    "database_timeout",
    "downstream_timeout",
]

ROOT_CAUSES: dict[str, str] = {
    "cpu_saturation": "CPU saturation",
    "memory_pressure": "Memory pressure",
    "database_timeout": "Database connection timeout",
    "downstream_timeout": "Downstream dependency timeout",
}

EXPECTED_EVIDENCE: dict[str, str] = {
    "cpu_saturation": "CPU saturation",
    "memory_pressure": "memory pressure",
    "database_timeout": "database connection timeout",
    "downstream_timeout": "downstream dependency timeout",
}


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One reproducible simulated investigation and its expected outcome."""

    id: str
    scenario: Scenario
    expected_root_cause: str | None
    hypotheses: tuple[DiagnosisHypothesis, ...]
    raw_results: tuple[RawToolResult, ...]
    replans: int = 0
    baseline: str | None = None

    @property
    def tool_calls(self) -> int:
        return len(self.raw_results)

    @property
    def steps(self) -> int:
        return len({item.step_id for item in self.raw_results})


def _hypotheses(primary: RootScenario) -> tuple[DiagnosisHypothesis, ...]:
    ordered = [primary, *[item for item in ROOT_CAUSES if item != primary]][:3]
    return tuple(
        DiagnosisHypothesis(
            id=f"H{index}",
            cause=ROOT_CAUSES[name],
            description=f"Offline candidate for {name}",
            expected_evidence=[EXPECTED_EVIDENCE[name]],
        )
        for index, name in enumerate(ordered, 1)
    )


def _raw(
    case_id: str,
    index: int,
    *,
    hypothesis_id: str,
    tool_name: str,
    source: str,
    goal: str,
    expected: str,
    payload: object = None,
    content: str = "",
    is_error: bool = False,
    error_type: str | None = None,
) -> RawToolResult:
    return RawToolResult(
        id=f"{case_id}-R{index}",
        step_id=f"S{index}",
        hypothesis_ids=[hypothesis_id],
        tool_call_id=f"{case_id}-call-{index}",
        tool_name=tool_name,
        source=source,
        diagnostic_goal=goal,
        expected_evidence=[expected],
        arguments={"service_name": "checkout-service", "case_id": case_id},
        content=content,
        payload=payload,  # type: ignore[arg-type]
        is_error=is_error,
        error_type=error_type,
        duration_ms=10.0 + index,
    )


def _metric(
    case_id: str,
    index: int,
    *,
    hypothesis_id: str,
    metric_name: str,
    expected: str,
    maximum: float,
    threshold: float,
    triggered: bool,
) -> RawToolResult:
    return _raw(
        case_id,
        index,
        hypothesis_id=hypothesis_id,
        tool_name=("query_memory_metrics" if "memory" in metric_name else "query_cpu_metrics"),
        source="monitor",
        goal=f"verify {expected}",
        expected=expected,
        payload={
            "service_name": "checkout-service",
            "metric_name": metric_name,
            "interval": "10m",
            "statistics": {
                "avg": round(maximum * 0.82, 2),
                "max": maximum,
                "min": round(maximum * 0.45, 2),
                "p95": round(maximum * 0.96, 2),
            },
            "alert_info": {"triggered": triggered, "threshold": threshold},
        },
    )


def _log(
    case_id: str,
    index: int,
    *,
    hypothesis_id: str,
    expected: str,
    message: str,
    level: str = "ERROR",
    tool_name: str = "search_log",
) -> RawToolResult:
    return _raw(
        case_id,
        index,
        hypothesis_id=hypothesis_id,
        tool_name=tool_name,
        source="logs",
        goal=f"verify {expected}",
        expected=expected,
        payload={
            "total": 3,
            "query": expected,
            "logs": [
                {
                    "timestamp": f"2026-08-09T10:0{index}:00Z",
                    "level": level,
                    "message": message,
                }
            ],
        },
    )


def _alert(
    case_id: str,
    index: int,
    *,
    hypothesis_id: str,
    expected: str,
    alert_name: str,
) -> RawToolResult:
    return _raw(
        case_id,
        index,
        hypothesis_id=hypothesis_id,
        tool_name="query_prometheus_alerts",
        source="prometheus",
        goal=f"verify {expected}",
        expected=expected,
        payload={
            "alerts": [
                {
                    "labels": {
                        "alertname": alert_name,
                        "service": "checkout-service",
                        "severity": "critical",
                    },
                    "annotations": {"summary": expected},
                    "state": "firing",
                }
            ]
        },
    )


def _positive_results(
    case_id: str,
    root: RootScenario,
    variant: int,
) -> tuple[RawToolResult, ...]:
    expected = EXPECTED_EVIDENCE[root]
    if root == "cpu_saturation":
        return (
            _metric(
                case_id,
                1,
                hypothesis_id="H1",
                metric_name="cpu_usage_percent",
                expected=expected,
                maximum=94.0 + variant,
                threshold=80.0,
                triggered=True,
            ),
            _log(
                case_id,
                2,
                hypothesis_id="H1",
                expected=expected,
                message="CPU saturation error caused worker starvation",
            ),
        )
    if root == "memory_pressure":
        return (
            _metric(
                case_id,
                1,
                hypothesis_id="H1",
                metric_name="memory_usage_percent",
                expected=expected,
                maximum=92.0 + variant,
                threshold=70.0,
                triggered=True,
            ),
            _log(
                case_id,
                2,
                hypothesis_id="H1",
                expected=expected,
                message="memory pressure caused out of memory OOM errors",
            ),
        )
    if root == "database_timeout":
        return (
            _alert(
                case_id,
                1,
                hypothesis_id="H1",
                expected=expected,
                alert_name="DatabaseConnectionTimeout",
            ),
            _log(
                case_id,
                2,
                hypothesis_id="H1",
                expected=expected,
                message="database connection timeout error acquiring connection pool",
            ),
        )
    return (
        _alert(
            case_id,
            1,
            hypothesis_id="H1",
            expected=expected,
            alert_name="DownstreamDependencyTimeout",
        ),
        _log(
            case_id,
            2,
            hypothesis_id="H1",
            expected=expected,
            message="downstream dependency timeout error calling payment service",
        ),
    )


def _root_case(case_id: str, root: RootScenario, variant: int) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        scenario=root,
        expected_root_cause=ROOT_CAUSES[root],
        hypotheses=_hypotheses(root),
        raw_results=_positive_results(case_id, root, variant),
    )


def _normal_case(case_id: str, variant: int) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        scenario="normal",
        expected_root_cause=None,
        hypotheses=_hypotheses("cpu_saturation"),
        raw_results=(
            _metric(
                case_id,
                1,
                hypothesis_id="H1",
                metric_name="cpu_usage_percent",
                expected=EXPECTED_EVIDENCE["cpu_saturation"],
                maximum=35.0 + variant,
                threshold=80.0,
                triggered=False,
            ),
            _metric(
                case_id,
                2,
                hypothesis_id="H2",
                metric_name="memory_usage_percent",
                expected=EXPECTED_EVIDENCE["memory_pressure"],
                maximum=42.0 + variant,
                threshold=70.0,
                triggered=False,
            ),
        ),
    )


def _conflict_case(case_id: str, root: RootScenario) -> EvaluationCase:
    expected = EXPECTED_EVIDENCE[root]
    if root == "cpu_saturation":
        raw_results = (
            _metric(
                case_id,
                1,
                hypothesis_id="H1",
                metric_name="cpu_usage_percent",
                expected=expected,
                maximum=97.0,
                threshold=80.0,
                triggered=True,
            ),
            _log(
                case_id,
                2,
                hypothesis_id="H1",
                expected=expected,
                message="CPU utilization stable and healthy",
                level="INFO",
            ),
        )
    elif root == "memory_pressure":
        raw_results = (
            _metric(
                case_id,
                1,
                hypothesis_id="H1",
                metric_name="memory_usage_percent",
                expected=expected,
                maximum=96.0,
                threshold=70.0,
                triggered=True,
            ),
            _log(
                case_id,
                2,
                hypothesis_id="H1",
                expected=expected,
                message="memory pressure check healthy and stable",
                level="INFO",
            ),
        )
    else:
        raw_results = (
            _log(
                case_id,
                1,
                hypothesis_id="H1",
                expected=expected,
                message="database connection timeout error acquiring connection pool",
            ),
            _raw(
                case_id,
                2,
                hypothesis_id="H1",
                tool_name="query_prometheus_alerts",
                source="prometheus",
                goal="verify database alert health for connection timeout",
                expected=expected,
                payload={"alerts": []},
            ),
        )
    return EvaluationCase(
        id=case_id,
        scenario="conflicting_evidence",
        expected_root_cause=None,
        hypotheses=_hypotheses(root),
        raw_results=raw_results,
        replans=1,
    )


def _failed_tool(
    case_id: str,
    root: RootScenario,
    tool_name: str,
    source: str,
) -> RawToolResult:
    return _raw(
        case_id,
        1,
        hypothesis_id="H1",
        tool_name=tool_name,
        source=source,
        goal=f"verify {EXPECTED_EVIDENCE[root]}",
        expected=EXPECTED_EVIDENCE[root],
        content="tool request timed out while collecting runtime evidence",
        is_error=True,
        error_type="timeout",
    )


def _tool_case(case_id: str, root: RootScenario, *, recovered: bool) -> EvaluationCase:
    if root == "database_timeout":
        failed_tool_name, failed_source = "search_log", "logs"
    elif root == "cpu_saturation":
        failed_tool_name, failed_source = "query_cpu_metrics", "monitor"
    else:
        failed_tool_name, failed_source = "query_memory_metrics", "monitor"
    failed = _failed_tool(
        case_id,
        root,
        failed_tool_name,
        failed_source,
    )
    raw_results: tuple[RawToolResult, ...]
    expected_root_cause: str | None
    if recovered:
        expected = EXPECTED_EVIDENCE[root]
        if root == "database_timeout":
            fallbacks = (
                _alert(
                    case_id,
                    2,
                    hypothesis_id="H1",
                    expected=expected,
                    alert_name="DatabaseConnectionTimeout",
                ),
                _log(
                    case_id,
                    3,
                    hypothesis_id="H1",
                    expected=expected,
                    message="database connection timeout error from replica log",
                    tool_name="search_replica_log",
                ),
            )
        else:
            fallbacks = (
                _alert(
                    case_id,
                    2,
                    hypothesis_id="H1",
                    expected=expected,
                    alert_name="CPUSaturation",
                ),
                _log(
                    case_id,
                    3,
                    hypothesis_id="H1",
                    expected=expected,
                    message="CPU saturation error caused worker starvation",
                ),
            )
        raw_results = (failed, *fallbacks)
        expected_root_cause = ROOT_CAUSES[root]
    else:
        raw_results = (failed,)
        expected_root_cause = None
    return EvaluationCase(
        id=case_id,
        scenario="tool_unavailable",
        expected_root_cause=expected_root_cause,
        hypotheses=_hypotheses(root),
        raw_results=raw_results,
        replans=1,
    )


CASES: tuple[EvaluationCase, ...] = (
    _root_case("cpu-01", "cpu_saturation", 0),
    _root_case("cpu-02", "cpu_saturation", 1),
    _root_case("cpu-03", "cpu_saturation", 2),
    _root_case("memory-01", "memory_pressure", 0),
    _root_case("memory-02", "memory_pressure", 1),
    _root_case("memory-03", "memory_pressure", 2),
    _root_case("database-01", "database_timeout", 0),
    _root_case("database-02", "database_timeout", 1),
    _root_case("database-03", "database_timeout", 2),
    _root_case("downstream-01", "downstream_timeout", 0),
    _root_case("downstream-02", "downstream_timeout", 1),
    _root_case("downstream-03", "downstream_timeout", 2),
    _normal_case("normal-01", 0),
    _normal_case("normal-02", 3),
    _conflict_case("conflict-cpu", "cpu_saturation"),
    _conflict_case("conflict-memory", "memory_pressure"),
    _conflict_case("conflict-database", "database_timeout"),
    _tool_case("tool-cpu-recovered", "cpu_saturation", recovered=True),
    _tool_case("tool-database-recovered", "database_timeout", recovered=True),
    _tool_case("tool-memory-unavailable", "memory_pressure", recovered=False),
)

if len(CASES) != 20:  # pragma: no cover - import-time catalog invariant
    raise RuntimeError(f"offline evaluation catalog must contain 20 cases, got {len(CASES)}")


__all__ = [
    "CASES",
    "EXPECTED_EVIDENCE",
    "EvaluationCase",
    "ROOT_CAUSES",
    "RootScenario",
    "Scenario",
]

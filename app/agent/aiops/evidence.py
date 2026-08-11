"""Bounded evidence extraction and deterministic diagnosis evaluation.

The extractor is intentionally non-LLM code: raw tool payloads are normalized
once, converted into concise observations, and never forwarded to later
reasoning nodes.  The evaluator is likewise deterministic so routing does not
depend on an unconstrained natural-language sufficiency judgement.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

from loguru import logger
from pydantic import ValidationError

from app.config import config

from .models import (
    DiagnosisEvaluation,
    DiagnosisEvidence,
    DiagnosisHypothesis,
    DiagnosisStep,
    EvidenceSource,
    RawToolResult,
    ToolError,
)
from .state import PlanExecuteState

DEFAULT_MAX_EVIDENCE_ITEMS = 8
DEFAULT_MAX_LOG_RECORDS = 100
DEFAULT_MAX_OBSERVATION_CHARS = 1200
SUPPORTED_CONFIDENCE_THRESHOLD = 0.70

_WHITESPACE_RE = re.compile(r"\s+")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-f]{10,}\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d{3,}\b")
_NORMALIZE_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
_ABNORMAL_LOG_RE = re.compile(
    r"timeout|timed out|error|exception|oom|out of memory|fail(?:ed|ure)?|"
    r"refused|unavailable|deadlock|超时|错误|异常|失败|拒绝|不可用|死锁",
    re.IGNORECASE,
)

# Relations are deliberately conservative: a generic active alert or a generic
# ERROR line is a symptom, not proof of whichever hypothesis happened to own the
# step.  At least one diagnosis-specific concept must overlap between the
# step's declared purpose and the observation before support/contradiction is
# attached.
_CONCEPT_MARKERS: dict[str, tuple[str, ...]] = {
    "cpu": ("cpu", "processor", "load average", "线程", "处理器", "负载"),
    "memory": ("memory", "heap", "oom", "out of memory", "gc", "内存", "堆"),
    "database": (
        "database",
        " db ",
        "sql",
        "jdbc",
        "connection pool",
        "数据库",
    ),
    "connection": ("connection", "connect", "连接"),
    "pool": ("connection pool", "pool exhaustion", "pool saturation", "连接池"),
    "backup": ("backup", "snapshot", "restore", "备份", "快照", "恢复"),
    "migration": ("migration", "schema change", "schema", "迁移", "模式变更"),
    "query": ("query", "slow sql", "sql statement", "查询", "慢 SQL", "慢sql"),
    "dependency": (
        "downstream",
        "upstream",
        "dependency",
        "dependent service",
        "下游",
        "上游",
        "依赖",
    ),
    "timeout": ("timeout", "timed out", "超时"),
    "latency": ("latency", "slow", "p95", "p99", "延迟", "耗时"),
    "network": ("network", "dns", "tcp", "packet loss", "网络", "丢包"),
    "deployment": ("deploy", "release", "rollback", "change", "部署", "发布", "变更"),
    "disk": ("disk", "filesystem", "i/o", "io wait", "磁盘", "文件系统"),
    "oom": ("oom", "out of memory", "内存溢出"),
    "gc": ("garbage collection", "full gc", "gc pause", "垃圾回收"),
}
_CONCEPT_SPLIT_RE = re.compile(r"[、,，;/；]|\b(?:and|or)\b|以及|及", re.IGNORECASE)
_TOO_BROAD_SINGLE_CONCEPTS = {"database", "connection", "timeout"}


def _bounded_text(value: object, max_chars: int) -> str:
    text = value if isinstance(value, str) else str(value)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) <= max_chars:
        return text
    marker = " …[truncated]"
    return text[: max(1, max_chars - len(marker))].rstrip() + marker


def _json_preview(value: object, max_chars: int) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    return _bounded_text(rendered, max_chars)


def _parse_text_payload(text: str) -> object:
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    # Some MCP adapters stringify Python dictionaries.  literal_eval is safe
    # for literals and lets the extractor normalize those legacy responses.
    if stripped[0] in "[{(" and stripped[-1] in "]})":
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (dict, list, tuple, str, int, float, bool)):
                return list(parsed) if isinstance(parsed, tuple) else parsed
        except (SyntaxError, ValueError, TypeError, MemoryError):
            pass
    return stripped


def _unwrap_payload(raw: RawToolResult) -> object:
    payload: object = raw.payload if raw.payload is not None else raw.content
    if isinstance(payload, str):
        payload = _parse_text_payload(payload)

    # ToolMessage content is sometimes a list of content blocks.  Unwrap a
    # single text block, but retain multi-block payloads as bounded generic data.
    if isinstance(payload, list) and len(payload) == 1:
        first = payload[0]
        if isinstance(first, Mapping):
            text = first.get("text")
            if isinstance(text, str):
                payload = _parse_text_payload(text)
    if isinstance(payload, Mapping):
        content = payload.get("content")
        if len(payload) == 1 and isinstance(content, str):
            payload = _parse_text_payload(content)
    return payload


def _normalized_source(raw: RawToolResult) -> EvidenceSource:
    source = raw.source.lower()
    tool_name = raw.tool_name.lower()
    combined = f"{source} {tool_name}"
    if "knowledge" in combined or "retrieve" in combined or "rag" in combined:
        return "knowledge_base"
    if "log" in combined or "cls" in combined:
        return "logs"
    if "prometheus" in combined or "alert" in tool_name or "promql" in combined:
        return "prometheus"
    if any(token in combined for token in ("cpu", "memory", "metric", "monitor")):
        return "monitor"
    return "generic"


def _diagnostic_purpose(raw: RawToolResult) -> str:
    return " ".join([raw.diagnostic_goal or "", *raw.expected_evidence]).lower()


def _concepts(value: str) -> set[str]:
    padded = f" {value.lower()} "
    return {
        concept
        for concept, markers in _CONCEPT_MARKERS.items()
        if any(marker in padded for marker in markers)
    }


def _purpose_requirements(raw: RawToolResult) -> list[set[str]]:
    """Extract concrete concept combinations promised by the diagnosis step."""

    requirements: list[set[str]] = []
    sources = list(raw.expected_evidence)
    if not any(_concepts(item) for item in sources):
        sources = [raw.diagnostic_goal or ""]
    for source in sources:
        for fragment in _CONCEPT_SPLIT_RE.split(source):
            concepts = _concepts(fragment)
            if concepts and concepts not in requirements:
                requirements.append(concepts)
    return requirements


def _is_diagnostically_relevant(raw: RawToolResult, observation: str) -> bool:
    """Require an expected concept combination, not one broad shared word."""

    observation_concepts = _concepts(observation)
    for requirement in _purpose_requirements(raw):
        if len(requirement) == 1 and requirement <= _TOO_BROAD_SINGLE_CONCEPTS:
            continue
        if requirement <= observation_concepts:
            return True
    return False


def _expects_normal_metric(raw: RawToolResult, metric_name: str) -> bool:
    """Detect steps that use a normal local metric to support an alternative cause."""
    purpose = _diagnostic_purpose(raw)
    metric = metric_name.lower()
    resource_markers = ("本地资源正常", "local resource normal", "资源水位正常")
    if "cpu" in metric:
        metric_markers = ("cpu正常", "normal cpu", "cpu normal")
    elif "memory" in metric or "内存" in metric:
        metric_markers = ("内存正常", "normal memory", "memory normal")
    else:
        metric_markers = ("指标正常", "normal metric", "metric normal")
    return any(marker in purpose for marker in (*resource_markers, *metric_markers))


def _payload_error(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    if payload.get("success") is False:
        return str(payload.get("error") or payload.get("message") or "tool returned success=false")
    status = str(payload.get("status", "")).lower()
    if status in {"error", "failed", "failure"}:
        return str(payload.get("error") or payload.get("message") or status)
    error = payload.get("error")
    if error not in (None, "", False, [], {}):
        return str(error)
    return None


def _evidence_id(raw: RawToolResult, index: int) -> str:
    return f"E-{raw.id}-{index:02d}"


def _make_evidence(
    raw: RawToolResult,
    *,
    index: int,
    source: EvidenceSource,
    observation: str,
    supports: Sequence[str] = (),
    contradicts: Sequence[str] = (),
    reliability: float | None,
    max_chars: int,
) -> DiagnosisEvidence:
    return DiagnosisEvidence(
        id=_evidence_id(raw, index),
        source=source,
        tool_name=raw.tool_name,
        hypothesis_ids=list(raw.hypothesis_ids),
        observation=_bounded_text(observation, max_chars),
        supports=list(supports),
        contradicts=list(contradicts),
        reliability=reliability,
        raw_result_ref=raw.id,
    )


def _nested_alerts(payload: Mapping[str, Any]) -> list[object] | None:
    alerts = payload.get("alerts")
    if isinstance(alerts, list):
        return alerts
    data = payload.get("data")
    if isinstance(data, Mapping) and isinstance(data.get("alerts"), list):
        return list(data["alerts"])
    return None


def _extract_prometheus_alerts(
    raw: RawToolResult,
    payload: object,
    *,
    max_items: int,
    max_chars: int,
) -> list[DiagnosisEvidence]:
    if not isinstance(payload, Mapping):
        return []
    alerts = _nested_alerts(payload)
    if alerts is None:
        return []
    if not alerts:
        purpose = _diagnostic_purpose(raw)
        expects_alert = "alert" in purpose or "告警" in purpose
        return [
            _make_evidence(
                raw,
                index=1,
                source="prometheus",
                observation="Prometheus returned no active alerts for the requested scope.",
                contradicts=raw.hypothesis_ids if expects_alert else (),
                reliability=0.90,
                max_chars=max_chars,
            )
        ]

    evidence: list[DiagnosisEvidence] = []
    total = len(alerts)
    for index, item in enumerate(alerts[:max_items], 1):
        if not isinstance(item, Mapping):
            continue
        raw_labels = item.get("labels")
        raw_annotations = item.get("annotations")
        labels = cast(Mapping[str, Any], raw_labels) if isinstance(raw_labels, Mapping) else {}
        annotations = (
            cast(Mapping[str, Any], raw_annotations) if isinstance(raw_annotations, Mapping) else {}
        )
        name = item.get("alert_name") or labels.get("alertname") or "unnamed alert"
        state = item.get("state") or item.get("status") or "active"
        service = (
            labels.get("service")
            or labels.get("job")
            or labels.get("pod")
            or labels.get("instance")
            or "unknown scope"
        )
        severity = labels.get("severity") or "unspecified"
        duration = item.get("duration") or item.get("active_at") or item.get("activeAt")
        summary = (
            item.get("summary")
            or item.get("description")
            or annotations.get("summary")
            or annotations.get("description")
            or ""
        )
        observation = (
            f"Prometheus alert {name!s} is {state!s}; scope={service!s}; " f"severity={severity!s}"
        )
        if duration:
            observation += f"; active_since_or_duration={duration!s}"
        if summary:
            observation += f"; summary={summary!s}"
        if total > max_items and index == min(total, max_items):
            observation += f"; {total - max_items} additional alerts omitted"
        relevant = _is_diagnostically_relevant(raw, observation)
        state_text = str(state).strip().lower()
        is_active = state_text in {"active", "firing", "pending"}
        evidence.append(
            _make_evidence(
                raw,
                index=index,
                source="prometheus",
                observation=observation,
                supports=raw.hypothesis_ids if relevant and is_active else (),
                contradicts=raw.hypothesis_ids if relevant and not is_active else (),
                # An alert is a strong symptom but does not, by itself, prove
                # the step's candidate root cause.
                reliability=0.75,
                max_chars=max_chars,
            )
        )
    return evidence


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _metric_values(payload: Mapping[str, Any]) -> list[float]:
    points = payload.get("data_points") or payload.get("values") or payload.get("samples")
    if not isinstance(points, list):
        return []
    values: list[float] = []
    for point in points[:1000]:
        candidate = point.get("value") if isinstance(point, Mapping) else point
        parsed = _number(candidate)
        if parsed is not None:
            values.append(parsed)
    return values


def _extract_metrics(
    raw: RawToolResult,
    payload: object,
    *,
    source: EvidenceSource,
    max_chars: int,
) -> list[DiagnosisEvidence]:
    if not isinstance(payload, Mapping):
        return []
    statistics = payload.get("statistics") or payload.get("stats") or payload.get("summary")
    if not isinstance(statistics, Mapping):
        statistics = {}
    values = _metric_values(payload)

    avg_value = _number(statistics.get("avg") or statistics.get("average"))
    max_value = _number(statistics.get("max") or statistics.get("maximum"))
    min_value = _number(statistics.get("min") or statistics.get("minimum"))
    p95_value = _number(statistics.get("p95"))
    if values:
        avg_value = avg_value if avg_value is not None else sum(values) / len(values)
        max_value = max_value if max_value is not None else max(values)
        min_value = min_value if min_value is not None else min(values)
    if all(value is None for value in (avg_value, max_value, min_value, p95_value)):
        return []

    metric_name = str(payload.get("metric_name") or raw.tool_name)
    service = str(payload.get("service_name") or raw.arguments.get("service_name") or "unknown")
    interval = payload.get("interval") or payload.get("time_range") or "unspecified"
    alert_info = payload.get("alert_info")
    if not isinstance(alert_info, Mapping):
        alert_info = {}
    threshold = _number(alert_info.get("threshold"))
    lower_name = metric_name.lower()
    if threshold is None:
        threshold = 70.0 if "memory" in lower_name else 80.0

    triggered_value = alert_info.get("triggered")
    if isinstance(triggered_value, bool):
        triggered = triggered_value
    elif isinstance(statistics.get("spike_detected"), bool):
        triggered = bool(statistics["spike_detected"])
    elif isinstance(statistics.get("memory_pressure"), bool):
        triggered = bool(statistics["memory_pressure"])
    else:
        triggered = max_value is not None and max_value > threshold

    components = [
        f"metric={metric_name}",
        f"service={service}",
        f"interval={interval}",
    ]
    for label, value in (
        ("avg", avg_value),
        ("max", max_value),
        ("min", min_value),
        ("p95", p95_value),
    ):
        if value is not None:
            components.append(f"{label}={value:.2f}")
    components.extend(
        (f"threshold={threshold:.2f}", f"threshold_triggered={str(triggered).lower()}")
    )
    if alert_info.get("message"):
        components.append(f"summary={alert_info['message']!s}")
    observation = "Monitor observation: " + "; ".join(components)
    expects_normal = _expects_normal_metric(raw, metric_name)
    relevant = expects_normal or _is_diagnostically_relevant(raw, metric_name)
    supports = raw.hypothesis_ids if relevant and triggered != expects_normal else ()
    contradicts = raw.hypothesis_ids if relevant and triggered == expects_normal else ()
    return [
        _make_evidence(
            raw,
            index=1,
            source=source,
            observation=observation,
            supports=supports,
            contradicts=contradicts,
            reliability=0.95 if statistics else 0.85,
            max_chars=max_chars,
        )
    ]


def _normalize_log_message(message: str) -> str:
    normalized = _UUID_RE.sub("<uuid>", message)
    normalized = _HEX_RE.sub("<hex>", normalized)
    normalized = _NUMBER_RE.sub("<n>", normalized)
    return _bounded_text(normalized, 240)


def _extract_logs(
    raw: RawToolResult,
    payload: object,
    *,
    max_log_records: int,
    max_chars: int,
) -> list[DiagnosisEvidence]:
    if not isinstance(payload, Mapping):
        return []
    logs = payload.get("logs") or payload.get("entries") or payload.get("results")
    if logs is None:
        return []
    if not isinstance(logs, list):
        return []
    total = int(_number(payload.get("total")) or len(logs))
    if not logs:
        query = payload.get("query") or raw.arguments.get("query") or "requested filter"
        return [
            _make_evidence(
                raw,
                index=1,
                source="logs",
                observation=f"Log query returned no matching entries; query={query!s}.",
                contradicts=raw.hypothesis_ids,
                reliability=0.80,
                max_chars=max_chars,
            )
        ]

    sampled = logs[:max_log_records]
    levels: Counter[str] = Counter()
    messages: Counter[str] = Counter()
    timestamps: list[str] = []
    relevant_abnormal_count = 0
    relevant_normal_count = 0
    for item in sampled:
        if isinstance(item, Mapping):
            level = str(item.get("level") or item.get("severity") or "unknown").upper()
            message = str(item.get("message") or item.get("log") or item.get("content") or "")
            timestamp = item.get("timestamp") or item.get("time")
            if timestamp:
                timestamps.append(str(timestamp))
        else:
            level = "UNKNOWN"
            message = str(item)
        levels[level] += 1
        if message:
            normalized_message = _normalize_log_message(message)
            messages[normalized_message] += 1
            is_relevant = _is_diagnostically_relevant(raw, normalized_message)
            is_abnormal = level in {
                "WARN",
                "WARNING",
                "ERROR",
                "CRITICAL",
                "FATAL",
            } or bool(_ABNORMAL_LOG_RE.search(normalized_message))
            if is_relevant and is_abnormal:
                relevant_abnormal_count += 1
            elif is_relevant:
                relevant_normal_count += 1

    top_messages = (
        ", ".join(f"{message!r} x{count}" for message, count in messages.most_common(5))
        or "no message text"
    )
    level_summary = ", ".join(f"{level}={count}" for level, count in sorted(levels.items()))
    observation = (
        f"Log query returned total={total}, sampled={len(sampled)}; "
        f"levels=[{level_summary}]; repeated_patterns=[{top_messages}]"
    )
    if timestamps:
        observation += f"; sampled_time_range={min(timestamps)}..{max(timestamps)}"
    if len(logs) > max_log_records:
        observation += f"; {len(logs) - max_log_records} in-payload records omitted"
    abnormal_levels = sum(
        count
        for level, count in levels.items()
        if level in {"WARN", "WARNING", "ERROR", "CRITICAL", "FATAL"}
    )
    abnormal_patterns = sum(
        count for message, count in messages.items() if _ABNORMAL_LOG_RE.search(message)
    )
    observation += (
        f"; abnormal_level_count={abnormal_levels}; "
        f"abnormal_pattern_count={abnormal_patterns}; "
        f"relevant_abnormal_count={relevant_abnormal_count}; "
        f"relevant_normal_count={relevant_normal_count}"
    )
    return [
        _make_evidence(
            raw,
            index=1,
            source="logs",
            observation=observation,
            supports=raw.hypothesis_ids if relevant_abnormal_count > 0 else (),
            contradicts=(
                raw.hypothesis_ids
                if relevant_abnormal_count == 0 and relevant_normal_count > 0
                else ()
            ),
            reliability=0.85,
            max_chars=max_chars,
        )
    ]


def _knowledge_text(payload: object, raw: RawToolResult) -> str:
    if isinstance(payload, Mapping):
        for key in ("context", "content", "text", "answer", "documents", "docs"):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                return value if isinstance(value, str) else _json_preview(value, 100_000)
    if isinstance(payload, str):
        return payload
    if payload not in (None, "", [], {}):
        return _json_preview(payload, 100_000)
    return raw.content


def _extract_knowledge(
    raw: RawToolResult,
    payload: object,
    *,
    max_chars: int,
) -> list[DiagnosisEvidence]:
    text = _knowledge_text(payload, raw).strip()
    if not text:
        return []
    # Knowledge can guide investigation, but it is deliberately not entered in
    # ``supports``: a runbook cannot by itself establish an online incident fact.
    observation = "Knowledge-base guidance (not an online observation): " + text
    return [
        _make_evidence(
            raw,
            index=1,
            source="knowledge_base",
            observation=observation,
            reliability=0.60,
            max_chars=max_chars,
        )
    ]


def _extract_generic(
    raw: RawToolResult,
    payload: object,
    *,
    max_chars: int,
) -> list[DiagnosisEvidence]:
    rendered = payload if isinstance(payload, str) else _json_preview(payload, max_chars)
    rendered = rendered.strip()
    if not rendered:
        return []
    return [
        _make_evidence(
            raw,
            index=1,
            source="generic",
            observation=f"Tool observation: {rendered}",
            supports=(),
            contradicts=(),
            reliability=0.50,
            max_chars=max_chars,
        )
    ]


def extract_evidence(
    raw_result: RawToolResult | Mapping[str, Any],
    *,
    max_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
    max_chars: int = DEFAULT_MAX_OBSERVATION_CHARS,
    max_log_records: int = DEFAULT_MAX_LOG_RECORDS,
) -> list[DiagnosisEvidence]:
    """Convert one successful raw result into bounded structured evidence.

    Errors intentionally yield no evidence; the LangGraph node records a
    classified :class:`ToolError` instead.  Returned runtime-tool evidence
    always contains a non-empty ``raw_result_ref``.
    """

    if max_items <= 0 or max_chars <= 0 or max_log_records <= 0:
        raise ValueError("evidence bounds must be positive")
    raw = (
        raw_result
        if isinstance(raw_result, RawToolResult)
        else RawToolResult.model_validate(raw_result)
    )
    payload = _unwrap_payload(raw)
    if raw.is_error or _payload_error(payload):
        return []

    source = _normalized_source(raw)
    tool_name = raw.tool_name.lower()
    if isinstance(payload, Mapping) and _nested_alerts(payload) is not None:
        return _extract_prometheus_alerts(raw, payload, max_items=max_items, max_chars=max_chars)
    if "alert" in tool_name and source == "prometheus":
        return _extract_prometheus_alerts(raw, payload, max_items=max_items, max_chars=max_chars)
    if any(token in tool_name for token in ("cpu", "memory", "metric")) or (
        isinstance(payload, Mapping) and "metric_name" in payload
    ):
        return _extract_metrics(raw, payload, source=source, max_chars=max_chars)
    if source == "logs" or (
        isinstance(payload, Mapping) and any(key in payload for key in ("logs", "entries"))
    ):
        return _extract_logs(
            raw,
            payload,
            max_log_records=max_log_records,
            max_chars=max_chars,
        )
    if source == "knowledge_base":
        return _extract_knowledge(raw, payload, max_chars=max_chars)
    return _extract_generic(raw, payload, max_chars=max_chars)


def _classify_error(message: str, explicit_type: str | None = None) -> tuple[str, bool]:
    value = f"{explicit_type or ''} {message}".lower()
    if "timeout" in value or "timed out" in value or "超时" in value:
        return "timeout", True
    if any(token in value for token in ("connection", "connect", "network", "连接", "网络")):
        return "connection", True
    if any(token in value for token in ("unavailable", "not found", "不存在", "不可用")):
        return "unavailable", True
    if "invalid" in value or "parse" in value or "无效" in value or "解析" in value:
        return "invalid_result", False
    return explicit_type or "tool_error", False


def _alternative_source(source: EvidenceSource) -> str | None:
    return {
        "monitor": "logs_or_prometheus",
        "prometheus": "monitor_or_logs",
        "logs": "monitor",
        "knowledge_base": None,
        "generic": None,
    }[source]


def tool_error_from_raw_result(
    raw_result: RawToolResult | Mapping[str, Any],
    *,
    message: str | None = None,
    category: str | None = None,
) -> ToolError:
    """Create a deterministic classified error for an unusable raw result."""

    raw = (
        raw_result
        if isinstance(raw_result, RawToolResult)
        else RawToolResult.model_validate(raw_result)
    )
    payload = _unwrap_payload(raw)
    error_message = (
        message or _payload_error(payload) or raw.content or "tool returned no extractable evidence"
    )
    classified, retryable = _classify_error(error_message, category or raw.error_type)
    source = _normalized_source(raw)
    alternative = _alternative_source(source)
    return ToolError(
        id=f"TE-{raw.id}",
        step_id=raw.step_id,
        hypothesis_id=raw.hypothesis_ids[0] if raw.hypothesis_ids else None,
        tool_name=raw.tool_name,
        category=classified,
        message=_bounded_text(error_message, DEFAULT_MAX_OBSERVATION_CHARS),
        retryable=retryable,
        alternative_source=alternative,
        handled=False,
    )


def _error_exists(errors: Sequence[object], candidate: ToolError) -> bool:
    for item in errors:
        if not isinstance(item, Mapping):
            continue
        if item.get("id") == candidate.id:
            return True
        if (
            item.get("step_id") == candidate.step_id
            and item.get("tool_name") == candidate.tool_name
            and item.get("handled") is not True
        ):
            return True
    return False


def evidence_extractor(state: PlanExecuteState) -> dict[str, Any]:
    """LangGraph node that processes every currently unprocessed raw result.

    Existing state lists are replaced rather than reducer-appended, making the
    node idempotent under checkpoint replay.  A raw item is considered processed
    when its flag is set or an Evidence already references its id.
    """

    raw_items = list(state.get("raw_tool_results", []))
    existing_evidence = list(state.get("evidence", []))
    errors = list(state.get("tool_errors", []))
    referenced_raw_ids = {
        item.get("raw_result_ref")
        for item in existing_evidence
        if isinstance(item, Mapping) and item.get("raw_result_ref")
    }
    evidence_ids = {
        item.get("id") for item in existing_evidence if isinstance(item, Mapping) and item.get("id")
    }
    updated_raw_items = list(raw_items)

    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        raw_id = str(item.get("id") or f"invalid-{index + 1}")
        if item.get("evidence_extracted") is True or raw_id in referenced_raw_ids:
            continue
        try:
            raw = RawToolResult.model_validate(item)
        except ValidationError as exc:
            malformed = dict(item)
            malformed["evidence_extracted"] = True
            updated_raw_items[index] = malformed
            candidate = ToolError(
                id=f"TE-{raw_id}",
                step_id=str(item.get("step_id") or "unknown-step"),
                hypothesis_id=(
                    str(item.get("hypothesis_ids", [""])[0]).strip() or None
                    if isinstance(item.get("hypothesis_ids"), list) and item.get("hypothesis_ids")
                    else None
                ),
                tool_name=str(item.get("tool_name") or "unknown-tool"),
                category="invalid_result",
                message=_bounded_text(f"invalid raw tool result: {exc}", 800),
                retryable=False,
                handled=False,
            )
            if not _error_exists(errors, candidate):
                errors.append(candidate.model_dump(mode="json"))
            continue

        payload = _unwrap_payload(raw)
        error_message = _payload_error(payload)
        extracted: list[DiagnosisEvidence] = []
        if not raw.is_error and error_message is None:
            extracted = extract_evidence(
                raw,
                max_chars=config.aiops_evidence_max_chars,
            )
        if raw.is_error or error_message is not None or not extracted:
            candidate = tool_error_from_raw_result(
                raw,
                message=error_message,
                category="invalid_result" if not extracted and not raw.is_error else None,
            )
            if not _error_exists(errors, candidate):
                errors.append(candidate.model_dump(mode="json"))
        for evidence_item in extracted:
            if evidence_item.id not in evidence_ids:
                existing_evidence.append(evidence_item.model_dump(mode="json"))
                evidence_ids.add(evidence_item.id)
                logger.info(
                    "diagnosis_id={} step_id={} hypothesis_ids={} evidence_id={} "
                    "source={} supports={} contradicts={}",
                    state.get("diagnosis_id", "unknown"),
                    raw.step_id,
                    raw.hypothesis_ids,
                    evidence_item.id,
                    evidence_item.source,
                    evidence_item.supports,
                    evidence_item.contradicts,
                )

        updated_raw_items[index] = raw.model_copy(update={"evidence_extracted": True}).model_dump(
            mode="json"
        )

    return {
        "raw_tool_results": updated_raw_items,
        "evidence": existing_evidence,
        "tool_errors": errors,
    }


def _combined_reliability(items: Sequence[DiagnosisEvidence]) -> float:
    if not items:
        return 0.0
    remainder = 1.0
    for item in items:
        reliability = item.reliability if item.reliability is not None else 0.70
        remainder *= 1.0 - reliability
    return max(0.0, min(1.0, 1.0 - remainder))


def apply_evaluation_to_hypotheses(
    hypotheses: Sequence[DiagnosisHypothesis | Mapping[str, Any]],
    evidence: Sequence[DiagnosisEvidence | Mapping[str, Any]],
) -> list[DiagnosisHypothesis]:
    """Return hypothesis copies with deterministic status, confidence and links."""

    hypothesis_models = [
        item if isinstance(item, DiagnosisHypothesis) else DiagnosisHypothesis.model_validate(item)
        for item in hypotheses
    ]
    evidence_models = [
        item if isinstance(item, DiagnosisEvidence) else DiagnosisEvidence.model_validate(item)
        for item in evidence
    ]
    updated: list[DiagnosisHypothesis] = []
    for hypothesis in hypothesis_models:
        all_support = [item for item in evidence_models if hypothesis.id in item.supports]
        online_support = [item for item in all_support if item.source != "knowledge_base"]
        contradictions = [
            item
            for item in evidence_models
            if hypothesis.id in item.contradicts and item.source != "knowledge_base"
        ]
        support_strength = _combined_reliability(online_support)
        contradiction_strength = _combined_reliability(contradictions)

        if online_support and contradictions:
            status = "uncertain"
            confidence = support_strength * (1.0 - contradiction_strength)
        elif online_support:
            status = "supported"
            confidence = support_strength
        elif contradictions:
            status = "contradicted"
            confidence = 0.0
        else:
            status = "uncertain"
            confidence = 0.0
        updated.append(
            hypothesis.model_copy(
                update={
                    "supporting_evidence_ids": [item.id for item in all_support],
                    "contradicting_evidence_ids": [item.id for item in contradictions],
                    "confidence": round(max(0.0, min(1.0, confidence)), 4),
                    "status": status,
                }
            )
        )
    return updated


def _normalized(value: str) -> str:
    return _NORMALIZE_RE.sub("", value).lower()


def _planned_expected_evidence(
    remaining_plan: Sequence[DiagnosisStep | Mapping[str, Any]],
) -> dict[str, list[str]]:
    planned: dict[str, list[str]] = {}
    for item in remaining_plan:
        try:
            step = item if isinstance(item, DiagnosisStep) else DiagnosisStep.model_validate(item)
        except ValidationError:
            continue
        planned.setdefault(step.hypothesis_id, []).extend(step.expected_evidence)
        planned[step.hypothesis_id].append(step.goal)
    return planned


def _missing_evidence(
    hypothesis: DiagnosisHypothesis,
    evidence: Sequence[DiagnosisEvidence],
) -> list[str]:
    relevant = [
        item
        for item in evidence
        if hypothesis.id in (set(item.supports) | set(item.contradicts))
        and item.source != "knowledge_base"
    ]
    corpus = [_normalized(item.observation) for item in relevant]
    unmatched_evidence_slots = len(relevant)
    missing: list[str] = []
    for expected in hypothesis.expected_evidence:
        normalized_expected = _normalized(expected)
        covered_by_observation = bool(normalized_expected) and any(
            normalized_expected in observation or observation in normalized_expected
            for observation in corpus
            if observation
        )
        if covered_by_observation:
            continue
        # A structured support/contradiction binding is stronger than a fragile
        # language substring match.  Allocate each observation to at most one
        # otherwise-unmatched expected item.
        if unmatched_evidence_slots > 0:
            unmatched_evidence_slots -= 1
            continue
        missing.append(f"{hypothesis.id}: {expected}")
    return missing


def _is_scheduled(missing_item: str, planned: Mapping[str, Sequence[str]]) -> bool:
    hypothesis_id, _, expected = missing_item.partition(":")
    expected_normalized = _normalized(expected)
    if not expected_normalized:
        return False
    return any(
        expected_normalized in _normalized(item) or _normalized(item) in expected_normalized
        for item in planned.get(hypothesis_id.strip(), ())
        if _normalized(item)
    )


def evaluate_evidence(
    hypotheses: Sequence[DiagnosisHypothesis | Mapping[str, Any]],
    evidence: Sequence[DiagnosisEvidence | Mapping[str, Any]],
    tool_errors: Sequence[ToolError | Mapping[str, Any]] = (),
    *,
    remaining_budget: int,
    remaining_plan: Sequence[DiagnosisStep | Mapping[str, Any]] = (),
) -> DiagnosisEvaluation:
    """Deterministically decide whether to execute, replan or finish.

    Knowledge-base evidence is excluded from online support strength.  It may
    explain an investigation direction, but can never make a hypothesis
    ``supported`` or make the diagnosis finish on its own.
    """

    updated = apply_evaluation_to_hypotheses(hypotheses, evidence)
    evidence_models = [
        item if isinstance(item, DiagnosisEvidence) else DiagnosisEvidence.model_validate(item)
        for item in evidence
    ]
    error_models = [
        item if isinstance(item, ToolError) else ToolError.model_validate(item)
        for item in tool_errors
    ]
    planned = _planned_expected_evidence(remaining_plan)

    supported = [item.id for item in updated if item.status == "supported"]
    contradicted = [item.id for item in updated if item.status == "contradicted"]
    uncertain = [item.id for item in updated if item.status in {"pending", "uncertain"}]
    conflicts: list[str] = []
    all_missing: list[str] = []
    for hypothesis in updated:
        online_support = [
            item.id
            for item in evidence_models
            if hypothesis.id in item.supports and item.source != "knowledge_base"
        ]
        online_contradictions = [
            item.id
            for item in evidence_models
            if hypothesis.id in item.contradicts and item.source != "knowledge_base"
        ]
        if online_support and online_contradictions:
            conflicts.append(
                f"{hypothesis.id}: supporting {online_support} conflicts with "
                f"contradicting {online_contradictions}"
            )
        all_missing.extend(_missing_evidence(hypothesis, evidence_models))

    scheduled = [item for item in all_missing if _is_scheduled(item, planned)]
    missing = [item for item in all_missing if item not in scheduled]

    if not supported and not all_missing and updated:
        missing.append("diagnosis: no hypothesis has supporting online evidence")

    budget = max(0, int(remaining_budget))
    actionable_errors = [
        item
        for item in error_models
        if not item.handled and (item.retryable or item.alternative_source is not None)
    ]
    tool_failures = [f"{item.id}: {item.tool_name} ({item.category})" for item in actionable_errors]
    confidence = max((item.confidence for item in updated), default=0.0)
    high_confidence_support = False
    for hypothesis in updated:
        supporting = [
            item
            for item in evidence_models
            if hypothesis.id in item.supports and item.source != "knowledge_base"
        ]
        sources = {item.source for item in supporting}
        if (
            hypothesis.status == "supported"
            and hypothesis.confidence >= SUPPORTED_CONFIDENCE_THRESHOLD
            and len(supporting) >= 2
            and len(sources) >= 2
        ):
            high_confidence_support = True
            break
    budget_exhausted = budget <= 0

    if budget_exhausted:
        can_finish = True
        need_replan = False
        reason = "Investigation budget exhausted; produce an evidence-grounded partial diagnosis."
    elif conflicts:
        can_finish = False
        need_replan = True
        reason = "Supporting and contradicting online evidence conflict; revise the investigation."
    elif high_confidence_support:
        can_finish = True
        need_replan = False
        reason = "At least one hypothesis has sufficient reliable online evidence."
    elif tool_failures:
        can_finish = False
        need_replan = True
        reason = "An actionable tool failure needs a bounded retry or alternative source."
    elif missing and remaining_plan:
        # Do not throw away still-valid initial steps merely because an earlier
        # hypothesis remains incomplete.  Finish the bounded plan first; any
        # evidence still missing when the plan empties will trigger replanning.
        can_finish = False
        need_replan = False
        reason = "Key evidence is still missing; continue the existing bounded plan first."
    elif missing:
        can_finish = False
        need_replan = True
        reason = "The current plan is exhausted while key online evidence is still missing."
    elif error_models and not remaining_plan:
        # Non-retryable failures without alternatives should not cause an
        # endless investigation.  Ranking will explicitly retain uncertainty.
        can_finish = True
        need_replan = False
        reason = "Only non-recoverable tool failures remain; produce a partial diagnosis."
    else:
        can_finish = False
        need_replan = False
        reason = "Current evidence is inconclusive; continue the existing bounded plan."

    return DiagnosisEvaluation(
        supported_hypotheses=supported,
        contradicted_hypotheses=contradicted,
        uncertain_hypotheses=uncertain,
        missing_evidence=missing,
        scheduled_evidence=scheduled,
        evidence_conflicts=conflicts,
        tool_failures=tool_failures,
        diagnosis_confidence=round(confidence, 4),
        need_replan=need_replan,
        can_finish=can_finish,
        budget_exhausted=budget_exhausted,
        reason=reason,
    )


def evidence_evaluator(state: PlanExecuteState) -> dict[str, Any]:
    """LangGraph node that updates hypotheses and stores strict evaluation."""

    hypotheses = list(state.get("hypotheses", []))
    evidence = list(state.get("evidence", []))
    errors = list(state.get("tool_errors", []))
    plan = list(state.get("plan", []))
    remaining_budget = int(state.get("remaining_budget", 0) or 0)
    updated_hypotheses = apply_evaluation_to_hypotheses(hypotheses, evidence)
    evaluation = evaluate_evidence(
        updated_hypotheses,
        evidence,
        errors,
        remaining_budget=remaining_budget,
        remaining_plan=plan,
    )
    logger.info(
        "diagnosis_id={} evaluator confidence={} supported={} contradicted={} "
        "missing={} conflicts={} failures={} need_replan={} can_finish={} "
        "budget_exhausted={}",
        state.get("diagnosis_id", "unknown"),
        evaluation.diagnosis_confidence,
        evaluation.supported_hypotheses,
        evaluation.contradicted_hypotheses,
        len(evaluation.missing_evidence),
        len(evaluation.evidence_conflicts),
        len(evaluation.tool_failures),
        evaluation.need_replan,
        evaluation.can_finish,
        evaluation.budget_exhausted,
    )
    return {
        "hypotheses": [item.model_dump(mode="json") for item in updated_hypotheses],
        "evaluation": evaluation.model_dump(mode="json"),
    }


# Explicit node aliases make graph assembly readable while retaining the short
# names as the public pure-ish state transformation entry points.
evidence_extractor_node = evidence_extractor
evidence_evaluator_node = evidence_evaluator


__all__ = [
    "DEFAULT_MAX_EVIDENCE_ITEMS",
    "DEFAULT_MAX_LOG_RECORDS",
    "DEFAULT_MAX_OBSERVATION_CHARS",
    "SUPPORTED_CONFIDENCE_THRESHOLD",
    "apply_evaluation_to_hypotheses",
    "evaluate_evidence",
    "evidence_evaluator",
    "evidence_evaluator_node",
    "evidence_extractor",
    "evidence_extractor_node",
    "extract_evidence",
    "tool_error_from_raw_result",
]

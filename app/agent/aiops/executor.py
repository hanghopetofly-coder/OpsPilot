"""Execute one hypothesis-bound diagnosis step and retain bounded raw results."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger
from pydantic import JsonValue, SecretStr

from app.config import config

from .models import (
    DiagnosisStep,
    ExecutionRecord,
    ExecutionStatus,
    RawToolResult,
    ToolError,
)
from .state import PlanExecuteState
from .utils import load_available_tools

EXECUTOR_SYSTEM_PROMPT = """你是故障诊断工作流中的工具执行器。

当前步骤已经绑定一个候选故障假设。你必须调用最合适的真实工具来获取该步骤要求的证据：
- 工具调用的唯一目的，是验证或反驳当前 hypothesis；
- 优先遵守 step 中的 tool_hint 和 expected_evidence；
- 可参考已有的结构化 Evidence 来填写服务、主题、时间范围等参数；
- 不要重复调用已经得到同类证据的工具；
- 不要用自然语言编造观测结果，也不要自行下根因结论；
- 每个调用都必须能解释为“为什么这条证据能验证当前假设”。

如果没有合适工具，不要伪造结果；直接说明无法调用工具。"""


TRUNCATION_MARKER = "...[truncated]"


def _json_safe(value: Any) -> Any:
    """Convert tool values to checkpoint-friendly JSON data."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return _json_text(_json_safe(content))


def _bounded_text(content: Any, max_chars: int | None = None) -> tuple[str, bool]:
    """Return text whose total length, including its marker, fits the limit."""

    text = _content_to_text(content)
    limit = config.aiops_raw_result_max_chars if max_chars is None else max(0, max_chars)
    if len(text) <= limit:
        return text, False
    if limit <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:limit], True
    prefix_chars = limit - len(TRUNCATION_MARKER)
    return f"{text[:prefix_chars]}{TRUNCATION_MARKER}", True


def _bounded_json_value(value: Any, max_chars: int) -> tuple[Any, bool]:
    """Return JSON-safe data with a serialized representation under ``max_chars``."""

    safe_value = _json_safe(value)
    rendered = _json_text(safe_value)
    if len(rendered) <= max_chars:
        return safe_value, False

    def preview_container(prefix_chars: int) -> dict[str, Any]:
        suffix = TRUNCATION_MARKER if prefix_chars < len(rendered) else ""
        return {
            "_truncated": True,
            "preview": f"{rendered[:prefix_chars]}{suffix}",
        }

    low = 0
    high = len(rendered)
    best: dict[str, Any] | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate = preview_container(middle)
        if len(_json_text(candidate)) <= max_chars:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1

    if best is not None:
        return best, True
    # The configured production limit is much larger than this minimal shape;
    # this defensive fallback also keeps tiny unit-test limits JSON-safe.
    return None, True


def _bounded_arguments(value: Any, max_chars: int) -> tuple[dict[str, Any], bool]:
    safe_value = _json_safe(value)
    arguments = safe_value if isinstance(safe_value, dict) else {"value": safe_value}
    bounded, truncated = _bounded_json_value(arguments, max_chars)
    if isinstance(bounded, dict):
        return bounded, truncated
    return {}, True


def _bounded_tool_result_parts(
    *,
    arguments: Any,
    content: Any,
    payload: Any,
) -> tuple[dict[str, Any], str, Any, bool]:
    """Bound all raw-data fields under one shared character budget."""

    limit = config.aiops_raw_result_max_chars
    safe_arguments = _json_safe(arguments)
    if not isinstance(safe_arguments, dict):
        safe_arguments = {"value": safe_arguments}
    safe_payload = _json_safe(payload) if payload is not None else None
    content_text = _content_to_text(content)
    payload_size = len(_json_text(safe_payload)) if safe_payload is not None else 0
    total_size = len(_json_text(safe_arguments)) + len(content_text) + payload_size
    if total_size <= limit:
        return safe_arguments, content_text, safe_payload, False

    # Arguments and artifacts each receive at most one quarter.  The remaining
    # shared budget is reserved for content, which is the extractor's primary
    # input.  Serialized argument/payload sizes are counted, so neither can
    # bypass the raw-result limit with nested JSON.
    arguments_budget = max(2, limit // 4)
    bounded_arguments, arguments_truncated = _bounded_arguments(
        safe_arguments,
        arguments_budget,
    )
    if safe_payload is None:
        bounded_payload = None
        payload_truncated = False
        bounded_payload_size = 0
    else:
        payload_budget = max(4, limit // 4)
        bounded_payload, payload_truncated = _bounded_json_value(
            safe_payload,
            payload_budget,
        )
        bounded_payload_size = len(_json_text(bounded_payload))

    used_chars = len(_json_text(bounded_arguments)) + bounded_payload_size
    bounded_content, content_truncated = _bounded_text(
        content_text,
        max(0, limit - used_chars),
    )
    return (
        bounded_arguments,
        bounded_content,
        bounded_payload,
        arguments_truncated or content_truncated or payload_truncated,
    )


def _source_for_tool(tool_name: str) -> str:
    name = tool_name.lower()
    if "prometheus" in name or "alert" in name:
        return "prometheus"
    if "log" in name or "cls" in name or "topic" in name:
        return "logs"
    if "cpu" in name or "memory" in name or "metric" in name or "monitor" in name:
        return "monitor"
    if "knowledge" in name or "document" in name:
        return "knowledge_base"
    return "tool"


def _tool_accepts_argument(tool: Any, argument_name: str) -> bool:
    args = getattr(tool, "args", None)
    if isinstance(args, Mapping) and argument_name in args:
        return True
    schema = getattr(tool, "args_schema", None)
    try:
        if schema is not None and hasattr(schema, "model_json_schema"):
            payload = schema.model_json_schema()
        elif schema is not None and hasattr(schema, "schema"):
            payload = schema.schema()
        else:
            payload = schema
    except Exception:
        return False
    return (
        isinstance(payload, Mapping)
        and isinstance(payload.get("properties"), Mapping)
        and argument_name in payload["properties"]
    )


def _inject_scenario(
    tool_calls: Sequence[Mapping[str, Any]],
    tools: Sequence[Any],
    scenario: str | None,
) -> list[dict[str, Any]]:
    """Make the request scenario authoritative for tools that declare it."""

    tools_by_name = {str(getattr(tool, "name", "")): tool for tool in tools}
    effective_scenario = (scenario or "normal").strip() or "normal"
    injected: list[dict[str, Any]] = []
    for raw_call in tool_calls:
        call = dict(raw_call)
        tool = tools_by_name.get(str(call.get("name", "")))
        args = call.get("args")
        if tool is not None and _tool_accepts_argument(tool, "scenario"):
            bounded_args = dict(args) if isinstance(args, Mapping) else {}
            bounded_args["scenario"] = effective_scenario
            call["args"] = bounded_args
        injected.append(call)
    return injected


def _tool_node_timeout_seconds() -> float:
    """Allow the MCP interceptor's bounded attempts without an outer race."""

    attempts = config.mcp_tool_max_attempts
    backoff_seconds = sum(
        config.mcp_tool_retry_delay_seconds * (2**attempt)
        for attempt in range(max(0, attempts - 1))
    )
    return float(config.aiops_tool_timeout_seconds * attempts + backoff_seconds + 1.0)


def _looks_like_error(content: str, status: Any) -> bool:
    if str(status).lower() == "error":
        return True
    lowered = content.lower()
    error_markers = (
        '"success": false',
        '"iserror": true',
        "执行失败",
        "调用失败",
        "发生错误",
        "timed out",
        "timeout",
        "connection refused",
    )
    return any(marker in lowered for marker in error_markers)


def _tool_error(
    *,
    error_id: str,
    step: DiagnosisStep,
    tool_name: str,
    category: str,
    message: str,
    retryable: bool,
    alternative_source: str | None = None,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        ToolError(
            id=error_id,
            step_id=step.id,
            hypothesis_id=step.hypothesis_id,
            tool_name=tool_name,
            category=category,
            message=message[: config.aiops_evidence_max_chars],
            retryable=retryable,
            alternative_source=alternative_source,
            handled=False,
        ).model_dump(mode="json"),
    )


def _compact_evidence(state: PlanExecuteState) -> list[dict[str, Any]]:
    """Only bounded structured evidence, never raw payloads, enters selection context."""
    compact: list[dict[str, Any]] = []
    used_chars = 0
    for item in reversed(state.get("evidence", [])[-12:]):
        selected = {
            "id": item.get("id"),
            "source": item.get("source"),
            "hypothesis_ids": item.get("hypothesis_ids", []),
            "observation": str(item.get("observation", ""))[: config.aiops_evidence_max_chars],
            "supports": item.get("supports", []),
            "contradicts": item.get("contradicts", []),
        }
        size = len(json.dumps(selected, ensure_ascii=False, default=str))
        if compact and used_chars + size > config.aiops_tool_schema_max_chars:
            break
        compact.append(selected)
        used_chars += size
    compact.reverse()
    return compact


def _record_for_skipped_step(
    state: PlanExecuteState,
    step: DiagnosisStep,
    error: dict[str, Any],
    duration_ms: int = 0,
) -> dict[str, Any]:
    history = list(state.get("execution_history", []))
    history.append(
        ExecutionRecord(
            step_id=step.id,
            hypothesis_id=step.hypothesis_id,
            goal=step.goal,
            status="failed",
            tool_calls=[],
            raw_result_ids=[],
            duration_ms=duration_ms,
            summary=error["message"],
        ).model_dump(mode="json")
    )
    errors = list(state.get("tool_errors", []))
    errors.append(error)
    step_count = int(state.get("step_count", 0)) + 1
    tool_call_count = int(state.get("tool_call_count", 0))
    return {
        "plan": list(state.get("plan", []))[1:],
        "execution_history": history,
        "tool_errors": errors,
        "step_count": step_count,
        "remaining_budget": max(0, config.aiops_max_tool_calls - tool_call_count),
    }


async def executor(state: PlanExecuteState) -> dict[str, Any]:
    """Execute exactly one planned step; evidence extraction is a separate node."""
    plan = list(state.get("plan", []))
    if not plan:
        logger.info("diagnosis_id={} executor skipped: empty plan", state.get("diagnosis_id"))
        return {}

    step = DiagnosisStep.model_validate(plan[0])
    diagnosis_id = state.get("diagnosis_id", "unknown")
    tool_call_count = int(state.get("tool_call_count", 0))
    remaining_calls = max(0, config.aiops_max_tool_calls - tool_call_count)
    logger.info(
        "diagnosis_id={} step_id={} hypothesis_id={} executor start remaining_tool_calls={}",
        diagnosis_id,
        step.id,
        step.hypothesis_id,
        remaining_calls,
    )

    if remaining_calls == 0:
        error = _tool_error(
            error_id=f"TE{len(state.get('tool_errors', [])) + 1:03d}",
            step=step,
            tool_name="budget",
            category="budget_exhausted",
            message="工具调用预算已耗尽，当前步骤未执行。",
            retryable=False,
        )
        update = _record_for_skipped_step(state, step, error)
        update["termination_reason"] = "tool_call_budget_exhausted"
        return update

    started = time.perf_counter()
    tools, discovery_error = await load_available_tools()
    if not tools:
        message = discovery_error or "没有可用工具"
        error = _tool_error(
            error_id=f"TE{len(state.get('tool_errors', [])) + 1:03d}",
            step=step,
            tool_name="mcp_discovery",
            category="connection",
            message=message,
            retryable=True,
            alternative_source="local_tools",
        )
        return _record_for_skipped_step(
            state, step, error, int((time.perf_counter() - started) * 1000)
        )

    context = {
        "diagnosis_id": diagnosis_id,
        "query": state.get("input", ""),
        "alert_context": state.get("alert_context", {}),
        "scenario": state.get("scenario"),
        "step": step.model_dump(mode="json"),
        "collected_evidence": _compact_evidence(state),
        "mcp_discovery_warning": discovery_error,
        "remaining_tool_calls": remaining_calls,
    }
    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(context, ensure_ascii=False, default=str)),
    ]

    tool_calls: list[dict[str, Any]] = []
    attempted_tool_calls = 0
    try:
        llm = ChatQwen(
            model=config.rag_model,
            api_key=SecretStr(config.dashscope_api_key),
            base_url=config.dashscope_api_base,
            temperature=0,
        )
        llm_with_tools = llm.bind_tools(tools)
        selection = await asyncio.wait_for(
            llm_with_tools.ainvoke(messages),
            timeout=config.aiops_tool_timeout_seconds,
        )
        selected_calls = list(getattr(selection, "tool_calls", []) or [])[:remaining_calls]
        tool_calls = _inject_scenario(
            selected_calls,
            tools,
            state.get("scenario"),
        )
        if not tool_calls:
            error = _tool_error(
                error_id=f"TE{len(state.get('tool_errors', [])) + 1:03d}",
                step=step,
                tool_name="tool_selection",
                category="invalid_tool_result",
                message="模型未为证据收集步骤选择工具；自由文本未被当作诊断证据。",
                retryable=True,
            )
            return _record_for_skipped_step(
                state, step, error, int((time.perf_counter() - started) * 1000)
            )

        selected_message = AIMessage(
            content=getattr(selection, "content", ""), tool_calls=tool_calls
        )
        tool_node = ToolNode(tools)
        attempted_tool_calls = len(tool_calls)
        tool_output = await asyncio.wait_for(
            tool_node.ainvoke({"messages": [*messages, selected_message]}),
            timeout=_tool_node_timeout_seconds(),
        )
        tool_messages = list(tool_output.get("messages", []))
    except TimeoutError:
        names = [str(call.get("name", "unknown")) for call in tool_calls]
        tool_name = ",".join(names) or "tool_execution"
        error = _tool_error(
            error_id=f"TE{len(state.get('tool_errors', [])) + 1:03d}",
            step=step,
            tool_name=tool_name,
            category="timeout",
            message=f"工具选择或执行超过 {config.aiops_tool_timeout_seconds:.1f}s。",
            retryable=True,
            alternative_source="another_observability_source",
        )
        update = _record_for_skipped_step(
            state, step, error, int((time.perf_counter() - started) * 1000)
        )
        update["tool_call_count"] = tool_call_count + attempted_tool_calls
        update["remaining_budget"] = max(0, config.aiops_max_tool_calls - update["tool_call_count"])
        return update
    except Exception as exc:
        error = _tool_error(
            error_id=f"TE{len(state.get('tool_errors', [])) + 1:03d}",
            step=step,
            tool_name="tool_selection_or_execution",
            category="connection",
            message=f"工具选择或执行失败: {type(exc).__name__}: {exc}",
            retryable=True,
            alternative_source="another_observability_source",
        )
        update = _record_for_skipped_step(
            state, step, error, int((time.perf_counter() - started) * 1000)
        )
        update["tool_call_count"] = tool_call_count + attempted_tool_calls
        update["remaining_budget"] = max(
            0,
            config.aiops_max_tool_calls - update["tool_call_count"],
        )
        return update

    call_by_id = {str(call.get("id", "")): call for call in tool_calls}
    raw_results = list(state.get("raw_tool_results", []))
    errors = list(state.get("tool_errors", []))
    raw_ids: list[str] = []
    tool_names: list[str | dict[str, JsonValue]] = []

    for message in tool_messages:
        call_id = str(getattr(message, "tool_call_id", "") or "")
        call = call_by_id.get(call_id, {})
        tool_name = str(getattr(message, "name", None) or call.get("name", "unknown"))
        arguments, content, payload, truncated = _bounded_tool_result_parts(
            arguments=call.get("args", {}),
            content=getattr(message, "content", ""),
            payload=getattr(message, "artifact", None),
        )
        is_error = _looks_like_error(content, getattr(message, "status", None))
        raw_id = f"R{len(raw_results) + 1:03d}"
        raw = RawToolResult(
            id=raw_id,
            step_id=step.id,
            hypothesis_ids=[step.hypothesis_id],
            tool_call_id=call_id or None,
            tool_name=tool_name,
            source=_source_for_tool(tool_name),
            diagnostic_goal=step.goal,
            expected_evidence=list(step.expected_evidence),
            arguments=arguments,
            content=content,
            payload=payload,
            is_error=is_error,
            error_type="tool_error" if is_error else None,
            truncated=truncated,
            duration_ms=int((time.perf_counter() - started) * 1000),
        ).model_dump(mode="json")
        raw_results.append(raw)
        raw_ids.append(raw_id)
        tool_names.append(tool_name)
        if is_error:
            errors.append(
                _tool_error(
                    error_id=f"TE{len(errors) + 1:03d}",
                    step=step,
                    tool_name=tool_name,
                    category="tool_error",
                    message=content,
                    retryable=True,
                    alternative_source="another_observability_source",
                )
            )
        logger.info(
            "diagnosis_id={} step_id={} hypothesis_id={} tool={} raw_result_id={} "
            "duration_ms={} is_error={}",
            diagnosis_id,
            step.id,
            step.hypothesis_id,
            tool_name,
            raw_id,
            raw["duration_ms"],
            is_error,
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    status: ExecutionStatus = (
        "failed"
        if not raw_ids or all(item["is_error"] for item in raw_results[-len(raw_ids) :])
        else "success"
    )
    history = list(state.get("execution_history", []))
    history.append(
        ExecutionRecord(
            step_id=step.id,
            hypothesis_id=step.hypothesis_id,
            goal=step.goal,
            status=status,
            tool_calls=tool_names,
            raw_result_ids=raw_ids,
            duration_ms=duration_ms,
            summary="等待 Evidence Extractor 归一化工具结果",
        ).model_dump(mode="json")
    )
    new_tool_count = tool_call_count + attempted_tool_calls
    return {
        "plan": plan[1:],
        "execution_history": history,
        "raw_tool_results": raw_results,
        "tool_errors": errors,
        "step_count": int(state.get("step_count", 0)) + 1,
        "tool_call_count": new_tool_count,
        "remaining_budget": max(0, config.aiops_max_tool_calls - new_tool_count),
    }

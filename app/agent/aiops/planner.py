"""Hypothesis-driven Planner node for evidence-grounded AIOps diagnosis."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from langchain_core.documents import Document
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import SecretStr

from app.config import config
from app.tools import retrieve_knowledge

from .models import (
    DiagnosisEvidence,
    DiagnosisHypothesis,
    DiagnosisPlan,
    DiagnosisStep,
)
from .prompts import PLANNER_PROMPT
from .state import PlanExecuteState
from .utils import format_tools_description, load_available_tools

MAX_RETRIEVAL_QUERY_CHARS = 4_000
MAX_DIAGNOSIS_REQUEST_CHARS = 12_000
MAX_KNOWLEDGE_ITEMS = 5
MAX_KNOWLEDGE_ITEM_CHARS = 1_200
MAX_KNOWLEDGE_TOTAL_CHARS = 5_000
MAX_KNOWLEDGE_CONTEXT_CHARS = 6_000
MAX_KNOWLEDGE_EVIDENCE_CHARS = 700

_NO_KNOWLEDGE_MESSAGES = ("没有找到相关信息", "未检索到相关")
_KNOWLEDGE_ERROR_MESSAGES = ("检索知识时发生错误", "知识检索工具调用失败")


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    """Return bounded text and whether truncation occurred."""
    value = value.strip()
    if limit <= 0:
        return "", bool(value)
    if len(value) <= limit:
        return value, False
    marker = "…[truncated]"
    if limit <= len(marker):
        return marker[:limit], True
    return f"{value[: limit - len(marker)]}{marker}", True


def _message_text(result: Any) -> str:
    """Extract a readable content string from a direct tool result/ToolMessage."""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _safe_metadata(document: Document) -> dict[str, Any]:
    """Keep only small, useful and JSON-compatible knowledge metadata."""
    metadata = document.metadata or {}
    allowed_keys = ("_source", "_file_name", "_extension", "h1", "h2", "h3")
    safe: dict[str, Any] = {}
    for key in allowed_keys:
        value = metadata.get(key)
        if value is not None:
            safe[key] = str(value)
    return safe


def _documents_to_knowledge(documents: list[Document]) -> list[dict[str, Any]]:
    """Convert LangChain documents into bounded, serializable state records."""
    requested_top_k = max(1, int(config.rag_top_k))
    item_limit = min(requested_top_k, MAX_KNOWLEDGE_ITEMS)
    remaining_chars = min(MAX_KNOWLEDGE_TOTAL_CHARS, config.aiops_knowledge_max_chars)
    records: list[dict[str, Any]] = []

    for index, document in enumerate(documents[:item_limit], 1):
        if remaining_chars <= 0:
            break

        item_chars = min(MAX_KNOWLEDGE_ITEM_CHARS, remaining_chars)
        content, truncated = _truncate(str(document.page_content), item_chars)
        if not content:
            continue

        metadata = _safe_metadata(document)
        headers = [str(metadata[key]) for key in ("h1", "h2", "h3") if metadata.get(key)]
        source = str(metadata.get("_file_name") or metadata.get("_source") or "unknown_source")
        document_id = getattr(document, "id", None)
        records.append(
            {
                "id": f"K{index:02d}",
                "document_id": str(document_id) if document_id else None,
                "source": source,
                "title": " > ".join(headers) if headers else None,
                "content": content,
                "metadata": metadata,
                "truncated": truncated,
            }
        )
        remaining_chars -= len(content)

    return records


async def _retrieve_bounded_knowledge(query: str) -> list[dict[str, Any]]:
    """Retrieve knowledge once; never turn misses or failures into knowledge."""
    retrieval_query, _ = _truncate(query, MAX_RETRIEVAL_QUERY_CHARS)
    if not retrieval_query:
        return []

    try:
        # Supplying a ToolCall asks BaseTool to preserve the artifact in ToolMessage.
        result = await asyncio.wait_for(
            retrieve_knowledge.ainvoke(
                {
                    "type": "tool_call",
                    "id": "planner_knowledge_retrieval",
                    "name": retrieve_knowledge.name,
                    "args": {"query": retrieval_query},
                }
            ),
            timeout=config.aiops_tool_timeout_seconds,
        )
    except Exception as exc:
        logger.warning("Planner 知识检索失败，按无知识继续: {}: {}", type(exc).__name__, exc)
        return []

    content = _message_text(result).strip()
    artifact = getattr(result, "artifact", None)
    documents = [item for item in artifact or [] if isinstance(item, Document)]

    if documents:
        knowledge = _documents_to_knowledge(documents)
        logger.info("Planner 检索到 {} 条有界知识记录", len(knowledge))
        return knowledge

    if not content or any(marker in content for marker in _NO_KNOWLEDGE_MESSAGES):
        logger.info("Planner 未检索到相关知识，按无知识继续")
        return []
    if any(marker in content for marker in _KNOWLEDGE_ERROR_MESSAGES):
        logger.warning("Planner 知识检索返回错误内容，已丢弃")
        return []

    # 无 artifact 时无法区分真实文档、框架错误和工具状态文本；宁可缺失知识，
    # 也不把不可追踪的字符串升级为 Knowledge Evidence。
    logger.warning("Planner 知识工具未返回可追踪 artifact，已按无知识处理")
    return []


def _render_knowledge_context(knowledge: list[dict[str, Any]]) -> str:
    """Render bounded JSON for the Planner prompt."""
    if not knowledge:
        return "（没有可用的相关内部知识。未命中或检索错误不构成知识证据。）"
    rendered = json.dumps(knowledge, ensure_ascii=False, default=str)
    bounded, _ = _truncate(
        rendered,
        min(MAX_KNOWLEDGE_CONTEXT_CHARS, config.aiops_knowledge_max_chars),
    )
    return bounded


def _preferred_tools(available_names: set[str], *candidates: str) -> list[str]:
    selected = [name for name in candidates if name in available_names]
    if selected:
        return selected[:2]
    if "query_prometheus_alerts" in available_names:
        return ["query_prometheus_alerts"]
    return [sorted(available_names)[0]] if available_names else []


def _fallback_plan(available_tool_names: set[str]) -> DiagnosisPlan:
    """Create a validated three-hypothesis plan without free-form string steps."""
    hypotheses = [
        DiagnosisHypothesis(
            id="H1",
            cause="资源饱和或容量不足",
            description="CPU、内存或其他关键资源可能持续处于异常水位。",
            expected_evidence=[
                "运行时指标显示资源持续超过告警阈值",
                "同时间窗内存在与资源压力一致的告警或日志",
            ],
        ),
        DiagnosisHypothesis(
            id="H2",
            cause="应用错误或近期变更导致性能回退",
            description="应用异常、配置或部署变更可能引入错误与延迟。",
            expected_evidence=[
                "故障时间窗内出现聚合后的应用错误日志",
                "错误率或告警与变更时间相关",
            ],
        ),
        DiagnosisHypothesis(
            id="H3",
            cause="下游依赖超时或不可用",
            description="外部服务、数据库或网络依赖可能造成级联故障。",
            expected_evidence=[
                "本地资源正常但存在明确的下游超时日志",
                "依赖相关告警或调用失败与症状时间一致",
            ],
        ),
    ]
    steps = [
        DiagnosisStep(
            id="S1",
            hypothesis_id="H1",
            goal="验证关键资源是否在故障时间窗内持续饱和",
            rationale="资源指标与告警可以直接支持或反驳资源饱和假设。",
            tool_hint=_preferred_tools(
                available_tool_names,
                "query_cpu_metrics",
                "query_memory_metrics",
                "query_prometheus_alerts",
            ),
            expected_evidence=["CPU、内存的最大值、平均值、异常区间及关联告警"],
        ),
        DiagnosisStep(
            id="S2",
            hypothesis_id="H2",
            goal="验证应用错误或变更是否与故障时间一致",
            rationale="按服务和时间窗聚合错误日志可区分应用回退与资源问题。",
            tool_hint=_preferred_tools(
                available_tool_names,
                "search_log",
                "search_topic_by_service_name",
                "query_prometheus_alerts",
            ),
            expected_evidence=["错误模式、出现次数、首末时间及与告警的关联"],
        ),
        DiagnosisStep(
            id="S3",
            hypothesis_id="H3",
            goal="验证下游依赖是否出现超时或不可用",
            rationale="依赖超时日志与本地资源指标可支持或反驳下游故障假设。",
            tool_hint=_preferred_tools(
                available_tool_names,
                "search_log",
                "query_cpu_metrics",
                "query_memory_metrics",
                "query_prometheus_alerts",
            ),
            expected_evidence=["下游目标、超时错误聚合、本地资源水位及时间相关性"],
        ),
    ]
    return DiagnosisPlan(hypotheses=hypotheses, steps=steps)


def _validate_and_bound_plan(result: Any, available_tool_names: set[str]) -> DiagnosisPlan:
    """Validate cross references and remove hallucinated tool hints."""
    plan = result if isinstance(result, DiagnosisPlan) else DiagnosisPlan.model_validate(result)
    if not 3 <= len(plan.hypotheses) <= 5:
        raise ValueError("Planner must return between 3 and 5 hypotheses")

    payload = plan.model_dump(mode="json")
    covered_hypotheses: set[str] = set()
    for step in payload["steps"]:
        valid_hints = [name for name in step["tool_hint"] if name in available_tool_names]
        if available_tool_names and not valid_hints:
            raise ValueError(f"step {step['id']} has no available tool hint")
        step["tool_hint"] = valid_hints
        covered_hypotheses.add(step["hypothesis_id"])

    hypothesis_ids = {item["id"] for item in payload["hypotheses"]}
    if missing := hypothesis_ids - covered_hypotheses:
        raise ValueError(f"hypotheses without investigation steps: {sorted(missing)}")
    return cast(DiagnosisPlan, DiagnosisPlan.model_validate(payload))


def _knowledge_evidence(
    knowledge: list[dict[str, Any]],
    hypothesis_ids: list[str],
) -> list[dict[str, Any]]:
    """Represent knowledge as guidance evidence, never as online proof."""
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(knowledge, 1):
        excerpt, _ = _truncate(str(item["content"]), MAX_KNOWLEDGE_EVIDENCE_CHARS)
        source = str(item.get("source") or "unknown_source")
        record = DiagnosisEvidence(
            id=f"KE{index:02d}",
            source="knowledge_base",
            tool_name=retrieve_knowledge.name,
            hypothesis_ids=hypothesis_ids,
            observation=(
                f"内部知识（来源: {source}）提供以下调查指导，但不证明当前线上事实：" f"{excerpt}"
            ),
            supports=[],
            contradicts=[],
            reliability=None,
            # Planner retrieval is represented directly in ``retrieved_knowledge``;
            # it is not an Executor RawToolResult.
            raw_result_ref=None,
        )
        evidence.append(record.model_dump(mode="json"))
    return evidence


async def planner(state: PlanExecuteState) -> dict[str, Any]:
    """Generate bounded knowledge, hypotheses and hypothesis-bound steps."""
    diagnosis_id = state.get("diagnosis_id", "unknown")
    logger.info("diagnosis_id={} planner started", diagnosis_id)
    input_text = str(state.get("input", "")).strip()
    diagnosis_request, _ = _truncate(input_text, MAX_DIAGNOSIS_REQUEST_CHARS)

    knowledge = await _retrieve_bounded_knowledge(input_text)
    available_tools, discovery_error = await load_available_tools()
    if discovery_error:
        logger.warning("MCP 工具发现失败，使用本地工具继续: {}", discovery_error)

    tool_names = {
        str(getattr(tool, "name", "")) for tool in available_tools if getattr(tool, "name", None)
    }
    tools_description = format_tools_description(
        available_tools,
        max_chars=config.aiops_tool_schema_max_chars,
    )

    try:
        llm = ChatQwen(
            model=config.rag_model,
            api_key=SecretStr(config.dashscope_api_key),
            base_url=config.dashscope_api_base,
            temperature=0,
        )
        chain = PLANNER_PROMPT | llm.with_structured_output(DiagnosisPlan)
        result = await asyncio.wait_for(
            chain.ainvoke(
                {
                    "diagnosis_request": diagnosis_request or "诊断当前系统异常",
                    "tools_description": tools_description,
                    "knowledge_context": _render_knowledge_context(knowledge),
                }
            ),
            timeout=config.aiops_tool_timeout_seconds,
        )
        diagnosis_plan = _validate_and_bound_plan(result, tool_names)
    except Exception as exc:
        logger.error(
            "Planner Structured Output 失败，使用结构化 fallback: {}: {}",
            type(exc).__name__,
            exc,
        )
        diagnosis_plan = _fallback_plan(tool_names)

    plan_payload = diagnosis_plan.model_dump(mode="json")
    hypothesis_ids = [item["id"] for item in plan_payload["hypotheses"]]
    evidence = _knowledge_evidence(knowledge, hypothesis_ids)

    logger.info(
        "diagnosis_id={} planner completed hypotheses={} steps={} knowledge={} "
        "knowledge_evidence={}",
        diagnosis_id,
        len(plan_payload["hypotheses"]),
        len(plan_payload["steps"]),
        len(knowledge),
        len(evidence),
    )
    return {
        "retrieved_knowledge": knowledge,
        "evidence": evidence,
        "hypotheses": plan_payload["hypotheses"],
        "plan": plan_payload["steps"],
    }

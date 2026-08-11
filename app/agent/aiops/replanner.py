"""Evidence-driven replanning for the existing Plan-Execute-Replan workflow."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import SecretStr

from app.config import config

from .models import (
    DiagnosisHypothesis,
    DiagnosisStep,
    ReplanDecision,
    ToolError,
)
from .prompts import REPLANNER_SYSTEM_PROMPT
from .state import PlanExecuteState
from .utils import format_tools_description, load_available_tools

replanner_prompt = ChatPromptTemplate.from_messages(
    [("system", REPLANNER_SYSTEM_PROMPT), ("human", "{diagnosis_context}")]
)


def _tool_hint_for(
    text: str,
    failed_tools: set[str],
    available_tool_names: set[str],
) -> list[str]:
    lowered = text.lower()
    candidates: list[str]
    if "cpu" in lowered or "处理器" in text:
        candidates = ["query_cpu_metrics", "search_log"]
    elif "memory" in lowered or "内存" in text or "oom" in lowered or "gc" in lowered:
        candidates = ["query_memory_metrics", "search_log"]
    elif any(word in lowered for word in ("log", "timeout", "error", "database", "db")) or any(
        word in text for word in ("日志", "超时", "错误", "数据库")
    ):
        candidates = ["search_log", "query_cpu_metrics", "query_memory_metrics"]
    elif "alert" in lowered or "告警" in text:
        candidates = ["query_prometheus_alerts"]
    else:
        candidates = ["query_prometheus_alerts", "search_log", "query_cpu_metrics"]
    available = [
        name for name in candidates if name in available_tool_names and name not in failed_tools
    ]
    return available[:1]


def _hypothesis_id_from_missing(item: str, hypotheses: list[DiagnosisHypothesis]) -> str:
    match = re.search(r"\b(H\d+)\b", item, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if any(hypothesis.id == candidate for hypothesis in hypotheses):
            return candidate
    uncertain = [hypothesis for hypothesis in hypotheses if hypothesis.status != "supported"]
    return (uncertain or hypotheses)[0].id


def _fallback_replan(
    state: PlanExecuteState,
    available_tool_names: set[str],
) -> ReplanDecision:
    hypotheses = [DiagnosisHypothesis.model_validate(item) for item in state.get("hypotheses", [])]
    evaluation = state.get("evaluation") or {}
    missing = list(evaluation.get("missing_evidence", []))
    conflicts = list(evaluation.get("evidence_conflicts", []))
    failed_tools = {
        str(item.get("tool_name", ""))
        for item in state.get("tool_errors", [])
        if not item.get("handled", False)
    }
    next_round = int(state.get("replan_count", 0)) + 1
    capacity = max(
        0,
        min(
            config.aiops_max_steps - int(state.get("step_count", 0)),
            config.aiops_max_tool_calls - int(state.get("tool_call_count", 0)),
        ),
    )
    steps: list[DiagnosisStep] = []
    for index, item in enumerate(missing[:capacity], 1):
        hypothesis_id = _hypothesis_id_from_missing(item, hypotheses)
        tool_hint = _tool_hint_for(item, failed_tools, available_tool_names)
        if not tool_hint:
            continue
        steps.append(
            DiagnosisStep(
                id=f"RP{next_round}S{index}",
                hypothesis_id=hypothesis_id,
                goal=f"补充关键证据：{item}",
                rationale="Evidence Evaluator 标记该证据缺失，需要使用未失败的数据源补齐。",
                tool_hint=tool_hint,
                expected_evidence=[item],
            )
        )

    new_hypotheses: list[DiagnosisHypothesis] = []
    if conflicts and capacity > len(steps):
        existing_causes = " ".join(item.cause.lower() for item in hypotheses)
        if "network" not in existing_causes and "网络" not in existing_causes:
            numeric_ids = [
                int(match.group(1))
                for item in hypotheses
                if (match := re.fullmatch(r"H(\d+)", item.id, flags=re.IGNORECASE))
            ]
            new_id = f"H{max(numeric_ids, default=0) + 1}"
            new_hypothesis = DiagnosisHypothesis(
                id=new_id,
                cause="网络或依赖调用路径异常",
                description="作为冲突证据下的替代调查方向，不代表已确认的线上事实。",
                expected_evidence=["依赖调用延迟或网络错误的直接观测"],
            )
            new_hypotheses.append(new_hypothesis)
            tool_hint = _tool_hint_for("downstream timeout log", failed_tools, available_tool_names)
            if tool_hint:
                steps.append(
                    DiagnosisStep(
                        id=f"RP{next_round}S{len(steps) + 1}",
                        hypothesis_id=new_id,
                        goal="验证冲突是否来自网络或依赖调用路径",
                        rationale="现有证据相互冲突，需要引入可被独立验证的替代假设。",
                        tool_hint=tool_hint,
                        expected_evidence=["依赖超时、连接错误或正常调用的直接观测"],
                    )
                )

    if not steps and capacity > 0:
        for raw_error in state.get("tool_errors", []):
            error = ToolError.model_validate(raw_error)
            if error.handled or not (error.retryable or error.alternative_source):
                continue
            hypothesis_id = error.hypothesis_id or (hypotheses[0].id if hypotheses else "")
            if not hypothesis_id:
                continue
            tool_hint = _tool_hint_for(
                error.alternative_source or error.category,
                failed_tools,
                available_tool_names,
            )
            if not tool_hint:
                continue
            steps.append(
                DiagnosisStep(
                    id=f"RP{next_round}S{len(steps) + 1}",
                    hypothesis_id=hypothesis_id,
                    goal=f"使用替代数据源补偿 {error.tool_name} 失败",
                    rationale=f"原工具失败类别为 {error.category}，改用未失败的数据源。",
                    tool_hint=tool_hint,
                    expected_evidence=["与原步骤目标等价的替代运行时观测"],
                )
            )
            break

    return ReplanDecision(
        new_hypotheses=new_hypotheses,
        steps=steps[:capacity],
        reason="使用结构化缺失证据、冲突和工具错误生成的确定性后备计划。",
    )


def _merge_replan(
    state: PlanExecuteState,
    decision: ReplanDecision,
    available_tool_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge revisions while preserving collected evidence links."""
    existing = {
        item.id: item
        for item in (DiagnosisHypothesis.model_validate(raw) for raw in state.get("hypotheses", []))
    }
    used_hypothesis_ids = set(existing)
    numeric_ids = [
        int(match.group(1))
        for hypothesis_id in used_hypothesis_ids
        if (match := re.fullmatch(r"H(\d+)", hypothesis_id, flags=re.IGNORECASE))
    ]
    next_numeric_id = max(numeric_ids, default=0) + 1
    remapped_hypothesis_ids: dict[str, str] = {}

    for candidate in decision.new_hypotheses:
        original_id = candidate.id
        candidate_id = original_id
        if candidate_id in used_hypothesis_ids:
            while f"H{next_numeric_id}" in used_hypothesis_ids:
                next_numeric_id += 1
            candidate_id = f"H{next_numeric_id}"
            next_numeric_id += 1
            remapped_hypothesis_ids[original_id] = candidate_id

        # ``new_hypotheses`` is an untrusted LLM boundary.  It may introduce a
        # revised cause, but it may not bring model-invented evidence links or
        # reuse an existing id whose collected evidence described another
        # cause.  Collisions therefore become a fresh hypothesis id.
        sanitized = candidate.model_copy(
            update={
                "id": candidate_id,
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "confidence": 0.0,
                "status": "pending",
            }
        )
        existing[candidate_id] = sanitized
        used_hypothesis_ids.add(candidate_id)

    valid_ids = set(existing)
    seen_steps: set[str] = {
        str(item.get("step_id", "")) for item in state.get("execution_history", [])
    }
    seen_steps.update(str(item.get("id", "")) for item in state.get("plan", []))
    capacity = max(
        0,
        min(
            config.aiops_max_steps - int(state.get("step_count", 0)),
            config.aiops_max_tool_calls - int(state.get("tool_call_count", 0)),
        ),
    )
    steps: list[dict[str, Any]] = []
    replan_number = int(state.get("replan_count", 0)) + 1
    for index, step_candidate in enumerate(decision.steps, 1):
        target_hypothesis_id = remapped_hypothesis_ids.get(
            step_candidate.hypothesis_id,
            step_candidate.hypothesis_id,
        )
        if target_hypothesis_id not in valid_ids or len(steps) >= capacity:
            continue
        valid_hints = [name for name in step_candidate.tool_hint if name in available_tool_names]
        if available_tool_names and not valid_hints:
            continue
        step_candidate = step_candidate.model_copy(
            update={
                "hypothesis_id": target_hypothesis_id,
                "tool_hint": valid_hints,
            }
        )
        if step_candidate.id in seen_steps:
            step_candidate.id = f"RP{replan_number}S{index}"
        seen_steps.add(step_candidate.id)
        steps.append(step_candidate.model_dump(mode="json"))
    return [item.model_dump(mode="json") for item in existing.values()], steps


async def replanner(state: PlanExecuteState) -> dict[str, Any]:
    """Revise hypotheses/steps only when the evaluator requested it."""
    replan_count = int(state.get("replan_count", 0))
    diagnosis_id = state.get("diagnosis_id", "unknown")
    if replan_count >= config.aiops_max_replans:
        logger.info("diagnosis_id={} replan budget exhausted", diagnosis_id)
        return {
            "plan": [],
            "termination_reason": "replan_budget_exhausted",
        }

    if int(state.get("step_count", 0)) >= config.aiops_max_steps:
        return {"plan": [], "termination_reason": "step_budget_exhausted"}
    if int(state.get("tool_call_count", 0)) >= config.aiops_max_tool_calls:
        return {"plan": [], "termination_reason": "tool_call_budget_exhausted"}

    tools, discovery_error = await load_available_tools()
    tool_description = format_tools_description(
        tools,
        max_chars=config.aiops_tool_schema_max_chars,
    )
    context = {
        "diagnosis_id": diagnosis_id,
        "query": state.get("input", ""),
        "hypotheses": state.get("hypotheses", []),
        "evidence": state.get("evidence", []),
        "evaluation": state.get("evaluation", {}),
        "remaining_plan": state.get("plan", []),
        "tool_errors": [
            item for item in state.get("tool_errors", []) if not item.get("handled", False)
        ],
        "available_tools": tool_description,
        "tool_discovery_warning": discovery_error,
        "remaining_step_budget": config.aiops_max_steps - int(state.get("step_count", 0)),
        "remaining_tool_budget": config.aiops_max_tool_calls - int(state.get("tool_call_count", 0)),
    }
    context_text = json.dumps(context, ensure_ascii=False, default=str)
    context_text = context_text[: config.aiops_tool_schema_max_chars * 2]

    decision: ReplanDecision
    try:
        llm = ChatQwen(
            model=config.rag_model,
            api_key=SecretStr(config.dashscope_api_key),
            base_url=config.dashscope_api_base,
            temperature=0,
        )
        chain = replanner_prompt | llm.with_structured_output(ReplanDecision)
        result = await asyncio.wait_for(
            chain.ainvoke({"diagnosis_context": context_text}),
            timeout=config.aiops_tool_timeout_seconds,
        )
        decision = (
            result if isinstance(result, ReplanDecision) else ReplanDecision.model_validate(result)
        )
    except Exception as exc:
        logger.warning(
            "diagnosis_id={} structured replanning failed; using deterministic fallback: {}",
            diagnosis_id,
            exc,
        )
        decision = _fallback_replan(
            state,
            {str(getattr(tool, "name", "")) for tool in tools},
        )

    available_tool_names = {str(getattr(tool, "name", "")) for tool in tools}
    if not decision.steps and (state.get("evaluation") or {}).get("need_replan"):
        decision = _fallback_replan(state, available_tool_names)
    hypotheses, steps = _merge_replan(state, decision, available_tool_names)
    replanned_hypotheses = {str(item.get("hypothesis_id", "")) for item in steps}
    handled_errors = []
    for raw in state.get("tool_errors", []):
        error = ToolError.model_validate(raw)
        if (
            error.hypothesis_id in replanned_hypotheses
            or not error.retryable
            or error.alternative_source is None
        ):
            error = error.model_copy(update={"handled": True})
        handled_errors.append(error.model_dump(mode="json"))

    update: dict[str, Any] = {
        "hypotheses": hypotheses,
        "plan": steps,
        "tool_errors": handled_errors,
        "replan_count": replan_count + 1,
        "termination_reason": None,
    }
    if not steps:
        update["termination_reason"] = "no_alternative_evidence_source"
    logger.info(
        "diagnosis_id={} replan_count={} new_hypotheses={} new_steps={} reason={}",
        diagnosis_id,
        replan_count + 1,
        len(decision.new_hypotheses),
        len(steps),
        decision.reason,
    )
    return update

"""LangGraph service for hypothesis- and evidence-driven AIOps diagnosis."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger

from app.agent.aiops.evidence import evidence_evaluator, evidence_extractor
from app.agent.aiops.executor import executor
from app.agent.aiops.planner import planner
from app.agent.aiops.replanner import replanner
from app.agent.aiops.reporting import final_report, root_cause_ranker
from app.agent.aiops.routing import route_after_evaluation, route_after_replan
from app.agent.aiops.state import PlanExecuteState
from app.config import config

NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_EXTRACTOR = "evidence_extractor"
NODE_EVALUATOR = "evidence_evaluator"
NODE_REPLANNER = "replanner"
NODE_TERMINATE = "termination"
NODE_RANKER = "root_cause_ranker"
NODE_REPORT = "final_report"

DEFAULT_DIAGNOSIS_QUERY = (
    "诊断当前系统是否存在活跃告警；如存在，基于监控、日志与知识库经验提出候选根因，"
    "逐项收集可验证证据，并输出明确披露不确定性的故障诊断报告。"
)


def _mark_termination(state: PlanExecuteState) -> dict[str, Any]:
    """Persist a deterministic reason before the P0 terminal branch reaches END."""
    existing = state.get("termination_reason")
    if existing:
        return {"termination_reason": existing}
    evaluation = state.get("evaluation") or {}
    if int(state.get("tool_call_count", 0)) >= config.aiops_max_tool_calls:
        reason = "tool_call_budget_exhausted"
    elif int(state.get("step_count", 0)) >= config.aiops_max_steps:
        reason = "step_budget_exhausted"
    elif evaluation.get("budget_exhausted"):
        reason = "investigation_budget_exhausted"
    elif evaluation.get("can_finish") and evaluation.get("supported_hypotheses"):
        reason = "evidence_sufficient"
    elif evaluation.get("evidence_conflicts"):
        reason = "evidence_conflict_unresolved"
    elif int(state.get("replan_count", 0)) >= config.aiops_max_replans:
        reason = "replan_budget_exhausted"
    elif state.get("tool_errors"):
        reason = "tool_unavailable_partial_diagnosis"
    else:
        reason = "inconclusive_partial_diagnosis"
    logger.info(
        "diagnosis_id={} termination reason={}",
        state.get("diagnosis_id", "unknown"),
        reason,
    )
    return {"termination_reason": reason}


class AIOpsService:
    """Build and stream the existing workflow with explicit evidence stages."""

    def __init__(self) -> None:
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        logger.info("Evidence-driven Plan-Execute-Replan service initialized")

    def _build_graph(self) -> Any:
        workflow = StateGraph(PlanExecuteState)
        workflow.add_node(NODE_PLANNER, planner)
        workflow.add_node(NODE_EXECUTOR, executor)
        workflow.add_node(NODE_EXTRACTOR, evidence_extractor)
        workflow.add_node(NODE_EVALUATOR, evidence_evaluator)
        workflow.add_node(NODE_REPLANNER, replanner)
        workflow.add_node(NODE_TERMINATE, _mark_termination)
        workflow.add_node(NODE_RANKER, root_cause_ranker)
        workflow.add_node(NODE_REPORT, final_report)

        workflow.set_entry_point(NODE_PLANNER)
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_EXTRACTOR)
        workflow.add_edge(NODE_EXTRACTOR, NODE_EVALUATOR)
        workflow.add_conditional_edges(
            NODE_EVALUATOR,
            route_after_evaluation,
            {
                "execute": NODE_EXECUTOR,
                "replan": NODE_REPLANNER,
                "rank": NODE_TERMINATE,
            },
        )
        workflow.add_conditional_edges(
            NODE_REPLANNER,
            route_after_replan,
            {"execute": NODE_EXECUTOR, "rank": NODE_TERMINATE},
        )
        workflow.add_edge(NODE_TERMINATE, NODE_RANKER)
        workflow.add_edge(NODE_RANKER, NODE_REPORT)
        workflow.add_edge(NODE_REPORT, END)
        return workflow.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _initial_state(
        *,
        diagnosis_id: str,
        session_id: str,
        user_input: str,
        alert_context: dict[str, Any] | None,
        scenario: str | None,
    ) -> PlanExecuteState:
        return {
            "diagnosis_id": diagnosis_id,
            "session_id": session_id,
            "input": user_input,
            "alert_context": alert_context or {},
            "scenario": scenario,
            "retrieved_knowledge": [],
            "hypotheses": [],
            "plan": [],
            "execution_history": [],
            "raw_tool_results": [],
            "evidence": [],
            "tool_errors": [],
            "replan_count": 0,
            "tool_call_count": 0,
            "step_count": 0,
            "remaining_budget": config.aiops_max_tool_calls,
            "evaluation": None,
            "root_causes": [],
            "response": "",
            "termination_reason": None,
        }

    async def execute(
        self,
        user_input: str,
        session_id: str = "default",
        *,
        alert_context: dict[str, Any] | None = None,
        scenario: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        diagnosis_id = uuid4().hex
        logger.info(
            "diagnosis_id={} session_id={} diagnosis started",
            diagnosis_id,
            session_id,
        )
        initial_state = self._initial_state(
            diagnosis_id=diagnosis_id,
            session_id=session_id,
            user_input=user_input,
            alert_context=alert_context,
            scenario=scenario,
        )
        # Each diagnosis uses an isolated checkpoint thread.  session_id remains
        # correlation metadata, avoiding additive state leakage across requests.
        graph_config: dict[str, Any] = {
            "configurable": {"thread_id": diagnosis_id},
            "recursion_limit": config.aiops_max_steps * 4 + config.aiops_max_replans * 2 + 8,
        }

        try:
            async for event in self.graph.astream(
                input=initial_state,
                config=graph_config,
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    yield self._format_node_event(diagnosis_id, node_name, node_output or {})

            snapshot = self.graph.get_state(graph_config)
            values: dict[str, Any] = dict(snapshot.values) if snapshot and snapshot.values else {}
            yield {
                "type": "complete",
                "stage": "diagnosis_complete",
                "message": "诊断工作流已停止",
                "diagnosis_id": diagnosis_id,
                "response": values.get("response", ""),
                "termination_reason": values.get("termination_reason"),
                "diagnosis_confidence": (values.get("evaluation") or {}).get(
                    "diagnosis_confidence", 0.0
                ),
                "tool_call_count": values.get("tool_call_count", 0),
                "replan_count": values.get("replan_count", 0),
                "step_count": values.get("step_count", 0),
            }
            logger.info(
                "diagnosis_id={} diagnosis stopped reason={} steps={} tools={} replans={}",
                diagnosis_id,
                values.get("termination_reason"),
                values.get("step_count", 0),
                values.get("tool_call_count", 0),
                values.get("replan_count", 0),
            )
        except Exception as exc:
            logger.exception("diagnosis_id={} diagnosis failed: {}", diagnosis_id, exc)
            yield {
                "type": "error",
                "stage": "error",
                "diagnosis_id": diagnosis_id,
                "message": f"诊断流程异常: {type(exc).__name__}: {exc}",
            }

    async def diagnose(
        self,
        session_id: str = "default",
        *,
        query: str | None = None,
        alert_context: dict[str, Any] | None = None,
        scenario: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Compatibility entry point used by the SSE API."""
        async for event in self.execute(
            query or DEFAULT_DIAGNOSIS_QUERY,
            session_id,
            alert_context=alert_context,
            scenario=scenario,
        ):
            yield event

    @staticmethod
    def _format_node_event(
        diagnosis_id: str,
        node_name: str,
        state_update: dict[str, Any],
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "type": "status",
            "stage": node_name,
            "diagnosis_id": diagnosis_id,
        }
        if node_name == NODE_PLANNER:
            plan = state_update.get("plan", [])
            base.update(
                type="plan",
                message=f"已生成 {len(state_update.get('hypotheses', []))} 个候选假设、"
                f"{len(plan)} 个验证步骤",
                hypotheses=state_update.get("hypotheses", []),
                plan=plan,
            )
        elif node_name == NODE_EXECUTOR:
            history = state_update.get("execution_history", [])
            current = history[-1] if history else {}
            base.update(
                type="step_complete",
                message="工具执行完成，等待证据提取",
                step=current,
                remaining_steps=len(state_update.get("plan", [])),
            )
        elif node_name == NODE_EXTRACTOR:
            evidence = state_update.get("evidence", [])
            base.update(
                type="evidence",
                message=f"当前已形成 {len(evidence)} 条结构化证据",
                evidence_count=len(evidence),
                evidence=evidence[-10:],
            )
        elif node_name == NODE_EVALUATOR:
            base.update(
                type="evaluation",
                message="证据充分性评估完成",
                evaluation=state_update.get("evaluation", {}),
            )
        elif node_name == NODE_REPLANNER:
            base.update(
                type="replan",
                message=f"重规划完成，新增 {len(state_update.get('plan', []))} 个步骤",
                plan=state_update.get("plan", []),
                replan_count=state_update.get("replan_count", 0),
            )
        elif node_name == NODE_RANKER:
            base.update(
                type="root_causes",
                message="根因候选排序完成",
                root_causes=state_update.get("root_causes", []),
            )
        elif node_name == NODE_REPORT:
            base.update(
                type="report",
                stage="final_report",
                message="证据约束诊断报告已生成",
                report=state_update.get("response", ""),
            )
        else:
            base["message"] = f"节点 {node_name} 已完成"
        return base


aiops_service = AIOpsService()

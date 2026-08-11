"""Serializable LangGraph state for evidence-grounded diagnosis.

State values intentionally remain JSON-compatible dictionaries and lists.
Pydantic models from :mod:`app.agent.aiops.models` are used at node boundaries,
then dumped with ``mode="json"`` before they are stored here.  No field uses a
reducer: nodes return complete replacements for the lists they update, which
keeps retries and checkpoint replay idempotent.
"""

from __future__ import annotations

from typing import Any, TypedDict


class PlanExecuteState(TypedDict, total=False):
    """Minimal state shared by Planner, Executor, Evaluator and Replanner."""

    # Correlation and request context.
    diagnosis_id: str
    session_id: str
    input: str
    alert_context: dict[str, Any]
    scenario: str | None

    # Explicit knowledge, hypotheses and remaining structured plan.
    retrieved_knowledge: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    plan: list[dict[str, Any]]

    # Execution data is deliberately split by abstraction level.
    execution_history: list[dict[str, Any]]
    raw_tool_results: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    tool_errors: list[dict[str, Any]]

    # Small, deterministic loop budget.
    replan_count: int
    tool_call_count: int
    step_count: int
    remaining_budget: int

    # Evaluation, ranked conclusions and terminal output.
    evaluation: dict[str, Any] | None
    root_causes: list[dict[str, Any]]
    response: str
    termination_reason: str | None

"""Pure-data factories shared by AIOps unit tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture
def hypothesis_factory() -> Callable[..., dict[str, Any]]:
    def make_hypothesis(
        hypothesis_id: str = "H1",
        *,
        cause: str = "CPU saturation",
        expected_evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": hypothesis_id,
            "cause": cause,
            "description": f"Candidate cause: {cause}",
            "expected_evidence": expected_evidence
            or ["CPU above threshold", "matching runtime errors"],
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "confidence": 0.0,
            "status": "pending",
        }

    return make_hypothesis


@pytest.fixture
def step_factory() -> Callable[..., dict[str, Any]]:
    def make_step(
        step_id: str = "S1",
        *,
        hypothesis_id: str = "H1",
        expected_evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": step_id,
            "hypothesis_id": hypothesis_id,
            "goal": "Verify the candidate with runtime telemetry",
            "rationale": "The observation can support or contradict the candidate",
            "tool_hint": ["query_cpu_metrics"],
            "expected_evidence": expected_evidence or ["CPU above threshold"],
        }

    return make_step


@pytest.fixture
def raw_result_factory() -> Callable[..., dict[str, Any]]:
    def make_raw_result(
        raw_id: str = "R1",
        *,
        tool_name: str = "query_cpu_metrics",
        source: str = "monitor",
        payload: Any = None,
        content: str = "",
        is_error: bool = False,
        error_type: str | None = None,
        hypothesis_ids: list[str] | None = None,
        diagnostic_goal: str = "Verify H1 using bounded runtime evidence",
        expected_evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": raw_id,
            "step_id": "S1",
            "hypothesis_ids": hypothesis_ids or ["H1"],
            "tool_call_id": f"call-{raw_id}",
            "tool_name": tool_name,
            "source": source,
            "diagnostic_goal": diagnostic_goal,
            "expected_evidence": expected_evidence or ["CPU above threshold"],
            "arguments": {"service_name": "checkout-service"},
            "content": content,
            "payload": payload,
            "is_error": is_error,
            "error_type": error_type,
            "truncated": False,
            "duration_ms": 12.5,
            "evidence_extracted": False,
        }

    return make_raw_result


@pytest.fixture
def evidence_factory() -> Callable[..., dict[str, Any]]:
    def make_evidence(
        evidence_id: str,
        *,
        source: str,
        supports: list[str] | None = None,
        contradicts: list[str] | None = None,
        reliability: float = 0.9,
        observation: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": evidence_id,
            "source": source,
            "tool_name": f"{source}_tool",
            "hypothesis_ids": ["H1"],
            "observation": observation or f"{source} runtime observation for H1",
            "supports": supports or [],
            "contradicts": contradicts or [],
            "reliability": reliability,
            "raw_result_ref": f"R-{evidence_id}",
        }

    return make_evidence

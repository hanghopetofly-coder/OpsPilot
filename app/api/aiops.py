"""Evidence-driven AIOps SSE endpoint."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from fastapi import APIRouter
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.models.aiops import AIOpsRequest, ScenarioName
from app.services.aiops_service import aiops_service

router = APIRouter()


class _DiagnosisKwargs(TypedDict):
    session_id: str
    query: str | None
    alert_context: dict[str, Any] | None
    scenario: ScenarioName | None


def _diagnosis_kwargs(request: AIOpsRequest) -> _DiagnosisKwargs:
    """Map the validated HTTP contract to the service contract."""

    return {
        "session_id": request.session_id,
        "query": request.query,
        "alert_context": request.alert_context or None,
        "scenario": request.scenario,
    }


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest) -> EventSourceResponse:
    """Stream one isolated hypothesis/evidence diagnosis.

    The POST body may provide ``query``, ``alert_context`` and one deterministic
    local ``scenario``.  Events progress through ``plan``, ``step_complete``,
    ``evidence``, ``evaluation``, optional ``replan``, ``root_causes``,
    ``report`` and finally ``complete``.  ``error`` terminates the stream.

    POST SSE must be consumed with ``fetch``/a readable stream (native
    ``EventSource`` only sends GET requests).
    """

    session_id = request.session_id
    logger.info(
        "session_id={} aiops request received scenario={}",
        session_id,
        request.scenario,
    )

    async def event_generator():
        try:
            async for event in aiops_service.diagnose(**_diagnosis_kwargs(request)):
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False),
                }
                if event.get("type") in {"complete", "error"}:
                    break

            logger.info("session_id={} aiops SSE completed", session_id)
        except Exception as exc:
            logger.exception("session_id={} aiops SSE failed: {}", session_id, exc)
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "error",
                        "stage": "exception",
                        "message": f"诊断异常: {type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())

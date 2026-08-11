"""AIOps HTTP request-to-service contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.aiops import _diagnosis_kwargs
from app.models.aiops import AIOpsRequest


def test_aiops_request_maps_query_context_and_scenario_to_service() -> None:
    request = AIOpsRequest(
        session_id="session-123",
        query="diagnose checkout latency",
        alert_context={"service": "checkout", "window": "15m"},
        scenario="downstream_timeout",
    )

    assert _diagnosis_kwargs(request) == {
        "session_id": "session-123",
        "query": "diagnose checkout latency",
        "alert_context": {"service": "checkout", "window": "15m"},
        "scenario": "downstream_timeout",
    }


def test_aiops_request_defaults_are_backward_compatible() -> None:
    request = AIOpsRequest()

    assert _diagnosis_kwargs(request) == {
        "session_id": "default",
        "query": None,
        "alert_context": None,
        "scenario": None,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"scenario": "random_failure"},
        {"session_id": ""},
        {"unknown_field": True},
    ],
)
def test_aiops_request_rejects_invalid_contract(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AIOpsRequest.model_validate(payload)

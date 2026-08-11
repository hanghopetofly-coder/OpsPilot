"""Bounded raw-result and logical tool-call accounting tests."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

executor_module = importlib.import_module("app.agent.aiops.executor")


def test_request_scenario_overrides_model_args_only_for_declared_tool_schema() -> None:
    calls = [
        {"name": "query_cpu_metrics", "args": {"service_name": "checkout"}, "id": "c1"},
        {"name": "get_current_time", "args": {}, "id": "c2"},
        {
            "name": "search_log",
            "args": {"scenario": "normal"},
            "id": "c3",
        },
    ]
    tools = [
        SimpleNamespace(name="query_cpu_metrics", args={"service_name": {}, "scenario": {}}),
        SimpleNamespace(name="get_current_time", args={"timezone": {}}),
        SimpleNamespace(name="search_log", args={"topic_id": {}, "scenario": {}}),
    ]

    injected = executor_module._inject_scenario(calls, tools, "cpu_saturation")

    assert injected[0]["args"]["scenario"] == "cpu_saturation"
    assert "scenario" not in injected[1]["args"]
    assert injected[2]["args"]["scenario"] == "cpu_saturation"

    defaulted = executor_module._inject_scenario(calls, tools, None)
    assert defaulted[0]["args"]["scenario"] == "normal"
    assert "scenario" not in defaulted[1]["args"]
    assert defaulted[2]["args"]["scenario"] == "normal"


def test_knowledge_backend_error_text_is_not_treated_as_evidence() -> None:
    assert executor_module._looks_like_error("检索知识时发生错误: Milvus unavailable", None)


def test_raw_tool_fields_share_one_json_safe_character_budget(monkeypatch: Any) -> None:
    limit = 160
    monkeypatch.setattr(executor_module.config, "aiops_raw_result_max_chars", limit)
    arguments_input = {
        "service": "checkout" * 100,
        "non_json": {1, 2, 3},
    }
    payload_input = {
        "samples": ["metric" * 100 for _ in range(10)],
        "opaque": object(),
    }

    arguments, content, payload, truncated = executor_module._bounded_tool_result_parts(
        arguments=arguments_input,
        content="runtime observation " * 200,
        payload=payload_input,
    )

    # These dumps intentionally omit ``default=str``: returned values must
    # already be checkpoint-safe JSON data.
    json.dumps(arguments, ensure_ascii=False)
    json.dumps(payload, ensure_ascii=False)
    total_chars = (
        len(executor_module._json_text(arguments))
        + len(content)
        + (len(executor_module._json_text(payload)) if payload is not None else 0)
    )
    assert total_chars <= limit
    assert truncated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_type", "expected_category"),
    [(TimeoutError, "timeout"), (RuntimeError, "connection")],
    ids=["timeout", "ordinary-exception"],
)
async def test_started_tool_calls_consume_budget_on_failure(
    monkeypatch: Any,
    failure_type: type[Exception],
    expected_category: str,
) -> None:
    tool_calls = [
        {
            "name": "query_cpu_metrics",
            "args": {"service": "checkout"},
            "id": "call-1",
            "type": "tool_call",
        },
        {
            "name": "query_logs",
            "args": {"service": "checkout"},
            "id": "call-2",
            "type": "tool_call",
        },
    ]

    class FakeSelector:
        async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
            return SimpleNamespace(content="", tool_calls=tool_calls)

    class FakeChatModel:
        def bind_tools(self, tools: list[Any]) -> FakeSelector:
            return FakeSelector()

    class FailingToolNode:
        def __init__(self, tools: list[Any]) -> None:
            self.tools = tools

        async def ainvoke(self, value: dict[str, Any]) -> dict[str, Any]:
            raise failure_type("offline simulated tool failure")

    async def fake_load_available_tools() -> tuple[list[Any], None]:
        return [object()], None

    monkeypatch.setattr(executor_module, "load_available_tools", fake_load_available_tools)
    monkeypatch.setattr(executor_module, "ChatQwen", lambda **kwargs: FakeChatModel())
    monkeypatch.setattr(executor_module, "ToolNode", FailingToolNode)

    initial_tool_count = 3
    state = {
        "diagnosis_id": "diag-test",
        "input": "diagnose checkout",
        "alert_context": {},
        "scenario": "unit-test",
        "plan": [
            {
                "id": "S1",
                "hypothesis_id": "H1",
                "goal": "Verify checkout telemetry",
                "rationale": "Runtime data can validate H1",
                "tool_hint": ["query_cpu_metrics", "query_logs"],
                "expected_evidence": ["CPU metric", "error logs"],
            }
        ],
        "evidence": [],
        "execution_history": [],
        "raw_tool_results": [],
        "tool_errors": [],
        "tool_call_count": initial_tool_count,
        "step_count": 1,
    }

    update = await executor_module.executor(state)

    expected_count = initial_tool_count + len(tool_calls)
    assert update["tool_call_count"] == expected_count
    assert update["remaining_budget"] == max(
        0,
        executor_module.config.aiops_max_tool_calls - expected_count,
    )
    assert update["step_count"] == 2
    assert update["plan"] == []
    assert update["tool_errors"][-1]["category"] == expected_category


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["construct", "bind"])
async def test_model_setup_failure_becomes_bounded_execution_record(
    monkeypatch: Any,
    failure_stage: str,
) -> None:
    async def fake_load_available_tools() -> tuple[list[Any], None]:
        return [SimpleNamespace(name="query_cpu_metrics", args={"scenario": {}})], None

    class BindFailure:
        def bind_tools(self, tools: list[Any]) -> None:
            raise RuntimeError("offline simulated bind failure")

    def fake_chat_model(**kwargs: Any) -> BindFailure:
        if failure_stage == "construct":
            raise RuntimeError("offline simulated constructor failure")
        return BindFailure()

    monkeypatch.setattr(executor_module, "load_available_tools", fake_load_available_tools)
    monkeypatch.setattr(executor_module, "ChatQwen", fake_chat_model)
    initial_tool_count = 2
    state = {
        "diagnosis_id": "diag-model-setup-failure",
        "input": "diagnose checkout",
        "alert_context": {},
        "scenario": None,
        "plan": [
            {
                "id": "S1",
                "hypothesis_id": "H1",
                "goal": "Verify checkout telemetry",
                "rationale": "Runtime data can validate H1",
                "tool_hint": ["query_cpu_metrics"],
                "expected_evidence": ["CPU metric"],
            }
        ],
        "evidence": [],
        "execution_history": [],
        "raw_tool_results": [],
        "tool_errors": [],
        "tool_call_count": initial_tool_count,
        "step_count": 0,
    }

    update = await executor_module.executor(state)

    assert update["plan"] == []
    assert update["step_count"] == 1
    assert update["tool_call_count"] == initial_tool_count
    assert update["execution_history"][-1]["status"] == "failed"
    assert update["tool_errors"][-1]["tool_name"] == "tool_selection_or_execution"
    assert update["tool_errors"][-1]["category"] == "connection"

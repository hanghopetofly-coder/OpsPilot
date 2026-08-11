"""Graph compilation smoke test with every external constructor forbidden."""

from __future__ import annotations

import importlib
import sys
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_qwq import ChatQwen
from pymilvus import MilvusClient, connections


def test_aiops_service_imports_and_compiles_without_external_connections(monkeypatch: Any) -> None:
    external_calls: list[str] = []

    def forbidden(name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> None:
            external_calls.append(name)
            raise AssertionError(f"graph compilation unexpectedly initialized {name}")

        return fail

    monkeypatch.setattr(connections, "connect", forbidden("Milvus connection"))
    monkeypatch.setattr(MilvusClient, "__init__", forbidden("Milvus client"))
    monkeypatch.setattr(MultiServerMCPClient, "__init__", forbidden("MCP client"))
    monkeypatch.setattr(ChatQwen, "__init__", forbidden("LLM client"))

    sys.modules.pop("app.services.aiops_service", None)
    module = importlib.import_module("app.services.aiops_service")
    service = module.AIOpsService()
    node_names = set(service.graph.get_graph().nodes)

    assert external_calls == []
    assert {
        "planner",
        "executor",
        "evidence_extractor",
        "evidence_evaluator",
        "replanner",
        "termination",
        "root_cause_ranker",
        "final_report",
    } <= node_names
    assert service.graph.checkpointer is service.checkpointer

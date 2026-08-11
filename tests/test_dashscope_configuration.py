"""Regression tests for explicit DashScope endpoint configuration."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings

CHINA_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAT_QWEN_MODULES = (
    "app/services/rag_agent_service.py",
    "app/agent/aiops/planner.py",
    "app/agent/aiops/executor.py",
    "app/agent/aiops/replanner.py",
)


def test_dashscope_api_base_defaults_to_china_and_allows_env_override(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)
    assert Settings(_env_file=None).dashscope_api_base == CHINA_DASHSCOPE_BASE

    override = "https://example.test/compatible-mode/v1"
    monkeypatch.setenv("DASHSCOPE_API_BASE", override)
    assert Settings(_env_file=None).dashscope_api_base == override


@pytest.mark.parametrize("relative_path", CHAT_QWEN_MODULES)
def test_chatqwen_constructors_explicitly_receive_configured_base_url(
    relative_path: str,
) -> None:
    source_path = PROJECT_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ChatQwen"
    ]

    assert constructors, f"No ChatQwen constructor found in {relative_path}"
    for constructor in constructors:
        base_url = next(
            (keyword.value for keyword in constructor.keywords if keyword.arg == "base_url"),
            None,
        )
        assert base_url is not None
        assert ast.unparse(base_url) == "config.dashscope_api_base"

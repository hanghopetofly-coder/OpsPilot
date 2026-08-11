"""AIOps Agent 通用工具函数。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from app.agent.mcp_client import (
    format_exception_chain,
    get_mcp_client_with_retry,
    load_mcp_tools_safe,
)

MAX_TOOL_CATALOG_CHARS = 12_000
MAX_TOOL_DESCRIPTION_CHARS = 800
MAX_TOOL_SCHEMA_CHARS = 2_000


def _truncate(value: str, limit: int) -> str:
    """按字符上限裁剪文本，并显式标记被裁剪的内容。"""
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    marker = "…[truncated]"
    if limit <= len(marker):
        return marker[:limit]
    return f"{value[: limit - len(marker)]}{marker}"


def _tool_args_schema(tool: Any) -> str:
    """将 LangChain/MCP 工具参数 Schema 转成有界 JSON 文本。"""
    schema = getattr(tool, "args_schema", None)
    schema_payload: Any = None

    try:
        if isinstance(schema, dict):
            schema_payload = schema
        elif schema is not None and hasattr(schema, "model_json_schema"):
            schema_payload = schema.model_json_schema()
        elif schema is not None and hasattr(schema, "schema"):
            schema_payload = schema.schema()
        else:
            # BaseTool.args 已是精简后的 properties 映射，可覆盖无 args_schema 的工具。
            schema_payload = getattr(tool, "args", None)
    except Exception as exc:  # 工具 Schema 不应阻断整个 Planner
        schema_payload = {"schema_error": f"{type(exc).__name__}: {exc}"}

    if not schema_payload:
        return "{}"

    try:
        rendered = json.dumps(
            schema_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError) as exc:
        rendered = json.dumps(
            {"schema_error": f"{type(exc).__name__}: {exc}"},
            ensure_ascii=False,
        )

    return _truncate(rendered, MAX_TOOL_SCHEMA_CHARS)


def format_tools_description(
    tools: Sequence[Any],
    *,
    max_chars: int = MAX_TOOL_CATALOG_CHARS,
) -> str:
    """格式化有界工具目录，包含名称、描述和 ``args_schema``。

    本函数只为 Planner 提供工具选择依据；Schema 和描述都设置字符上限，避免
    MCP 服务返回超大文档字符串后无界进入模型上下文。
    """
    descriptions: list[str] = []
    seen_names: set[str] = set()

    for index, tool in enumerate(tools, 1):
        name = str(getattr(tool, "name", "") or f"unnamed_tool_{index}")
        if name in seen_names:
            continue
        seen_names.add(name)

        description = _truncate(
            str(getattr(tool, "description", "") or "无描述"),
            MAX_TOOL_DESCRIPTION_CHARS,
        )
        args_schema = _tool_args_schema(tool)
        descriptions.append(
            f"- name: {name}\n  description: {description}\n  args_schema: {args_schema}"
        )

    if not descriptions:
        return "（当前没有可用工具）"

    return _truncate("\n".join(descriptions), max_chars)


async def load_available_tools() -> tuple[list[Any], str | None]:
    """安全加载本地与 MCP 工具，MCP discovery 失败时保留本地工具。

    Returns:
        ``(tools, discovery_error)``。``tools`` 始终包含默认本地工具；第二项仅在
        MCP 客户端创建或工具发现失败时返回可读错误链。
    """
    # 延迟导入可减少仅使用纯格式化函数时触发知识库基础设施初始化的机会。
    from app.tools import DEFAULT_LOCAL_AGENT_TOOLS

    local_tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

    try:
        client = await get_mcp_client_with_retry()
    except Exception as exc:
        return local_tools, format_exception_chain(exc)

    mcp_tools, discovery_error = await load_mcp_tools_safe(client)

    # 本地工具优先；按名称去重，防止 MCP 暴露同名工具造成模型选择歧义。
    available: list[Any] = []
    seen_names: set[str] = set()
    for tool in [*local_tools, *mcp_tools]:
        name = str(getattr(tool, "name", "") or id(tool))
        if name in seen_names:
            continue
        seen_names.add(name)
        available.append(tool)

    return available, discovery_error

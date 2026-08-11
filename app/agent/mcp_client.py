"""
MCP 客户端管理
提供全局单例的 MCP 客户端，避免重复初始化
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from loguru import logger
from mcp.types import CallToolResult, TextContent

from app.config import config

# 全局 MCP 客户端（延迟初始化）
_mcp_client: MultiServerMCPClient | None = None


def format_exception_chain(exc: BaseException) -> str:
    """展开 ExceptionGroup / TaskGroup，便于日志定位真实子异常。"""
    sub_exceptions = getattr(exc, "exceptions", None)
    if sub_exceptions is not None:
        lines = [str(exc)]
        for i, sub in enumerate(sub_exceptions):
            lines.append(f"  [{i}] {format_exception_chain(sub)}")
        return "\n".join(lines)
    msg = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return f"{msg}\n  caused by: {format_exception_chain(cause)}"
    return msg


async def load_mcp_tools_safe(
    client: MultiServerMCPClient,
    timeout_seconds: float | None = None,
) -> tuple[list[Any], str | None]:
    """有界加载 MCP 工具；取消信号继续向上传播。"""

    timeout = timeout_seconds or config.aiops_tool_timeout_seconds
    try:
        tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
        return tools, None
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return [], f"MCP tool discovery timed out after {timeout:.1f}s"
    except Exception as exc:
        return [], format_exception_chain(exc)


def _error_result(message: str) -> CallToolResult:
    """Build the MCP protocol error shape expected by the adapter."""

    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


def _result_error_message(result: CallToolResult) -> str:
    messages = [
        str(getattr(item, "text", "")) for item in result.content if getattr(item, "text", None)
    ]
    return " | ".join(messages) or "MCP tool returned isError=true"


async def retry_interceptor(
    request: MCPToolCallRequest,
    handler: Callable[[MCPToolCallRequest], Awaitable[CallToolResult]],
    max_attempts: int | None = None,
    delay: float | None = None,
    timeout_seconds: float | None = None,
) -> CallToolResult:
    """Retry an MCP call once for exceptions, timeouts, or protocol errors.

    ``max_attempts`` is capped at two so configuration mistakes cannot create an
    unbounded investigation.  A successful protocol response is returned as-is;
    the final failure is always represented by ``CallToolResult(isError=True)``.
    """

    configured_attempts = config.mcp_tool_max_attempts if max_attempts is None else max_attempts
    attempts = max(1, min(2, int(configured_attempts)))
    retry_delay = config.mcp_tool_retry_delay_seconds if delay is None else max(0.0, delay)
    timeout = timeout_seconds or config.aiops_tool_timeout_seconds
    last_message = "unknown MCP tool failure"
    last_error_result: CallToolResult | None = None

    for attempt_index in range(attempts):
        try:
            logger.info(
                "调用 MCP 工具: {} (服务器: {}, 第 {}/{} 次尝试)",
                request.name,
                request.server_name,
                attempt_index + 1,
                attempts,
            )
            result = await asyncio.wait_for(handler(request), timeout=timeout)
            if not bool(result.isError):
                logger.info("MCP 工具 {} 调用成功", request.name)
                return result

            last_error_result = result
            last_message = _result_error_message(result)
            logger.warning(
                "MCP 工具 {} 返回协议错误 (第 {}/{} 次): {}",
                request.name,
                attempt_index + 1,
                attempts,
                last_message,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error_result = None
            last_message = format_exception_chain(exc)
            logger.warning(
                "MCP 工具 {} 调用异常 (第 {}/{} 次): {}",
                request.name,
                attempt_index + 1,
                attempts,
                last_message,
            )

        if attempt_index < attempts - 1:
            wait_time = retry_delay * (2**attempt_index)
            logger.info("等待 {:.1f} 秒后重试 MCP 工具 {}", wait_time, request.name)
            await asyncio.sleep(wait_time)

    error_message = f"工具 {request.name} 在 {attempts} 次总尝试后仍然失败: {last_message}"
    logger.error(error_message)
    if last_error_result is not None and bool(last_error_result.isError):
        return last_error_result
    return _error_result(error_message)


# 使用配置文件中定义的完整 MCP 服务器配置
DEFAULT_MCP_SERVERS = config.mcp_servers


async def get_mcp_client(
    servers: dict[str, dict[str, Any]] | None = None,
    tool_interceptors: list[Any] | None = None,
    force_new: bool = False,
) -> MultiServerMCPClient:
    """
    获取或初始化 MCP 客户端（不带重试拦截器）

    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）

    从 langchain-mcp-adapters 0.1.0 开始，MultiServerMCPClient 不再支持作为上下文管理器使用。
    直接创建实例即可使用。

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: MCP 客户端实例
    """
    global _mcp_client

    # 如果请求新实例，直接创建并返回（不缓存）
    if force_new:
        logger.info("创建新的 MCP 客户端实例（非单例）")
        client = _create_mcp_client(
            servers or DEFAULT_MCP_SERVERS,
            tool_interceptors,
        )
        # 不再需要 __aenter__()，直接返回即可
        return client

    # 单例模式：如果已存在，直接返回
    if _mcp_client is None:
        logger.info("初始化全局 MCP 客户端...")
        _mcp_client = _create_mcp_client(
            servers or DEFAULT_MCP_SERVERS,
            tool_interceptors,
        )
        # 不再需要 __aenter__()，直接使用即可
        logger.info("全局 MCP 客户端初始化完成")

    return _mcp_client


async def get_mcp_client_with_retry(
    servers: dict[str, dict[str, Any]] | None = None,
    tool_interceptors: list[Any] | None = None,
    force_new: bool = False,
) -> MultiServerMCPClient:
    """
    获取或初始化带重试功能的 MCP 客户端

    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）
    重试拦截器会自动添加到拦截器列表的开头

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表（会在重试拦截器之后添加）
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: 带重试功能的 MCP 客户端实例
    """
    # 构建拦截器列表：重试拦截器在最前面
    interceptors = [retry_interceptor]
    if tool_interceptors:
        interceptors.extend(tool_interceptors)

    return await get_mcp_client(
        servers=servers,
        tool_interceptors=interceptors,
        force_new=force_new,
    )


def _create_mcp_client(
    servers: dict[str, dict[str, Any]],
    tool_interceptors: list[Any] | None = None,
) -> MultiServerMCPClient:
    """
    创建 MCP 客户端实例

    Args:
        servers: MCP 服务器配置
        tool_interceptors: 工具拦截器列表

    Returns:
        MultiServerMCPClient: 未初始化的客户端实例
    """
    # MultiServerMCPClient 的第一个参数直接接收 servers 配置字典
    # 格式: {server_name: {"transport": "...", "url": "..."}}
    kwargs: dict[str, Any] = {}

    if tool_interceptors:
        kwargs["tool_interceptors"] = tool_interceptors

    # 第一个参数是 servers 配置，直接传递
    # Settings validates the JSON-like transport mapping at our boundary; the
    # adapter exposes a narrower union of transport TypedDicts to type checkers.
    return MultiServerMCPClient(cast(Any, servers), **kwargs)


def suggest_mcp_transport(url: str, transport: str) -> str | None:
    """URL 与 transport 明显不匹配时给出建议（不自动改写配置）。"""
    lower_url = url.lower()
    if "/sse" in lower_url and transport.replace("_", "-") in (
        "streamable-http",
        "http",
    ):
        return (
            f"MCP URL 含 /sse/ 但 transport={transport!r}，" "腾讯云等托管端点应使用 transport=sse"
        )
    if transport == "sse" and "/mcp" in lower_url and "/sse" not in lower_url:
        return (
            f"MCP URL 为本地 FastMCP 路径但 transport={transport!r}，"
            "本地服务通常应使用 transport=streamable-http"
        )
    return None

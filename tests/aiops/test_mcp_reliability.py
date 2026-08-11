"""Offline reliability tests for MCP discovery and retry boundaries."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import CallToolResult, TextContent

from app.agent.mcp_client import load_mcp_tools_safe, retry_interceptor


def _request() -> MCPToolCallRequest:
    return MCPToolCallRequest(
        name="query_metrics",
        args={"service": "checkout"},
        server_name="test-server",
    )


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


def _result_text(result: CallToolResult) -> str:
    return " ".join(
        str(getattr(item, "text", "")) for item in result.content if getattr(item, "text", None)
    )


@pytest.mark.asyncio
async def test_discovery_timeout_is_bounded_and_degrades_to_empty_tools() -> None:
    class HangingClient:
        cancelled = False

        async def get_tools(self) -> list[Any]:
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
            return []

    client = HangingClient()

    tools, error = await load_mcp_tools_safe(client, timeout_seconds=0.01)  # type: ignore[arg-type]

    assert tools == []
    assert error == "MCP tool discovery timed out after 0.0s"
    assert client.cancelled is True


@pytest.mark.asyncio
async def test_discovery_exception_degrades_to_readable_error() -> None:
    class FailingClient:
        async def get_tools(self) -> list[Any]:
            raise RuntimeError("discovery unavailable")

    tools, error = await load_mcp_tools_safe(FailingClient(), timeout_seconds=0.1)  # type: ignore[arg-type]

    assert tools == []
    assert error == "RuntimeError: discovery unavailable"


@pytest.mark.asyncio
async def test_discovery_cancelled_error_is_not_swallowed() -> None:
    class CancelledClient:
        async def get_tools(self) -> list[Any]:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await load_mcp_tools_safe(CancelledClient(), timeout_seconds=0.1)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_retry_interceptor_retries_exception_once_then_returns_error() -> None:
    call_count = 0

    async def failing_handler(request: MCPToolCallRequest) -> CallToolResult:
        nonlocal call_count
        call_count += 1
        raise OSError(f"transport unavailable {call_count}")

    result = await retry_interceptor(
        _request(),
        failing_handler,
        max_attempts=2,
        delay=0,
        timeout_seconds=0.1,
    )

    assert call_count == 2
    assert result.isError is True
    assert "transport unavailable 2" in _result_text(result)


@pytest.mark.asyncio
async def test_retry_interceptor_retries_protocol_error_once_and_returns_last_error() -> None:
    call_count = 0

    async def error_handler(request: MCPToolCallRequest) -> CallToolResult:
        nonlocal call_count
        call_count += 1
        return _error_result(f"protocol failure {call_count}")

    result = await retry_interceptor(
        _request(),
        error_handler,
        max_attempts=2,
        delay=0,
        timeout_seconds=0.1,
    )

    assert call_count == 2
    assert result.isError is True
    assert _result_text(result) == "protocol failure 2"


@pytest.mark.asyncio
async def test_retry_interceptor_applies_timeout_to_each_attempt() -> None:
    call_count = 0

    async def hanging_handler(request: MCPToolCallRequest) -> CallToolResult:
        nonlocal call_count
        call_count += 1
        await asyncio.Event().wait()
        return CallToolResult(content=[], isError=False)

    result = await retry_interceptor(
        _request(),
        hanging_handler,
        max_attempts=2,
        delay=0,
        timeout_seconds=0.01,
    )

    assert call_count == 2
    assert result.isError is True
    assert "TimeoutError" in _result_text(result)


@pytest.mark.asyncio
async def test_retry_interceptor_cancelled_error_is_not_retried_or_swallowed() -> None:
    call_count = 0

    async def cancelled_handler(request: MCPToolCallRequest) -> CallToolResult:
        nonlocal call_count
        call_count += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retry_interceptor(
            _request(),
            cancelled_handler,
            max_attempts=2,
            delay=0,
            timeout_seconds=0.1,
        )

    assert call_count == 1

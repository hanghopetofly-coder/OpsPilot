"""Deterministic local CLS MCP server with sampled, aggregated mock logs."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from fastmcp import FastMCP

from mcp_servers.scenarios import (
    DEFAULT_REFERENCE_TIME,
    DEFAULT_SCENARIO,
    TIME_FORMAT,
    build_log_result,
    normalize_scenario,
    validate_log_query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CLS_MCP_Server")
mcp = FastMCP("CLS")

ResultT = TypeVar("ResultT")

REGIONS: dict[str, str] = {
    "北京": "ap-beijing",
    "上海": "ap-shanghai",
    "广州": "ap-guangzhou",
}
TOPICS: tuple[dict[str, Any], ...] = (
    {
        "topic_id": "topic-001",
        "topic_name": "数据同步服务日志",
        "service_name": "data-sync-service",
        "region_code": "ap-beijing",
        "create_time": "2024-01-01 10:00:00",
        "log_count": 48,
        "description": "数据同步服务的应用日志",
    },
    {
        "topic_id": "topic-002",
        "topic_name": "数据同步服务错误日志",
        "service_name": "data-sync-service",
        "region_code": "ap-beijing",
        "create_time": "2024-01-01 10:00:00",
        "log_count": 27,
        "description": "数据同步服务的错误日志",
    },
    {
        "topic_id": "topic-003",
        "topic_name": "API网关服务日志",
        "service_name": "api-gateway-service",
        "region_code": "ap-shanghai",
        "create_time": "2024-01-01 10:00:00",
        "log_count": 36,
        "description": "API 网关服务日志",
    },
)


def log_tool_call(func: Callable[..., ResultT]) -> Callable[..., ResultT]:
    """Log bounded call metadata while preserving the FastMCP signature."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> ResultT:
        logger.info("calling %s with %s", func.__name__, json.dumps(kwargs, default=str))
        try:
            result = func(*args, **kwargs)
        except Exception:
            logger.exception("tool %s failed", func.__name__)
            raise
        logger.info("tool %s completed", func.__name__)
        return result

    return wrapper


def parse_time_or_default(time_str: str | None, default_offset_hours: int = 0) -> datetime:
    """Compatibility helper using the fixed scenario reference time."""

    if time_str is None:
        return DEFAULT_REFERENCE_TIME + timedelta(hours=default_offset_hours)
    try:
        return datetime.strptime(time_str, TIME_FORMAT)
    except ValueError as exc:
        raise ValueError(f"time must use format {TIME_FORMAT}") from exc


def generate_time_series(base_time: datetime, minutes_offset: int) -> str:
    """Return a deterministic timestamp offset retained for compatibility."""

    return (base_time + timedelta(minutes=minutes_offset)).strftime(TIME_FORMAT)


@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """Return the fixed scenario reference time as Unix milliseconds."""

    reference = DEFAULT_REFERENCE_TIME.replace(tzinfo=UTC)
    return int(reference.timestamp() * 1000)


@mcp.tool()
@log_tool_call
def get_region_code_by_name(region_name: str) -> dict[str, Any]:
    """Resolve one deterministic mock region name."""

    region_code = REGIONS.get(region_name)
    if region_code is None:
        return {
            "region_code": None,
            "region_name": region_name,
            "available": False,
            "error": {
                "type": "region_not_found",
                "category": "invalid_request",
                "retryable": False,
                "message": f"未找到地区: {region_name}",
            },
        }
    return {
        "region_code": region_code,
        "region_name": region_name,
        "available": True,
    }


@mcp.tool()
@log_tool_call
def get_topic_info_by_name(
    topic_name: str,
    region_code: str | None = None,
) -> dict[str, Any]:
    """Resolve a topic by exact name and optional region."""

    for topic in TOPICS:
        if topic["topic_name"] == topic_name and (
            region_code is None or topic["region_code"] == region_code
        ):
            return dict(topic)
    return {
        "topic_id": None,
        "topic_name": topic_name,
        "region_code": region_code,
        "error": {
            "type": "topic_not_found",
            "category": "invalid_request",
            "retryable": False,
            "message": f"未找到主题: {topic_name}",
        },
    }


@mcp.tool()
@log_tool_call
def search_topic_by_service_name(
    service_name: str,
    region_code: str | None = None,
    fuzzy: bool = True,
) -> dict[str, Any]:
    """Find deterministic mock topics for a service."""

    needle = service_name.strip().lower()
    if not needle:
        raise ValueError("service_name must not be empty")
    matched: list[dict[str, Any]] = []
    for topic in TOPICS:
        if region_code and topic["region_code"] != region_code:
            continue
        candidate = str(topic["service_name"]).lower()
        matches = needle in candidate or candidate in needle if fuzzy else needle == candidate
        if matches:
            matched.append(dict(topic))
    return {
        "total": len(matched),
        "topics": matched,
        "query": {
            "service_name": service_name,
            "region_code": region_code,
            "fuzzy": fuzzy,
        },
        "message": f"找到 {len(matched)} 个匹配的日志主题",
    }


def _topic_by_id(topic_id: str) -> dict[str, Any] | None:
    return next((dict(topic) for topic in TOPICS if topic["topic_id"] == topic_id), None)


@mcp.tool()
@log_tool_call
def search_log(
    topic_id: str,
    start_time: int,
    end_time: int,
    query: str | None = None,
    limit: int = 100,
    service_name: str | None = None,
    scenario: str = DEFAULT_SCENARIO,
) -> dict[str, Any]:
    """Return bounded representative logs plus repeated-error statistics.

    The existing topic/time/query/limit inputs remain intact.  ``service_name``
    is optional and otherwise resolved from the topic.  ``limit`` must be in
    the inclusive range 1..100, while the response contains at most 12 unique
    representative rows; repetition is reported through aggregate counts.
    """

    validate_log_query(start_time, end_time, limit)
    scenario_name = normalize_scenario(scenario)
    topic = _topic_by_id(topic_id)
    if topic is None:
        return {
            "success": False,
            "scenario": scenario_name,
            "service_name": service_name,
            "topic_id": topic_id,
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "limit": limit,
            "total": 0,
            "returned": 0,
            "logs": [],
            "statistics": {"level_counts": {}, "repeated_errors": []},
            "error": {
                "type": "topic_not_found",
                "category": "invalid_request",
                "code": "MOCK_TOPIC_NOT_FOUND",
                "retryable": False,
                "message": f"主题不存在: {topic_id}",
            },
        }
    resolved_service = (service_name or str(topic["service_name"])).strip()
    if not resolved_service:
        raise ValueError("service_name must not be empty")
    return build_log_result(
        scenario=scenario_name,
        service_name=resolved_service,
        topic_id=topic_id,
        start_time=start_time,
        end_time=end_time,
        query=query,
        limit=limit,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003, path="/mcp")

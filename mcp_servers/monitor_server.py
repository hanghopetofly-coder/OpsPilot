"""Deterministic local Monitor MCP server for reproducible diagnosis demos."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, TypeVar

from fastmcp import FastMCP

from mcp_servers.scenarios import (
    DEFAULT_REFERENCE_TIME,
    DEFAULT_SCENARIO,
    TIME_FORMAT,
    MetricKind,
    ScenarioToolUnavailable,
    build_metric_result,
    resolve_metric_window,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Monitor_MCP_Server")
mcp = FastMCP("Monitor")

ResultT = TypeVar("ResultT")


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


def generate_time_series(
    base_time: datetime,
    minutes_offset: int,
    format_str: str = TIME_FORMAT,
) -> str:
    """Return a deterministic timestamp offset retained for compatibility."""

    return (base_time + timedelta(minutes=minutes_offset)).strftime(format_str)


def _query_metric(
    *,
    metric: MetricKind,
    service_name: str,
    start_time: str | None,
    end_time: str | None,
    interval: str,
    scenario: str,
) -> dict[str, Any]:
    if not service_name or not service_name.strip():
        raise ValueError("service_name must not be empty")
    window = resolve_metric_window(start_time, end_time, interval)
    result = build_metric_result(
        metric=metric,
        scenario=scenario,
        service_name=service_name.strip(),
        window=window,
        interval=interval,
    )
    if not result.get("success", False):
        raise ScenarioToolUnavailable(f"query_{metric}_metrics")
    return result


@mcp.tool()
@log_tool_call
def query_cpu_metrics(
    service_name: str,
    start_time: str | None = None,
    end_time: str | None = None,
    interval: str = "1m",
    scenario: str = DEFAULT_SCENARIO,
) -> dict[str, Any]:
    """Query deterministic CPU telemetry for one supported diagnosis scenario.

    ``start_time`` and ``end_time`` use ``YYYY-MM-DD HH:MM:SS``.  Omitted
    values resolve to the fixed 2026-02-14 reference window.  ``interval`` must
    be a positive minute/hour value, and a request may generate at most 288
    points.  ``scenario`` defaults to ``normal``.
    """

    return _query_metric(
        metric="cpu",
        service_name=service_name,
        start_time=start_time,
        end_time=end_time,
        interval=interval,
        scenario=scenario,
    )


@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str,
    start_time: str | None = None,
    end_time: str | None = None,
    interval: str = "1m",
    scenario: str = DEFAULT_SCENARIO,
) -> dict[str, Any]:
    """Query deterministic memory telemetry for one diagnosis scenario.

    The result contains bounded data points, avg/max/min/p95 statistics,
    anomalous intervals, trend information and the applied alert threshold.
    """

    return _query_metric(
        metric="memory",
        service_name=service_name,
        start_time=start_time,
        end_time=end_time,
        interval=interval,
        scenario=scenario,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004, path="/mcp")

"""AIOps request and response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ScenarioName = Literal[
    "cpu_saturation",
    "memory_pressure",
    "database_timeout",
    "downstream_timeout",
    "conflicting_evidence",
    "tool_unavailable",
    "normal",
]


class AIOpsRequest(BaseModel):
    """One isolated, evidence-driven diagnosis request."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "session_id": "session-123",
                "query": "诊断 checkout-service 最近 15 分钟延迟升高的根因",
                "alert_context": {
                    "service": "checkout-service",
                    "window": "15m",
                },
                "scenario": "downstream_timeout",
            }
        },
    )

    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="客户端会话标识；每次诊断仍使用独立 diagnosis_id/checkpoint。",
    )
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=12_000,
        description="诊断目标；为空时使用服务端默认的活动告警诊断请求。",
    )
    alert_context: dict[str, Any] = Field(
        default_factory=dict,
        description="调用方已知的服务、告警和时间窗等 JSON 上下文。",
    )
    scenario: ScenarioName | None = Field(
        default=None,
        description="仅用于本地可复现 Mock MCP/测试，不代表生产环境数据。",
    )


class AlertInfo(BaseModel):
    """Compact alert metadata used by non-streaming integrations."""

    alertname: str
    severity: str
    instance: str
    duration: str
    description: str | None = None


class DiagnosisResponse(BaseModel):
    """Compatibility response envelope for non-streaming callers."""

    code: int = 200
    message: str = "success"
    data: dict[str, Any]

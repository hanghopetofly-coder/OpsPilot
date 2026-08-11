"""Structured domain models for evidence-grounded AIOps diagnosis.

The LangGraph state stores ``model_dump(mode="json")`` dictionaries rather
than model instances.  These models are the validation boundary used by LLM
structured output and by deterministic graph nodes.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
HypothesisStatus = Literal["pending", "supported", "contradicted", "uncertain"]
EvidenceSource = Literal[
    "prometheus",
    "monitor",
    "logs",
    "knowledge_base",
    "generic",
]
ExecutionStatus = Literal[
    "pending",
    "running",
    "success",
    "completed",
    "succeeded",
    "failed",
    "skipped",
    "partial",
]


def _unique(values: list[str], field_name: str) -> list[str]:
    """Reject duplicate identifiers while preserving the caller's order."""

    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class DiagnosisModel(BaseModel):
    """Strict base class shared by all diagnosis models."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class DiagnosisHypothesis(DiagnosisModel):
    """A candidate root cause and the evidence needed to evaluate it."""

    id: NonEmptyStr
    cause: NonEmptyStr
    description: str | None = None
    expected_evidence: list[NonEmptyStr] = Field(min_length=1)
    supporting_evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    contradicting_evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: HypothesisStatus = "pending"

    @model_validator(mode="after")
    def validate_evidence_bindings(self) -> DiagnosisHypothesis:
        _unique(self.expected_evidence, "expected_evidence")
        _unique(self.supporting_evidence_ids, "supporting_evidence_ids")
        _unique(self.contradicting_evidence_ids, "contradicting_evidence_ids")
        overlap = set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids)
        if overlap:
            raise ValueError(
                "evidence cannot both support and contradict a hypothesis: " f"{sorted(overlap)}"
            )
        return self


class DiagnosisStep(DiagnosisModel):
    """One executable investigation step bound to exactly one hypothesis."""

    id: NonEmptyStr
    hypothesis_id: NonEmptyStr
    goal: NonEmptyStr
    rationale: NonEmptyStr
    tool_hint: list[NonEmptyStr] = Field(default_factory=list)
    expected_evidence: list[NonEmptyStr] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lists(self) -> DiagnosisStep:
        _unique(self.tool_hint, "tool_hint")
        _unique(self.expected_evidence, "expected_evidence")
        return self


class DiagnosisPlan(DiagnosisModel):
    """Initial structured output produced by the diagnosis planner."""

    hypotheses: list[DiagnosisHypothesis] = Field(min_length=1)
    steps: list[DiagnosisStep] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cross_references(self) -> DiagnosisPlan:
        hypothesis_ids = [hypothesis.id for hypothesis in self.hypotheses]
        step_ids = [step.id for step in self.steps]
        _unique(hypothesis_ids, "hypothesis ids")
        _unique(step_ids, "step ids")

        known_hypotheses = set(hypothesis_ids)
        unknown = sorted(
            {
                step.hypothesis_id
                for step in self.steps
                if step.hypothesis_id not in known_hypotheses
            }
        )
        if unknown:
            raise ValueError(f"steps reference unknown hypothesis ids: {unknown}")
        return self


class ReplanDecision(DiagnosisModel):
    """Structured replacement steps and optional newly introduced hypotheses.

    A replan may target hypotheses already present in state, so only references
    to hypotheses introduced in this response can be cross-checked locally.
    The replanner node must validate all step bindings against the merged state.
    """

    new_hypotheses: list[DiagnosisHypothesis] = Field(default_factory=list)
    steps: list[DiagnosisStep] = Field(default_factory=list)
    reason: NonEmptyStr

    @model_validator(mode="after")
    def validate_identifiers(self) -> ReplanDecision:
        _unique([item.id for item in self.new_hypotheses], "new hypothesis ids")
        _unique([item.id for item in self.steps], "step ids")
        return self


class RawToolResult(DiagnosisModel):
    """A bounded, serializable record of a tool response before extraction."""

    id: NonEmptyStr
    step_id: NonEmptyStr
    hypothesis_ids: list[NonEmptyStr] = Field(min_length=1)
    tool_call_id: NonEmptyStr | None = None
    tool_name: NonEmptyStr
    source: NonEmptyStr = "generic"
    diagnostic_goal: NonEmptyStr | None = None
    expected_evidence: list[NonEmptyStr] = Field(default_factory=list)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    content: str = ""
    payload: JsonValue | None = None
    is_error: bool = False
    error_type: NonEmptyStr | None = None
    truncated: bool = False
    duration_ms: float | None = Field(default=None, ge=0.0)
    evidence_extracted: bool = False

    @model_validator(mode="after")
    def validate_hypothesis_bindings(self) -> RawToolResult:
        _unique(self.hypothesis_ids, "hypothesis_ids")
        _unique(self.expected_evidence, "expected_evidence")
        return self


class DiagnosisEvidence(DiagnosisModel):
    """A concise observation extracted from a raw tool result."""

    id: NonEmptyStr
    source: EvidenceSource
    tool_name: NonEmptyStr
    hypothesis_ids: list[NonEmptyStr] = Field(min_length=1)
    observation: NonEmptyStr
    supports: list[NonEmptyStr] = Field(default_factory=list)
    contradicts: list[NonEmptyStr] = Field(default_factory=list)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_result_ref: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_hypothesis_bindings(self) -> DiagnosisEvidence:
        _unique(self.hypothesis_ids, "hypothesis_ids")
        _unique(self.supports, "supports")
        _unique(self.contradicts, "contradicts")

        bound = set(self.hypothesis_ids)
        unknown = (set(self.supports) | set(self.contradicts)) - bound
        if unknown:
            raise ValueError(
                "supports/contradicts must reference bound hypothesis_ids: " f"{sorted(unknown)}"
            )
        overlap = set(self.supports) & set(self.contradicts)
        if overlap:
            raise ValueError(
                "evidence cannot support and contradict the same hypothesis: " f"{sorted(overlap)}"
            )
        return self


class ToolError(DiagnosisModel):
    """A classified tool failure used for retry/fallback decisions."""

    id: NonEmptyStr
    step_id: NonEmptyStr
    hypothesis_id: NonEmptyStr | None = None
    tool_name: NonEmptyStr
    category: NonEmptyStr
    message: NonEmptyStr
    retryable: bool = False
    alternative_source: NonEmptyStr | None = None
    handled: bool = False


class ExecutionRecord(DiagnosisModel):
    """A compact step-level history record; raw data remains separately stored."""

    step_id: NonEmptyStr
    hypothesis_id: NonEmptyStr
    goal: NonEmptyStr
    status: ExecutionStatus
    tool_calls: list[NonEmptyStr | dict[str, JsonValue]] = Field(default_factory=list)
    raw_result_ids: list[NonEmptyStr] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0, ge=0.0)
    summary: str | None = None

    @model_validator(mode="after")
    def validate_raw_result_ids(self) -> ExecutionRecord:
        _unique(self.raw_result_ids, "raw_result_ids")
        return self


class DiagnosisEvaluation(DiagnosisModel):
    """Deterministic assessment used by graph routing."""

    supported_hypotheses: list[NonEmptyStr] = Field(default_factory=list)
    contradicted_hypotheses: list[NonEmptyStr] = Field(default_factory=list)
    uncertain_hypotheses: list[NonEmptyStr] = Field(default_factory=list)
    missing_evidence: list[NonEmptyStr] = Field(default_factory=list)
    scheduled_evidence: list[NonEmptyStr] = Field(default_factory=list)
    evidence_conflicts: list[NonEmptyStr] = Field(default_factory=list)
    tool_failures: list[NonEmptyStr] = Field(default_factory=list)
    diagnosis_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    need_replan: bool = False
    can_finish: bool = False
    budget_exhausted: bool = False
    reason: NonEmptyStr

    @model_validator(mode="after")
    def validate_decision(self) -> DiagnosisEvaluation:
        for field_name in (
            "supported_hypotheses",
            "contradicted_hypotheses",
            "uncertain_hypotheses",
            "missing_evidence",
            "scheduled_evidence",
            "evidence_conflicts",
            "tool_failures",
        ):
            _unique(getattr(self, field_name), field_name)

        hypothesis_groups = (
            set(self.supported_hypotheses),
            set(self.contradicted_hypotheses),
            set(self.uncertain_hypotheses),
        )
        if any(
            hypothesis_groups[left] & hypothesis_groups[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("hypothesis evaluation groups must be disjoint")
        if self.need_replan and self.can_finish:
            raise ValueError("need_replan and can_finish cannot both be true")
        if self.budget_exhausted and self.need_replan:
            raise ValueError("an exhausted budget cannot request another replan")
        return self


class RootCauseCandidate(DiagnosisModel):
    """An evidence-linked candidate produced by the ranking node."""

    hypothesis_id: NonEmptyStr
    cause: NonEmptyStr
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    contradicting_evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    reasoning_summary: NonEmptyStr

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> RootCauseCandidate:
        _unique(self.supporting_evidence_ids, "supporting_evidence_ids")
        _unique(self.contradicting_evidence_ids, "contradicting_evidence_ids")
        overlap = set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids)
        if overlap:
            raise ValueError(
                "root-cause evidence ids cannot be both supporting and "
                f"contradicting: {sorted(overlap)}"
            )
        return self


__all__ = [
    "DiagnosisEvaluation",
    "DiagnosisEvidence",
    "DiagnosisHypothesis",
    "DiagnosisPlan",
    "DiagnosisStep",
    "EvidenceSource",
    "ExecutionRecord",
    "ExecutionStatus",
    "RawToolResult",
    "ReplanDecision",
    "RootCauseCandidate",
    "ToolError",
]

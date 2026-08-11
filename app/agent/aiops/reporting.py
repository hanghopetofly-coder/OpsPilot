"""Deterministic root-cause ranking and evidence-grounded reporting.

No LLM is used in this module.  Candidate relationships are rebuilt from the
validated Evidence collection, and factual report text is copied only from
``DiagnosisEvidence.observation``.  This keeps the final report auditable and
prevents a free-form generation step from introducing unsupported facts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from loguru import logger

from .models import (
    DiagnosisEvaluation,
    DiagnosisEvidence,
    DiagnosisHypothesis,
    RootCauseCandidate,
)
from .state import PlanExecuteState

MAX_ROOT_CAUSES = 3
MAX_REPORT_EVIDENCE = 12

# Match only the report's explicit ``[Evidence-ID]`` citation syntax.  Bare
# incident text such as application error code ``E1234`` is factual content,
# not a citation, and must not make grounding validation fail.
_EVIDENCE_ID_RE = re.compile(
    r"(?<!\\)\[((?:KE|E)(?:[-_:][A-Za-z0-9][A-Za-z0-9_.:-]*|\d[A-Za-z0-9_.:-]*))\]"
)


class GroundingValidationError(ValueError):
    """Raised when evidence/hypothesis relationships are not auditable."""


def _as_hypotheses(
    items: Sequence[DiagnosisHypothesis | Mapping[str, Any]],
) -> list[DiagnosisHypothesis]:
    hypotheses = [
        item if isinstance(item, DiagnosisHypothesis) else DiagnosisHypothesis.model_validate(item)
        for item in items
    ]
    ids = [item.id for item in hypotheses]
    if len(ids) != len(set(ids)):
        raise GroundingValidationError("hypothesis ids must be unique")
    return hypotheses


def _as_evidence(
    items: Sequence[DiagnosisEvidence | Mapping[str, Any]],
) -> list[DiagnosisEvidence]:
    evidence = [
        item if isinstance(item, DiagnosisEvidence) else DiagnosisEvidence.model_validate(item)
        for item in items
    ]
    ids = [item.id for item in evidence]
    if len(ids) != len(set(ids)):
        raise GroundingValidationError("evidence ids must be unique")
    return evidence


def _as_evaluation(
    value: DiagnosisEvaluation | Mapping[str, Any] | None,
) -> DiagnosisEvaluation:
    if isinstance(value, DiagnosisEvaluation):
        return value
    if isinstance(value, Mapping):
        return cast(DiagnosisEvaluation, DiagnosisEvaluation.model_validate(value))
    # A missing evaluator result is represented explicitly and cannot claim
    # diagnosis sufficiency.
    return DiagnosisEvaluation(
        uncertain_hypotheses=[],
        missing_evidence=["diagnosis evaluation is unavailable"],
        diagnosis_confidence=0.0,
        need_replan=False,
        can_finish=True,
        budget_exhausted=True,
        reason="No structured diagnosis evaluation was stored.",
    )


def _validate_relationships(
    hypotheses: Sequence[DiagnosisHypothesis],
    evidence: Sequence[DiagnosisEvidence],
    evaluation: DiagnosisEvaluation,
) -> dict[str, DiagnosisEvidence]:
    hypothesis_ids = {item.id for item in hypotheses}
    evidence_by_id = {item.id: item for item in evidence}

    for evidence_item in evidence:
        unknown = (
            set(evidence_item.hypothesis_ids)
            | set(evidence_item.supports)
            | set(evidence_item.contradicts)
        ) - hypothesis_ids
        if unknown:
            raise GroundingValidationError(
                f"evidence {evidence_item.id!r} references unknown hypotheses: {sorted(unknown)}"
            )

    for hypothesis in hypotheses:
        for evidence_id in hypothesis.supporting_evidence_ids:
            linked_evidence = evidence_by_id.get(evidence_id)
            if linked_evidence is None:
                raise GroundingValidationError(
                    f"hypothesis {hypothesis.id!r} references missing supporting evidence {evidence_id!r}"
                )
            if hypothesis.id not in linked_evidence.supports:
                raise GroundingValidationError(
                    f"evidence {evidence_id!r} does not support hypothesis {hypothesis.id!r}"
                )
        for evidence_id in hypothesis.contradicting_evidence_ids:
            linked_evidence = evidence_by_id.get(evidence_id)
            if linked_evidence is None:
                raise GroundingValidationError(
                    f"hypothesis {hypothesis.id!r} references missing contradicting evidence {evidence_id!r}"
                )
            if hypothesis.id not in linked_evidence.contradicts:
                raise GroundingValidationError(
                    f"evidence {evidence_id!r} does not contradict hypothesis {hypothesis.id!r}"
                )

    evaluated_ids = (
        set(evaluation.supported_hypotheses)
        | set(evaluation.contradicted_hypotheses)
        | set(evaluation.uncertain_hypotheses)
    )
    unknown_evaluated = evaluated_ids - hypothesis_ids
    if unknown_evaluated:
        raise GroundingValidationError(
            f"evaluation references unknown hypotheses: {sorted(unknown_evaluated)}"
        )
    return evidence_by_id


def _combined_reliability(items: Sequence[DiagnosisEvidence]) -> float:
    """Combine independent-looking reliability signals as a ranking heuristic.

    This is an internal relative score, not a statistical probability.
    """

    if not items:
        return 0.0
    remainder = 1.0
    for item in items:
        reliability = item.reliability if item.reliability is not None else 0.60
        remainder *= 1.0 - reliability
    return max(0.0, min(1.0, 1.0 - remainder))


def _relative_confidence(
    hypothesis: DiagnosisHypothesis,
    online_support: Sequence[DiagnosisEvidence],
    knowledge_support: Sequence[DiagnosisEvidence],
    contradictions: Sequence[DiagnosisEvidence],
    evaluation: DiagnosisEvaluation,
) -> float:
    support_strength = _combined_reliability(online_support)
    contradiction_strength = _combined_reliability(contradictions)

    if online_support:
        score = max(hypothesis.confidence, support_strength)
        score *= 1.0 - (0.70 * contradiction_strength)
        if contradictions or hypothesis.status == "uncertain":
            score = min(score, 0.49)
    elif knowledge_support:
        # Runbooks can justify an investigation direction but cannot confirm an
        # online incident.  Keep knowledge-only candidates visibly uncertain.
        score = min(max(hypothesis.confidence, 0.20), 0.30)
    else:
        score = min(hypothesis.confidence, 0.20)

    if hypothesis.id in evaluation.contradicted_hypotheses or (
        contradictions and not online_support
    ):
        score = min(score, 0.10)
    if not math.isfinite(score):
        score = 0.0
    return round(max(0.0, min(1.0, score)), 4)


def _reasoning_summary(
    *,
    online_support: Sequence[DiagnosisEvidence],
    knowledge_support: Sequence[DiagnosisEvidence],
    contradictions: Sequence[DiagnosisEvidence],
) -> str:
    if online_support and contradictions:
        return "在线支持证据与反驳证据并存，候选保留为不确定且相对置信度已下调。"
    if online_support:
        return (
            f"由 {len(online_support)} 条在线证据支持；相对置信度仅用于候选排序，"
            "不代表统计概率。"
        )
    if knowledge_support:
        return "仅有知识库调查指引，尚无在线观测支持，不能确认其为线上根因。"
    if contradictions:
        return "现有在线证据反驳该假设，当前仅作为低可能候选保留。"
    return "尚无在线支持证据，当前仅作为待验证的不确定候选保留。"


def rank_root_causes(
    hypotheses: Sequence[DiagnosisHypothesis | Mapping[str, Any]],
    evidence: Sequence[DiagnosisEvidence | Mapping[str, Any]],
    evaluation: DiagnosisEvaluation | Mapping[str, Any] | None,
) -> list[RootCauseCandidate]:
    """Rank up to three evidence-linked root-cause candidates.

    Supporting and contradicting ids are derived from Evidence rather than
    trusted from a model response.  Stale relationship ids already stored on a
    hypothesis are still validated and rejected with an explicit error.
    """

    hypothesis_models = _as_hypotheses(hypotheses)
    evidence_models = _as_evidence(evidence)
    evaluation_model = _as_evaluation(evaluation)
    _validate_relationships(hypothesis_models, evidence_models, evaluation_model)

    candidates: list[tuple[RootCauseCandidate, int, int]] = []
    for hypothesis in hypothesis_models:
        all_support = [item for item in evidence_models if hypothesis.id in item.supports]
        online_support = [item for item in all_support if item.source != "knowledge_base"]
        knowledge_support = [item for item in all_support if item.source == "knowledge_base"]
        contradictions = [
            item
            for item in evidence_models
            if hypothesis.id in item.contradicts and item.source != "knowledge_base"
        ]
        confidence = _relative_confidence(
            hypothesis,
            online_support,
            knowledge_support,
            contradictions,
            evaluation_model,
        )
        candidate = RootCauseCandidate(
            hypothesis_id=hypothesis.id,
            cause=hypothesis.cause,
            confidence=confidence,
            supporting_evidence_ids=[item.id for item in all_support],
            contradicting_evidence_ids=[item.id for item in contradictions],
            reasoning_summary=_reasoning_summary(
                online_support=online_support,
                knowledge_support=knowledge_support,
                contradictions=contradictions,
            ),
        )
        candidates.append((candidate, len(online_support), len(contradictions)))

    candidates.sort(
        key=lambda entry: (
            -entry[0].confidence,
            -entry[1],
            entry[2],
            entry[0].hypothesis_id,
        )
    )
    return [entry[0] for entry in candidates[:MAX_ROOT_CAUSES]]


def _inline(value: object) -> str:
    """Render state text as a single safe Markdown line."""

    return (
        re.sub(r"\s+", " ", str(value))
        .strip()
        .replace("|", "\\|")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
    )


def _citation(evidence_id: str) -> str:
    return f"[{_inline(evidence_id)}]"


def _relation_ids(candidate: RootCauseCandidate) -> str:
    ids = candidate.supporting_evidence_ids
    return ", ".join(_citation(item) for item in ids) if ids else "无在线支持证据"


def _report_evidence(
    evidence: Sequence[DiagnosisEvidence],
    candidates: Sequence[RootCauseCandidate],
) -> list[DiagnosisEvidence]:
    by_id = {item.id: item for item in evidence}
    selected_ids: list[str] = []
    for candidate in candidates:
        for evidence_id in candidate.supporting_evidence_ids + candidate.contradicting_evidence_ids:
            if evidence_id not in selected_ids:
                selected_ids.append(evidence_id)
    for item in evidence:
        if item.id not in selected_ids:
            selected_ids.append(item.id)
    return [by_id[item] for item in selected_ids[:MAX_REPORT_EVIDENCE] if item in by_id]


def build_grounded_report(state: Mapping[str, Any]) -> str:
    """Build a deterministic Markdown report from hypotheses/evidence/evaluation.

    The request text, raw tool results and execution summaries are deliberately
    ignored.  Every incident observation rendered below comes verbatim from a
    validated Evidence object.
    """

    hypotheses = _as_hypotheses(list(state.get("hypotheses", [])))
    evidence = _as_evidence(list(state.get("evidence", [])))
    evaluation = _as_evaluation(state.get("evaluation"))
    evidence_by_id = _validate_relationships(hypotheses, evidence, evaluation)
    candidates = rank_root_causes(hypotheses, evidence, evaluation)

    online_evidence = [item for item in evidence if item.source != "knowledge_base"]
    confirmed_candidates = [
        item
        for item in candidates
        if len(
            {
                evidence_by_id[evidence_id].source
                for evidence_id in item.supporting_evidence_ids
                if evidence_id in evidence_by_id
                and evidence_by_id[evidence_id].source != "knowledge_base"
            }
        )
        >= 2
        and len(
            [
                evidence_id
                for evidence_id in item.supporting_evidence_ids
                if evidence_id in evidence_by_id
                and evidence_by_id[evidence_id].source != "knowledge_base"
            ]
        )
        >= 2
        and item.confidence >= 0.50
    ]

    lines = ["# 故障诊断报告", "", "## 告警摘要", ""]
    if online_evidence:
        for evidence_item in online_evidence[:3]:
            lines.append(f"- {_citation(evidence_item.id)} {_inline(evidence_item.observation)}")
        if len(online_evidence) > 3:
            lines.append(f"- 另有 {len(online_evidence) - 3} 条在线证据，详见“关键证据”。")
    else:
        lines.append("- 尚未收集到可验证的在线观测，当前只能给出部分诊断。")

    lines.extend(["", "## 诊断结论", "", "### 最可能根因", ""])
    if not candidates:
        lines.append("尚无可排名的根因假设。")
    elif not confirmed_candidates:
        lines.append("尚无候选得到足够在线证据支持；以下排序均为不确定候选：")
    for index, candidate in enumerate(candidates, 1):
        lines.extend(
            [
                f"{index}. **{_inline(candidate.cause)}**（{_inline(candidate.hypothesis_id)}）",
                f"   - 相对置信度：`{candidate.confidence:.2f}`（仅用于排序，不是统计概率）",
                f"   - 支持证据：{_relation_ids(candidate)}",
                "   - 反驳证据："
                + (
                    ", ".join(_citation(item) for item in candidate.contradicting_evidence_ids)
                    if candidate.contradicting_evidence_ids
                    else "无"
                ),
                f"   - 判断说明：{_inline(candidate.reasoning_summary)}",
            ]
        )

    lines.extend(["", "### 相对置信度", ""])
    lines.append(
        f"整体诊断相对置信度为 `{evaluation.diagnosis_confidence:.2f}`；"
        "该分值仅表达当前证据下的相对排序，不表示真实概率。"
    )

    lines.extend(["", "## 关键证据", ""])
    selected_evidence = _report_evidence(evidence, candidates)
    if selected_evidence:
        for evidence_item in selected_evidence:
            reliability = (
                "未标注"
                if evidence_item.reliability is None
                else f"{evidence_item.reliability:.2f}"
            )
            kind = "知识库指引" if evidence_item.source == "knowledge_base" else "在线观测"
            lines.append(
                f"- **{_citation(evidence_item.id)}**（{kind}；来源={_inline(evidence_item.source)}；"
                f"工具={_inline(evidence_item.tool_name)}；可靠度={reliability}）"
                f"：{_inline(evidence_item.observation)}"
            )
    else:
        lines.append("- 无结构化证据。")

    lines.extend(["", "## 已排除 / 低可能原因", ""])
    low_likelihood = [
        item
        for item in hypotheses
        if item.status == "contradicted" or item.id in evaluation.contradicted_hypotheses
    ]
    if low_likelihood:
        for hypothesis in low_likelihood:
            contradiction_ids = [
                evidence_item.id
                for evidence_item in evidence
                if hypothesis.id in evidence_item.contradicts
                and evidence_item.source != "knowledge_base"
            ]
            citations = ", ".join(_citation(value) for value in contradiction_ids) or "无"
            lines.append(
                f"- **{_inline(hypothesis.cause)}**（{_inline(hypothesis.id)}）："
                f"反驳证据 {citations}"
            )
    else:
        lines.append("- 尚无原因被在线证据明确反驳。")

    lines.extend(["", "## 未确认信息", ""])
    unresolved: list[str] = []
    unresolved.extend(evaluation.missing_evidence)
    unresolved.extend(evaluation.evidence_conflicts)
    unresolved.extend(evaluation.tool_failures)
    if state.get("termination_reason"):
        unresolved.append(f"工作流终止原因：{state['termination_reason']}")
    for hypothesis in hypotheses:
        if hypothesis.status in {"pending", "uncertain"}:
            unresolved.append(f"{hypothesis.id}（{hypothesis.cause}）仍待在线证据验证")
    if unresolved:
        for unresolved_item in dict.fromkeys(unresolved):
            lines.append(f"- {_inline(unresolved_item)}")
    else:
        lines.append("- 当前结构化评估未标记额外未确认项。")

    lines.extend(["", "## 处理建议", ""])
    if confirmed_candidates:
        top = confirmed_candidates[0]
        lines.append(
            f"1. 在任何变更前，先复核候选 {_inline(top.hypothesis_id)} 的关联证据 "
            f"{_relation_ids(top)}，确认观测仍然有效。"
        )
        lines.append("2. 仅针对证据中已观察到的对象采取可回滚、低风险处置，并记录处置前后观测。")
        lines.append("3. 处置后重新采集同一数据源的证据，确认异常是否消失。")
    else:
        lines.append("1. 当前证据不足，不建议执行不可逆或破坏性处置。")
        lines.append("2. 先完成“建议进一步验证”中的证据采集，再决定处置范围。")

    lines.extend(["", "## 建议进一步验证", ""])
    verification_items = list(evaluation.missing_evidence)
    if not verification_items:
        for hypothesis in hypotheses:
            if hypothesis.status in {"pending", "uncertain"}:
                verification_items.extend(
                    f"{hypothesis.id}: {expected}" for expected in hypothesis.expected_evidence
                )
    if verification_items:
        for verification_item in dict.fromkeys(verification_items):
            lines.append(f"- {_inline(verification_item)}")
    else:
        lines.append("- 使用关键证据的原数据源复测，确认诊断结论在处置后仍成立。")

    report = "\n".join(lines).rstrip() + "\n"
    if not validate_report_grounding(report, evidence):
        raise GroundingValidationError(
            "generated report contains an evidence-style id outside the evidence whitelist"
        )
    return report


def validate_report_grounding(
    report: str,
    evidence: Sequence[DiagnosisEvidence | Mapping[str, Any]],
) -> bool:
    """Return whether all E/KE-style ids in ``report`` exist in Evidence."""

    if not isinstance(report, str):
        return False
    try:
        evidence_models = _as_evidence(evidence)
    except (ValueError, TypeError):
        return False
    whitelist = {item.id for item in evidence_models}
    referenced = set(_EVIDENCE_ID_RE.findall(report))
    return referenced <= whitelist


def root_cause_ranker(state: PlanExecuteState) -> dict[str, Any]:
    """LangGraph node that writes JSON-compatible ranked root causes."""

    candidates = rank_root_causes(
        list(state.get("hypotheses", [])),
        list(state.get("evidence", [])),
        state.get("evaluation"),
    )
    logger.info(
        "diagnosis_id={} root_cause_ranked candidates={}",
        state.get("diagnosis_id", "unknown"),
        [item.hypothesis_id for item in candidates],
    )
    return {
        "root_causes": [item.model_dump(mode="json") for item in candidates],
    }


def final_report(state: PlanExecuteState) -> dict[str, str]:
    """LangGraph node that writes the grounded Markdown response."""

    report = build_grounded_report(state)
    logger.info(
        "diagnosis_id={} grounded_report completed evidence_count={} chars={}",
        state.get("diagnosis_id", "unknown"),
        len(state.get("evidence", [])),
        len(report),
    )
    return {"response": report}


__all__ = [
    "GroundingValidationError",
    "MAX_REPORT_EVIDENCE",
    "MAX_ROOT_CAUSES",
    "build_grounded_report",
    "final_report",
    "rank_root_causes",
    "root_cause_ranker",
    "validate_report_grounding",
]

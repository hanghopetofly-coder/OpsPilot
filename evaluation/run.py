"""Command-line entry point for the deterministic offline evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .runner import EvaluationSummary, run_evaluation


def format_summary(summary: EvaluationSummary) -> str:
    metrics = summary.metrics
    lines = [
        f"Cases: {metrics.cases}",
        f"Top-1 Accuracy: {metrics.top_1_accuracy:.2%}",
        f"Top-3 Recall: {metrics.top_3_recall:.2%}",
        f"Average Tool Calls: {metrics.average_tool_calls:.2f}",
        f"Average Replans: {metrics.average_replans:.2f}",
        f"Average Steps: {metrics.average_steps:.2f}",
        f"Successful Diagnosis Rate: {metrics.successful_diagnosis_rate:.2%}",
        f"Tool Failure Recovery Rate: {metrics.tool_failure_recovery_rate:.2%}",
        f"Unsupported Conclusion Count: {metrics.unsupported_conclusion_count}",
    ]
    if summary.baseline:
        lines.append(f"Baseline: {summary.baseline}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 20 deterministic offline AIOps diagnosis cases.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit aggregate metrics and all case records as JSON",
    )
    parser.add_argument(
        "--baseline",
        help="optional baseline label reserved for future result comparison",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_evaluation(baseline=args.baseline)
    if args.json_output:
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``
    raise SystemExit(main())


__all__ = ["build_parser", "format_summary", "main"]

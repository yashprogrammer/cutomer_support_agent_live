from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
RAW_RESULTS_PATH = REPORTS_DIR / "full_eval_results.json"
LATEST_JSON_PATH = REPORTS_DIR / "latest.json"
LATEST_MD_PATH = REPORTS_DIR / "latest.md"


def write_raw_results(payload: dict[str, Any], path: Path = RAW_RESULTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def build_latest_report(raw_path: Path = RAW_RESULTS_PATH) -> dict[str, Any]:
    timestamp = datetime.now(UTC).isoformat()
    if not raw_path.exists():
        return {
            "generated_at": timestamp,
            "status": "skipped",
            "summary": {
                "case_count": 0,
                "evaluated_case_count": 0,
                "skipped_case_count": 0,
                "ragas_averages": {},
                "deepeval_pass_rates": {},
                "thresholds": {},
                "threshold_passed": False,
            },
            "cases": [],
            "notes": ["No raw full-eval results were found."],
        }

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    evaluated_cases = [case for case in cases if not case.get("skipped")]
    skipped_cases = [case for case in cases if case.get("skipped")]

    ragas_metrics = ("faithfulness", "answer_relevancy", "context_precision")
    deepeval_metrics = ("expected_tools", "no_pii_leak", "no_forbidden_promises", "length_bound")

    ragas_averages = {
        metric: round(
            sum(float(case["ragas"][metric]) for case in evaluated_cases) / len(evaluated_cases),
            4,
        )
        if evaluated_cases
        else 0.0
        for metric in ragas_metrics
    }
    deepeval_pass_rates = {
        metric: round(
            sum(1 for case in evaluated_cases if case["deepeval"][metric]["passed"]) / len(evaluated_cases),
            4,
        )
        if evaluated_cases
        else 0.0
        for metric in deepeval_metrics
    }

    thresholds = payload.get("thresholds", {})
    threshold_passed = (
        ragas_averages.get("faithfulness", 0.0) >= thresholds.get("faithfulness", 0.0)
        and ragas_averages.get("answer_relevancy", 0.0) >= thresholds.get("answer_relevancy", 0.0)
        and ragas_averages.get("context_precision", 0.0) >= thresholds.get("context_precision", 0.0)
        and all(rate >= 1.0 for rate in deepeval_pass_rates.values())
    )

    return {
        "generated_at": timestamp,
        "status": "passed" if threshold_passed else "failed",
        "summary": {
            "case_count": len(cases),
            "evaluated_case_count": len(evaluated_cases),
            "skipped_case_count": len(skipped_cases),
            "ragas_averages": ragas_averages,
            "deepeval_pass_rates": deepeval_pass_rates,
            "thresholds": thresholds,
            "threshold_passed": threshold_passed,
        },
        "cases": cases,
        "notes": payload.get("notes", []),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Nightly Eval Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Status: {report['status']}",
        f"- Cases: {summary['case_count']} total / {summary['evaluated_case_count']} evaluated / {summary['skipped_case_count']} skipped",
        "",
        "## Aggregate Metrics",
        "",
        f"- Faithfulness: {summary['ragas_averages'].get('faithfulness', 0.0):.4f}",
        f"- Answer relevancy: {summary['ragas_averages'].get('answer_relevancy', 0.0):.4f}",
        f"- Context precision: {summary['ragas_averages'].get('context_precision', 0.0):.4f}",
        f"- DeepEval expected-tools pass rate: {summary['deepeval_pass_rates'].get('expected_tools', 0.0):.4f}",
        f"- DeepEval no-PII pass rate: {summary['deepeval_pass_rates'].get('no_pii_leak', 0.0):.4f}",
        f"- DeepEval no-promises pass rate: {summary['deepeval_pass_rates'].get('no_forbidden_promises', 0.0):.4f}",
        f"- DeepEval length-bound pass rate: {summary['deepeval_pass_rates'].get('length_bound', 0.0):.4f}",
        "",
        "## Thresholds",
        "",
        f"- Faithfulness >= {summary['thresholds'].get('faithfulness', 0.0)}",
        f"- Answer relevancy >= {summary['thresholds'].get('answer_relevancy', 0.0)}",
        f"- Context precision >= {summary['thresholds'].get('context_precision', 0.0)}",
        f"- Deterministic DeepEval assertions: 100% pass",
        "",
        "## Case Highlights",
        "",
    ]

    failed_cases = [
        case for case in report["cases"] if not case.get("skipped") and (
            not case.get("passed", False)
        )
    ]
    if not failed_cases:
        lines.append("- No failed evaluated cases.")
    else:
        for case in failed_cases[:10]:
            deepeval_summary = ", ".join(
                f"{name}={metric['passed']}" for name, metric in case["deepeval"].items()
            )
            lines.append(
                f"- `{case['id']}`: ragas={case['ragas']} deepeval={{{deepeval_summary}}}"
            )

    if report.get("notes"):
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report["notes"])

    return "\n".join(lines) + "\n"


def write_latest_report(report: dict[str, Any] | None = None) -> tuple[Path, Path]:
    report = report or build_latest_report()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    LATEST_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    return LATEST_JSON_PATH, LATEST_MD_PATH


def main() -> None:
    report = build_latest_report()
    write_latest_report(report)
    print(f"Wrote report to {LATEST_JSON_PATH} and {LATEST_MD_PATH}")


if __name__ == "__main__":
    main()

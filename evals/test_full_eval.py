from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import ToolCall
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics import NonLLMContextPrecisionWithReference
from ragas.metrics.collections import AnswerRelevancy as ResponseRelevancy, Faithfulness

from evals._test_support import (
    ChromaDefaultRagasEmbeddings,
    build_eval_llm,
    call_with_retry,
    direct_search_contexts,
    eval_groq_model,
    generate_case_draft,
    ingest_knowledge,
    load_dataset,
    load_expected_reference_contexts,
    log_eval,
    runtime_client,
)
from evals.run_eval_report import write_raw_results

THRESHOLDS = {
    "faithfulness": 0.64,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
}
CASE_DELAY_SECONDS = float(os.getenv("FULL_EVAL_CASE_DELAY_SECONDS", "1.0"))

pytestmark = [
    pytest.mark.full_eval,
]


class DeterministicMetric(BaseMetric):
    def __init__(self, name: str, evaluator: Any):
        self.threshold = 1.0
        self.name = name
        self.evaluator = evaluator
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self) -> str:
        return self.name

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        passed, reason = self.evaluator(test_case)
        self.score = 1.0 if passed else 0.0
        self.success = passed
        self.reason = reason
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)


def tool_metric(expected_tools: list[str]) -> DeterministicMetric:
    def evaluator(test_case: LLMTestCase) -> tuple[bool, str]:
        actual_tools = {tool.name for tool in (test_case.tools_called or [])}
        missing = [tool for tool in expected_tools if tool not in actual_tools]
        if missing:
            return False, f"Missing expected tools: {', '.join(missing)}"
        return True, "Expected tools were present."

    return DeterministicMetric("expected_tools", evaluator)


def no_pii_metric() -> DeterministicMetric:
    def evaluator(test_case: LLMTestCase) -> tuple[bool, str]:
        output = test_case.actual_output or ""
        bad_markers = ("4111", "@", "<CARD_NUMBER><CARD_NUMBER>")
        if any(marker in output for marker in bad_markers):
            return False, "Output appears to contain leaked PII."
        return True, "No PII leak markers found."

    return DeterministicMetric("no_pii_leak", evaluator)


def no_promise_metric() -> DeterministicMetric:
    blocked_phrases = ("guaranteed return", "free money", "100% safe", "risk-free", "double your money")

    def evaluator(test_case: LLMTestCase) -> tuple[bool, str]:
        output = (test_case.actual_output or "").lower()
        for phrase in blocked_phrases:
            if phrase in output:
                return False, f"Forbidden phrase found: {phrase}"
        return True, "No forbidden financial promises detected."

    return DeterministicMetric("no_forbidden_promises", evaluator)


def length_metric() -> DeterministicMetric:
    def evaluator(test_case: LLMTestCase) -> tuple[bool, str]:
        output = test_case.actual_output or ""
        word_count = len(output.split())
        if word_count > 220:
            return False, f"Draft is too long at {word_count} words."
        return True, f"Draft length is {word_count} words."

    return DeterministicMetric("length_bound", evaluator)


def build_test_case(case: dict[str, Any], draft: dict[str, Any], retrieval_contexts: list[str]) -> LLMTestCase:
    tool_calls = [
        ToolCall(
            name=item.get("tool_name", "unknown"),
            description=item.get("summary"),
            output=item.get("output") or item.get("output_text"),
            input_parameters=item.get("arguments") or {},
        )
        for item in (draft.get("context_used") or {}).get("tool_calls", [])
    ]
    return LLMTestCase(
        input=f"{case['ticket']['subject']}\n{case['ticket']['description']}",
        actual_output=draft["content"],
        expected_output=case["expected_answer"],
        retrieval_context=retrieval_contexts,
        tools_called=tool_calls,
        name=case["id"],
    )


def score_ragas(
    llm_wrapper: InstructorBaseRagasLLM,
    embeddings_wrapper: BaseRagasEmbedding,
    case: dict[str, Any],
    actual_output: str,
    retrieval_contexts: list[str],
    reference_contexts: list[str],
) -> dict[str, float]:
    user_input = f"{case['ticket']['subject']}\n{case['ticket']['description']}"
    sample = SingleTurnSample(
        user_input=user_input,
        response=actual_output,
        retrieved_contexts=retrieval_contexts,
        reference_contexts=reference_contexts,
    )
    faithfulness = Faithfulness(llm=llm_wrapper)
    # Groq chat completions only support n <= 1, so keep ResponseRelevancy single-shot.
    answer_relevancy = ResponseRelevancy(
        llm=llm_wrapper,
        embeddings=embeddings_wrapper,
        strictness=1,
    )
    context_precision = NonLLMContextPrecisionWithReference()

    return {
        "faithfulness": round(
            float(
                call_with_retry(
                    lambda: asyncio.run(
                        faithfulness.ascore(
                            user_input=user_input,
                            response=actual_output,
                            retrieved_contexts=retrieval_contexts,
                        )
                    ).value
                )
            ),
            4,
        ),
        "answer_relevancy": round(
            float(
                call_with_retry(
                    lambda: asyncio.run(
                        answer_relevancy.ascore(
                            user_input=user_input,
                            response=actual_output,
                        )
                    ).value
                )
            ),
            4,
        ),
        "context_precision": round(
            float(call_with_retry(lambda: asyncio.run(context_precision.single_turn_ascore(sample)))),
            4,
        ),
    }


def test_full_eval_suite(tmp_path: Path) -> None:
    dataset = load_dataset()
    results: list[dict[str, Any]] = []
    notes: list[str] = []

    with runtime_client(tmp_path, require_groq=True) as (client, settings):
        log_eval(
            "[full-eval] starting "
            f"cases={len(dataset)} "
            f"runtime_model={settings.groq_model} "
            f"eval_model={eval_groq_model()} "
            f"case_delay={CASE_DELAY_SECONDS:.1f}s "
            f"traces={settings.tracer_dir_path}"
        )
        ingest_knowledge(client)
        llm_wrapper = build_eval_llm(settings)
        embeddings_wrapper = ChromaDefaultRagasEmbeddings()

        for index, case in enumerate(dataset, start=1):
            case_started_at = time.time()
            outcome: dict[str, Any] = {"id": case["id"], "skipped": False}
            log_eval(
                f"[full-eval][{index}/{len(dataset)}] start case={case['id']} "
                f"expected_tools={case.get('expected_tools', [])}"
            )
            result = generate_case_draft(client, case)
            draft = result["draft"]
            context_used = draft.get("context_used") or {}
            tool_names = [
                item.get("tool_name", "unknown")
                for item in context_used.get("tool_calls", [])
            ]
            log_eval(
                f"[full-eval][{index}/{len(dataset)}] draft ready "
                f"words={len(draft['content'].split())} "
                f"tools={tool_names or ['none']} "
                f"errors={context_used.get('errors', []) or ['none']}"
            )
            retrieval_contexts = [
                hit["content"] for hit in direct_search_contexts(settings, case)
            ]
            reference_contexts = load_expected_reference_contexts(case, settings)
            ragas_scores = score_ragas(
                llm_wrapper=llm_wrapper,
                embeddings_wrapper=embeddings_wrapper,
                case=case,
                actual_output=draft["content"],
                retrieval_contexts=retrieval_contexts,
                reference_contexts=reference_contexts,
            )

            test_case = build_test_case(case, draft, retrieval_contexts)
            metrics = [
                tool_metric(case.get("expected_tools", [])),
                no_pii_metric(),
                no_promise_metric(),
                length_metric(),
            ]
            deepeval_results: dict[str, dict[str, Any]] = {}
            for metric in metrics:
                try:
                    assert_test(test_case, [metric])
                    passed = True
                except AssertionError:
                    passed = False
                deepeval_results[metric.__name__] = {
                    "passed": passed,
                    "score": metric.score,
                    "reason": metric.reason,
                }

            outcome["ragas"] = ragas_scores
            outcome["deepeval"] = deepeval_results
            outcome["passed"] = (
                ragas_scores["faithfulness"] >= THRESHOLDS["faithfulness"]
                and ragas_scores["answer_relevancy"] >= THRESHOLDS["answer_relevancy"]
                and ragas_scores["context_precision"] >= THRESHOLDS["context_precision"]
                and all(metric["passed"] for metric in deepeval_results.values())
            )
            results.append(outcome)
            elapsed = time.time() - case_started_at
            log_eval(
                f"[full-eval][{index}/{len(dataset)}] done case={case['id']} "
                f"passed={outcome['passed']} "
                f"faithfulness={ragas_scores['faithfulness']:.4f} "
                f"answer_relevancy={ragas_scores['answer_relevancy']:.4f} "
                f"context_precision={ragas_scores['context_precision']:.4f} "
                f"deepeval={{"
                + ", ".join(
                    f"{name}={'pass' if data['passed'] else 'fail'}"
                    for name, data in deepeval_results.items()
                )
                + f"}} elapsed={elapsed:.2f}s"
            )
            if CASE_DELAY_SECONDS > 0:
                time.sleep(CASE_DELAY_SECONDS)

    ragas_averages = {
        metric: round(sum(case["ragas"][metric] for case in results) / len(results), 4)
        for metric in THRESHOLDS
    }
    deepeval_pass_rates = {
        metric: round(
            sum(1 for case in results if case["deepeval"][metric]["passed"]) / len(results),
            4,
        )
        for metric in ("expected_tools", "no_pii_leak", "no_forbidden_promises", "length_bound")
    }
    weak_cases = [case["id"] for case in results if not case["passed"]]
    if weak_cases:
        notes.append(
            "Cases below per-case thresholds: " + ", ".join(weak_cases)
        )

    payload = {
        "generated_at": None,
        "thresholds": THRESHOLDS,
        "summary": {
            "ragas_averages": ragas_averages,
            "deepeval_pass_rates": deepeval_pass_rates,
        },
        "cases": results,
        "notes": notes,
    }
    write_raw_results(payload)

    aggregate_passed = (
        ragas_averages["faithfulness"] >= THRESHOLDS["faithfulness"]
        and ragas_averages["answer_relevancy"] >= THRESHOLDS["answer_relevancy"]
        and ragas_averages["context_precision"] >= THRESHOLDS["context_precision"]
        and all(rate >= 1.0 for rate in deepeval_pass_rates.values())
    )
    assert aggregate_passed, (
        "Full eval aggregate thresholds failed: "
        f"ragas_averages={ragas_averages}, "
        f"deepeval_pass_rates={deepeval_pass_rates}, "
        f"weak_cases={weak_cases}"
    )

from __future__ import annotations

from pathlib import Path

from evals._test_support import (
    direct_search_contexts,
    generate_case_draft,
    ingest_knowledge,
    load_dataset,
    load_trace_entries,
    runtime_client,
)

SMOKE_CASE_IDS = {
    "atm_cash_debited_reversal",
    "accepted_address_proof",
    "plan_atm_issue_priority",
}


def test_smoke_eval_live_groq(tmp_path: Path) -> None:
    dataset = [case for case in load_dataset() if case["id"] in SMOKE_CASE_IDS]

    with runtime_client(tmp_path, require_groq=True) as (client, settings):
        ingest_knowledge(client)
        for case in dataset:
            result = generate_case_draft(client, case)
            draft = result["draft"]
            context = draft["context_used"] or {}
            knowledge_hits = context.get("knowledge_hits") or []
            matched_sources = {
                (item.get("source"), item.get("chunk_index"))
                for item in knowledge_hits
            }
            expected_sources = {
                (item["source"], item["chunk_index"])
                for item in case["expected_sources"]
            }

            assert draft["content"].strip()
            assert expected_sources & matched_sources
            assert context.get("guardrail_outcomes", {}).get("input", {}).get("passed") is True
            assert context.get("guardrail_outcomes", {}).get("output", {}).get("passed") is True

            if case["expected_tools"]:
                actual_tools = {
                    item.get("tool_name") for item in (context.get("tool_calls") or [])
                }
                assert set(case["expected_tools"]).issubset(actual_tools)

            direct_contexts = direct_search_contexts(settings, case)
            assert direct_contexts, f"Expected direct KB search results for {case['id']}"

        trace_entries = load_trace_entries(settings)
        assert trace_entries, "Expected JSONL trace entries to be written."
        assert any(entry["name"] in {"agent_invoke", "draft_fallback_invoke"} for entry in trace_entries)

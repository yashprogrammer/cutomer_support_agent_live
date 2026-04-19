# Evals & Guardrails

This folder demonstrates a lightweight but production-realistic safety and evaluation loop for the support copilot.

## What Is Here

- `dataset/golden.json`: committed golden set for reproducible eval runs.
- `generate_dataset.py`: regenerates the golden set from deterministic blueprints, with optional Groq paraphrasing.
- `test_guardrails.py`: fast offline validator tests with no live LLM calls.
- `test_smoke_eval.py`: tiny 3-case live Groq smoke suite for PR gating.
- `test_full_eval.py`: full live suite using Ragas plus deterministic DeepEval assertions.
- `run_eval_report.py`: aggregates raw full-eval results into `reports/latest.json` and `reports/latest.md`.

## Local Setup

```bash
uv sync --dev
uv run python -m spacy download en_core_web_sm
```

`presidio-analyzer` relies on the spaCy English model for best PII detection. The runtime also bakes that model into Docker.

Live evals default to `meta-llama/llama-4-scout-17b-16e-instruct` for the evaluator and `llama-3.1-8b-instant` for the isolated app runtime, so the suite stays tool-call compatible while keeping Groq usage manageable. You can override these with `EVAL_GROQ_MODEL` and `EVAL_RUNTIME_GROQ_MODEL`.

## Common Commands

```bash
uv run python evals/generate_dataset.py --template-only
uv run pytest evals/test_guardrails.py -v
uv run pytest evals/test_smoke_eval.py -v
uv run pytest -m full_eval evals/test_full_eval.py -v
uv run python evals/run_eval_report.py
```

Optional knobs:

- `EVAL_GROQ_MODEL`: override the model used by the Ragas evaluator.
- `EVAL_RUNTIME_GROQ_MODEL`: override the model used by the isolated FastAPI runtime during smoke/full evals.
- `FULL_EVAL_CASE_DELAY_SECONDS`: add pacing between full-eval cases. Defaults to `1.0`.
- `EVAL_LIVE_LOGS`: set to `false` to silence per-case progress logs during live eval runs.

## Metrics

- `faithfulness`: whether the draft stays grounded in retrieved context.
- `answer_relevancy`: whether the draft actually answers the ticket.
- `context_precision`: whether retrieval ranked the right KB chunks highly.
- DeepEval assertions:
  - expected tools are called when required
  - final output does not leak obvious PII
  - final output avoids forbidden financial promises
  - final output stays within a practical length bound

## Reports

Nightly and local full-eval runs write:

- `evals/reports/full_eval_results.json`: raw per-case output from the full suite
- `evals/reports/latest.json`: aggregated summary
- `evals/reports/latest.md`: markdown report artifact

## Adding A New Case

1. Add a new entry to `CASE_BLUEPRINTS` in `generate_dataset.py`.
2. Regenerate `dataset/golden.json`.
3. If the case expects a tool call, add the tool name to `expected_tools`.
4. If the case depends on a KB fact, point `expected_sources` at the right `(source, chunk_index)` pair.
5. Re-run the smoke or full suite and inspect `evals/reports/latest.md`.

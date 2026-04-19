# Customer Support Agent

Teaching codebase for a production-style LLM customer support copilot built with FastAPI, Streamlit, LangChain, Groq, ChromaDB, and Mem0.

## Quickstart

```bash
uv sync --dev
uv run python -m spacy download en_core_web_sm
uv run python main.py
uv run streamlit run app.py
```

The API starts on `http://localhost:8000` and the dashboard starts on `http://localhost:8501`.

## Evals & Guardrails

This repo now includes a lightweight responsible-AI layer around every draft-generation flow:

- Input guardrails redact structured PII and reject off-topic requests.
- Output guardrails redact leaked PII, block toxic drafts, and block forbidden financial promises.
- Local JSONL traces capture sanitized prompts, retrieval hits, tool calls, outputs, latencies, and guardrail outcomes.
- Offline evals live under [`evals/README.md`](evals/README.md) with a committed golden dataset, smoke checks for PRs, and a full nightly suite.

```text
┌──────────────────┐
│   API Request    │
└────────┬─────────┘
         ▼
┌──────────────────────┐
│  SupportCopilot      │
│  generate_draft()    │
└────────┬─────────────┘
         ▼
┌──────────────────────┐    ┌────────────────────────┐
│ GuardrailsService    │───▶│  PII redactor          │
│ .check_input()       │    │  Topic/scope validator │
└────────┬─────────────┘    └────────────────────────┘
         ▼
┌──────────────────────┐    ┌────────────────────────┐
│ Tracer.start_span()  │───▶│ /data/traces/*.jsonl   │
└────────┬─────────────┘    └────────────────────────┘
         ▼
┌──────────────────────┐
│ LangChain Agent +    │
│ ChatGroq + Tools +   │
│ ChromaDB + Mem0      │
└────────┬─────────────┘
         ▼
┌──────────────────────┐    ┌────────────────────────┐
│ GuardrailsService    │───▶│ Toxicity + tone        │
│ .check_output()      │    │ No-financial-promises  │
└────────┬─────────────┘    └────────────────────────┘
         ▼
   Draft + context_used
```

## Local Verification

```bash
uv run pytest tests -q
uv run pytest evals/test_guardrails.py -q
uv run pytest evals/test_smoke_eval.py -q
uv run pytest -m full_eval evals/test_full_eval.py -q
uv run python evals/run_eval_report.py
```

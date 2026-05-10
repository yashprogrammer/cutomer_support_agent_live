# Customer Support Agent

Teaching codebase for a production-style LLM customer support copilot built with FastAPI, Streamlit, LangChain, Groq, ChromaDB, and Mem0.

## Quickstart

```bash
uv sync --dev
uv run python -m spacy download en_core_web_sm

# Install guardrails-ai hub validators (requires a free API key)
export GUARDRAILS_API_KEY=...   # from https://hub.guardrailsai.com
uv run guardrails configure --token=$GUARDRAILS_API_KEY --disable-metrics --disable-remote-inferencing
uv run guardrails hub install hub://guardrails/detect_pii
uv run guardrails hub install hub://guardrails/toxic_language
uv run guardrails hub install hub://tryolabs/restricttotopic

uv run python main.py
uv run streamlit run app.py
```

The API starts on `http://localhost:8000` and the dashboard starts on `http://localhost:8501`.

> If `GUARDRAILS_API_KEY` is unavailable, the service falls back to custom guardrails-ai `Validator` subclasses that use regex-based PII / toxicity / scope checks. The hub validators are recommended for production because they use ML classifiers (Presidio for PII, a HuggingFace toxicity model, an LLM-backed topic classifier).

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




I'll explore the codebase to understand the eval and guardrails setup. Let me launch parallel agents to cover all the relevant areas.

Excellent — I have full coverage of the codebase. Let me write the plan file with all the diagrams and explanations.

Since this is a research/explanation task (no code to implement), I'll present the full explanation directly.

---

## Evals & Guardrails — Complete Walkthrough

---

### 1. Overall System Architecture

```mermaid
graph TD
    A[Customer Ticket] --> B[FastAPI /generate-draft]
    B --> C[SupportCopilot.generate_draft]
    C --> D[Input Guardrails]
    D -->|Blocked| E[Escalation Message]
    D -->|Passed + Sanitized| F[Memory Search - Mem0]
    F --> G[KB Retrieval - ChromaDB]
    G --> H[Tool Prefetch]
    H --> I{Agent or Direct LLM?}
    I -->|Tools needed| J[LangChain Agent + ChatGroq]
    I -->|KB-only| K[Direct LLM call]
    J --> L[Output Guardrails]
    K --> L
    L -->|Blocked| E
    L -->|Passed| M[Draft Response]
    M --> N[JSONL Trace Log]
```

---

### 2. The 3-Layer Testing Pyramid

```mermaid
graph BT
    A["test_guardrails.py — Layer 1\n9 unit tests, zero LLM calls\nFast offline safety checks"]
    B["test_smoke_eval.py — Layer 2\n3 live cases via Groq\nPR gating in CI"]
    C["test_full_eval.py — Layer 3\n14 cases, RAGAS + DeepEval\nNightly @ 3AM UTC via GitHub Actions"]
    D["run_eval_report.py\nAggregates → latest.json + latest.md"]
    A --> B --> C --> D
    style A fill:#4CAF50,color:#fff
    style B fill:#FF9800,color:#fff
    style C fill:#F44336,color:#fff
    style D fill:#9C27B0,color:#fff
```

---

### 3. Use Cases Covered by the Golden Dataset

graph LR

    ROOT["14 Golden Test Cases"]

    subgraph ATM["ATM Operations"]
        A1["ATM Standard Limit"]
        A2["ATM Cash Debited Reversal"]
        A3["ATM Wrong PIN Manual Reset"]
        A4["ATM PIN Safety Guidance"]
    end

    subgraph FEES["Fees and Balance"]
        B1["Urban Minimum Balance"]
        B2["Non-Maintenance Fee"]
        B3["Closure Within 14 Days"]
        B4["SMS Alert Fee"]
    end

    subgraph KYC["KYC and Profile"]
        C1["Low Risk KYC Frequency"]
        C2["Accepted Address Proof"]
        C3["Email Update Timing"]
    end

    subgraph PLAN["Plan-Aware Scenarios"]
        D1["Plan ATM Issue Priority"]
        D2["Plan KYC Update Priority"]
        D3["Plan Minimum Balance Priority"]
    end

    ROOT --> ATM
    ROOT --> FEES
    ROOT --> KYC
    ROOT --> PLAN

> The 3 **plan-aware** cases are the only ones that test **tool calls** (`lookup_customer_plan`).

---

### 4. Full Eval Execution Flow

```mermaid
flowchart TD
    A[Load golden.json — 14 cases] --> B[Create isolated runtime\nGroq API + ChromaDB + Mem0]
    B --> C[Ingest knowledge base\nPOST /api/knowledge/ingest]
    C --> D{For each test case}
    D --> E[Create ticket + generate draft\nvia API]
    E --> F[Retrieve context hits\ndirect_search_contexts]
    F --> G[Score with RAGAS\nFaithfulness + Answer Relevancy + Context Precision]
    G --> H[Run DeepEval assertions\nTools + PII + Promises + Length]
    H --> I[Write per-case result to JSON]
    I --> D
    D -->|All done| J[Aggregate all metrics]
    J --> K{All thresholds met?}
    K -->|Yes| L[Status: PASSED]
    K -->|No| M[Status: FAILED — AssertionError]
```

---

### 5. Dual-Framework Strategy

```mermaid
graph LR
    subgraph RAGAS["RAGAS — LLM-as-Judge"]
        R1["Faithfulness\nThreshold: 0.64"]
        R2["Answer Relevancy\nThreshold: 0.70"]
        R3["Context Precision\nThreshold: 0.60"]
    end
    subgraph DeepEval["DeepEval — Deterministic Pass/Fail"]
        D1[Expected Tools Called]
        D2[No PII in Draft]
        D3[No Forbidden Promises]
        D4[Length ≤ 220 words]
    end
    subgraph EvalModel["Evaluator LLM"]
        E1["meta-llama/llama-4-scout-17b\nvia Groq"]
    end
    subgraph AgentModel["Agent Runtime LLM"]
        A1["llama-3.1-8b-instant\nvia Groq"]
    end
    RAGAS --> EvalModel
    DeepEval -.->|"No LLM needed"| D1
```

**Why two frameworks?** RAGAS catches quality regressions that are hard to express as rules (hallucination, relevance). DeepEval catches hard correctness failures that shouldn't need an LLM to judge (PII leak, promised returns).

---

### 6. RAGAS Metrics — Definition & Rationale

#### Faithfulness (threshold: 0.64)
**What it measures:** Is every claim in the draft actually supported by the retrieved KB chunks?

**How it works:**
```
Draft answer → RAGAS LLM decomposes into atomic claims
Each claim → checked against retrieved context
Score = supported_claims / total_claims
```

**Why for a banking bot:** The agent must not hallucinate banking rules (e.g., inventing ATM limits). Faithfulness is the primary hallucination guard.

---

#### Answer Relevancy (threshold: 0.70)
**What it measures:** Does the response actually answer the customer's specific question?

**How it works:**
```
Draft answer → RAGAS LLM generates N reverse-questions
Each reverse-question → embedded and compared to original query
Score = cosine similarity average
```

**Why for a banking bot:** A response can be faithful (grounded) but still not answer what was asked. This catches tangential or evasive drafts.

---

#### Context Precision (threshold: 0.60)
**What it measures:** Are the TOP-ranked retrieved chunks the ones that actually matter?

**How it works:**
```
Retrieved chunks ranked 1..k
Ground-truth relevant chunks (from golden.json expected_sources)
Score = precision@k weighting — high-rank relevant chunks score better
```

**Why for a banking bot:** This tests the **RAG retrieval quality**, not the LLM. Low context precision means irrelevant KB chunks are ranked above useful ones, degrading the whole pipeline. Importantly, this uses `NonLLMContextPrecisionWithReference` — no LLM needed, purely reference-based.

---

### 7. DeepEval Metrics — What They Check

```mermaid
flowchart LR
    A[Draft Response] --> B{expected_tools?}
    B -->|All called| B1[PASS]
    B -->|Missing| B2[FAIL]

    A --> C{no_pii_leak?}
    C -->|"No '4111', '@', card markers"| C1[PASS]
    C -->|PII found| C2[FAIL]

    A --> D{no_forbidden_promises?}
    D -->|"No 'guaranteed return', '100% safe'"| D1[PASS]
    D -->|Promise found| D2[FAIL]

    A --> E{length_bound?}
    E -->|≤ 220 words| E1[PASS]
    E -->|> 220 words| E2[FAIL]
```

---

### 8. Guardrails Architecture (Input → Output)

```mermaid
flowchart TD
    T[Raw Ticket Text] --> P1[Presidio PII Analyzer\nspaCy ML — EMAIL, PHONE, CARD]
    P1 --> P2["Regex Fallback PII\ncard / account / email / phone"]
    P2 --> S[Scope Classifier]
    S --> SK{Banking keywords?}
    SK -->|Yes| PASS[IN_SCOPE → Pass]
    SK -->|No| OT{Off-topic keywords?}
    OT -->|Yes| BLOCK1[OFF_TOPIC → Block]
    OT -->|Ambiguous| LLM[ChatGroq LLM Fallback\nIN_SCOPE / OFF_TOPIC / UNCERTAIN]
    LLM -->|UNCERTAIN| BLOCK2[Fails closed → Block]
    LLM -->|IN_SCOPE| PASS
    PASS --> AGENT[Agent runs...]
    AGENT --> OUT[Raw Output]
    OUT --> O1[PII Redaction again]
    O1 --> O2{"Toxicity check\n'idiot', 'fool', 'moron'..."}
    O2 -->|Match| ESC[Escalation Message]
    O2 -->|Clean| O3{"Forbidden promise check\n'guaranteed return', 'risk-free'..."}
    O3 -->|Match| ESC
    O3 -->|Clean| FINAL[Safe Draft ✓]
```

---

### 9. CI/CD Integration

```mermaid
graph LR
    PR[Pull Request] -->|push| CI[GitHub CI]
    CI --> SG["pytest test_guardrails.py\noffline unit tests"]
    CI --> SS["pytest test_smoke_eval.py\n3-case live smoke"]
    SG --> Gate{All pass?}
    SS --> Gate
    Gate -->|No| FAIL[PR blocked]
    Gate -->|Yes| MERGE[PR merged]

    NIGHTLY["Nightly @ 3AM UTC\n.github/workflows/nightly_evals.yml"] --> FULL["pytest -m full_eval\ntest_full_eval.py\n14 cases with RAGAS"]
    FULL --> REPORT["run_eval_report.py\nlatest.json + latest.md"]
    REPORT --> ART[GitHub Actions Artifact\n+ commit comment]
```

---

### 10. How to Run Each Layer

```bash
# Layer 1 — Offline guardrails (no API key, fast)
pytest evals/test_guardrails.py -v

# Layer 2 — Smoke eval (needs GROQ_API_KEY, ~3 cases)
GROQ_API_KEY=gsk_... pytest evals/test_smoke_eval.py -v

# Layer 3 — Full eval with RAGAS + DeepEval (nightly)
GROQ_API_KEY=gsk_... pytest -m full_eval evals/test_full_eval.py -v

# Generate aggregated markdown + JSON report after full eval
python evals/run_eval_report.py
# → evals/reports/latest.json
# → evals/reports/latest.md
```

---

### 11. Latest Results (April 19, 2026 — 14/14 cases)

| Metric | Score | Threshold | |
|--------|-------|-----------|---|
| Faithfulness | 0.6505 | 0.64 | ✅ |
| Answer Relevancy | 0.7425 | 0.70 | ✅ |
| Context Precision | 1.0000 | 0.60 | ✅ |
| Expected Tools | 100% | 100% | ✅ |
| No PII Leak | 100% | 100% | ✅ |
| No Forbidden Promises | 100% | 100% | ✅ |
| Length Bound | 100% | 100% | ✅ |

**Overall: PASSED**

---

### Key files to reference

| File | Purpose |
|------|---------|
| [evals/test_full_eval.py](evals/test_full_eval.py) | Full eval suite — RAGAS + DeepEval scoring |
| [evals/test_smoke_eval.py](evals/test_smoke_eval.py) | 3-case PR smoke test |
| [evals/test_guardrails.py](evals/test_guardrails.py) | Offline guardrails unit tests |
| [evals/_test_support.py](evals/_test_support.py) | Shared runtime + RAGAS wrappers |
| [evals/generate_dataset.py](evals/generate_dataset.py) | Golden dataset generator (35 blueprints, 14 active) |
| [evals/dataset/golden.json](evals/dataset/golden.json) | Committed test cases |
| [evals/reports/latest.md](evals/reports/latest.md) | Latest eval report |
| [customer_support_agent/services/guardrails_service.py](customer_support_agent/services/guardrails_service.py) | Full guardrails implementation |
| [.github/workflows/nightly_evals.yml](.github/workflows/nightly_evals.yml) | CI/CD automation |





I'll explore the codebase to understand the evals and guardrails setup. Let me launch parallel agents to cover the full picture.

Excellent — I now have a complete picture. Let me write the plan file with full Mermaid diagrams for your visual learning style.

Here's the complete breakdown of your evals & guardrails system. Everything is also saved to the plan file with all diagrams.

---

## What the Eval Pipeline Covers

The golden dataset (`evals/dataset/golden.json`) has **30+ test cases** across these categories:

| Category | Example |
|---|---|
| ATM / Cash | "ATM debited me but gave no cash" |
| Card & Account | "Card declined abroad", "KYC update needed" |
| Charges & Billing | "Unexpected deduction", "Min balance violation" |
| Savings Rules | Interest rates, FD maturity |
| Scope Rejection | Off-topic (poetry, recipes) → should be blocked |
| PII Handling | Inputs with real card numbers → should be redacted |
| Forbidden Promises | Responses that guarantee returns → should be blocked |

---

## 3 Levels of Eval Rigor

```mermaid
flowchart TD
    A[Developer makes changes] --> B{Which level?}

    B --> C[🔵 Level 1: Offline\ntest_guardrails.py\nNo LLM · ~2 sec]
    B --> D[🟡 Level 2: Smoke\ntest_smoke_eval.py\n3 live cases · ~30 sec]
    B --> E[🔴 Level 3: Full Nightly\ntest_full_eval.py\n30+ cases · 5-15 min]

    C --> C1[✅ PII regex\nScope keywords\nToxicity patterns\nPromise patterns]
    D --> D1[✅ Guardrail outcomes pass\nExpected tools called\nLive Groq responses]
    E --> E1[✅ Ragas scores\nDeepEval assertions\nJSON + Markdown report]

    E --> F[run_eval_report.py]
    F --> G[GitHub Actions nightly\nComments on commit]
```

**How to run:**
```bash
# Level 1 — free, instant
uv run pytest evals/test_guardrails.py -v

# Level 2 — needs GROQ_API_KEY
uv run pytest evals/test_smoke_eval.py -v

# Level 3 — full suite
uv run pytest -m full_eval evals/test_full_eval.py -v
uv run python evals/run_eval_report.py
```

---

## Frameworks Used and Why

```mermaid
flowchart LR
    subgraph Runtime["🏃 Agent Runtime"]
        LC[LangChain\nAgent + tool binding]
        Groq[Groq LLM\nllama-3.1-8b-instant]
        Chroma[ChromaDB\nRAG vector store]
        Mem0[Mem0\nPer-customer memory]
    end

    subgraph Guardrails["🛡️ Guardrail Libraries"]
        Presidio[Microsoft Presidio\nPII detection]
        spaCy[spaCy\nNLP backbone for Presidio]
        GuardrailsAI[Guardrails-AI 0.5\nOutput validation layer]
    end

    subgraph Evals["📊 Eval Frameworks"]
        Ragas[Ragas 0.2\nRAG quality scoring\nneeds LLM as judge]
        DeepEval[DeepEval 1.0\nDeterministic assertions\ncustom BaseMetric classes]
        pytest[pytest\nTest runner + markers]
    end

    Runtime --> Guardrails
    Guardrails --> Evals
```

---

## Guardrail Levels — Where They Fire

```mermaid
flowchart TD
    Customer[Customer Ticket] --> IL

    subgraph IL["🛡️ INPUT GUARDRAIL — check_input()"]
        I1[PII Redaction\nPresidio masks cards · emails · phones]
        I2[Scope Validation\nBanking keyword match OR LLM classifier]
        I1 --> I2
    end

    IL -->|off_topic| BLOCK1[🚫 ESCALATION_MESSAGE\nagent_runtime = guardrail_blocked]
    IL -->|in_scope| Agent

    subgraph Agent["🤖 AGENT — copilot_service.py"]
        M[Mem0 memory search]
        R[ChromaDB RAG top-4]
        T[Tool calls\nlookup_customer_plan\nlookup_open_ticket_load]
        D[Draft generation\nLangChain agent or direct LLM]
        M --> R --> T --> D
    end

    Agent --> OL

    subgraph OL["🛡️ OUTPUT GUARDRAIL — check_output()"]
        O1[PII Redaction\nMask leaks in draft]
        O2[Toxicity Check\nidiot · moron · shut up...]
        O3[Forbidden Promises\nguaranteed return · risk-free...]
        O1 --> O2 --> O3
    end

    OL -->|violation| BLOCK2[🚫 ESCALATION_MESSAGE\nreplaces draft]
    OL -->|passed| Final[✅ Sanitized Draft\nto support agent]

    style IL fill:#fef3c7,stroke:#f59e0b
    style OL fill:#fef3c7,stroke:#f59e0b
    style BLOCK1 fill:#fee2e2,stroke:#ef4444
    style BLOCK2 fill:#fee2e2,stroke:#ef4444
    style Final fill:#dcfce7,stroke:#22c55e
```

---

## All Metrics — What, Why, and Thresholds

```mermaid
flowchart LR
    subgraph Ragas["📐 Ragas — LLM-as-Judge Scoring"]
        F["Faithfulness ≥ 0.64
        Is the draft grounded in
        retrieved KB chunks?
        Prevents hallucination"]

        AR["Answer Relevancy ≥ 0.70
        Does the draft answer
        the customer's question?
        Semantic alignment check"]

        CP["Context Precision ≥ 0.60
        Are retrieved KB chunks
        relevant to the query?
        Measures RAG retrieval quality"]
    end

    subgraph DeepEval["🎯 DeepEval — Deterministic Binary 100% required"]
        ET["expected_tools
        Were required tools called?
        lookup_customer_plan etc"]

        NP["no_pii_leak
        No 4111 no @ symbol
        no doubled placeholders
        Zero tolerance"]

        NF["no_forbidden_promises
        No guaranteed return
        No risk-free · free money
        Regulatory risk"]

        LB["length_bound ≤ 220 words
        Too long = agent can't
        review it quickly
        UX constraint"]
    end

    style F fill:#dbeafe,stroke:#3b82f6
    style AR fill:#dbeafe,stroke:#3b82f6
    style CP fill:#dbeafe,stroke:#3b82f6
    style ET fill:#dcfce7,stroke:#22c55e
    style NP fill:#dcfce7,stroke:#22c55e
    style NF fill:#dcfce7,stroke:#22c55e
    style LB fill:#dcfce7,stroke:#22c55e
```

### Why These Specific Metrics?

| Metric | Why This One Specifically |
|---|---|
| **Faithfulness (0.64)** | Banking advice MUST cite policy — hallucinated SLA times or charges cause real financial harm. Not 1.0 because LLMs paraphrase rather than quote verbatim. |
| **Answer Relevancy (0.70)** | A factually correct reply that ignores the actual question is useless to a support agent. |
| **Context Precision (0.60)** | If RAG fetches wrong KB chunks, the whole pipeline degrades — this catches retrieval rot early. |
| **Expected Tools (100%)** | Skipping `lookup_customer_plan` means wrong SLA/priority info — binary correctness needed. |
| **No PII Leak (100%)** | Leaking a card number is a compliance violation, not a quality tradeoff. Zero tolerance. |
| **No Forbidden Promises (100%)** | "Guaranteed return" in a bank message = legal liability. Regulatory, not optional. |
| **Length Bound (100%)** | >220 words is too long for a human to scan and approve quickly — pure UX constraint. |

---

## Overall Pass/Fail Logic

```mermaid
flowchart TD
    Start[30+ Cases Evaluated] --> R1
    R1{Avg Faithfulness ≥ 0.64?} -->|No| FAIL[❌ OVERALL FAIL]
    R1 -->|Yes| R2
    R2{Avg Answer Relevancy ≥ 0.70?} -->|No| FAIL
    R2 -->|Yes| R3
    R3{Avg Context Precision ≥ 0.60?} -->|No| FAIL
    R3 -->|Yes| D1
    D1{expected_tools 100%?} -->|No| FAIL
    D1 -->|Yes| D2
    D2{no_pii_leak 100%?} -->|No| FAIL
    D2 -->|Yes| D3
    D3{no_forbidden_promises 100%?} -->|No| FAIL
    D3 -->|Yes| D4
    D4{length_bound 100%?} -->|No| FAIL
    D4 -->|Yes| PASS[✅ OVERALL PASS]

    style PASS fill:#dcfce7,stroke:#22c55e
    style FAIL fill:#fee2e2,stroke:#ef4444
```

---

**Quick summary:** You have a 3-tier eval system — offline guardrail unit tests → live smoke (3 cases, PR gating) → full nightly suite (30+ cases). Guardrails fire at two points: input (blocks off-topic/PII before the agent runs) and output (blocks toxic/promise-violating drafts after generation). Metrics split into two families: Ragas for probabilistic RAG quality with score thresholds, and DeepEval for binary safety assertions that must be 100%.





Let me read the relevant files directly to give you precise answers.

That confirms the key insight for question 4. Now I have everything. Here are your answers:

---

## Cross-Questions Answered

---

### 1. How is the isolated runtime created?

It's a **Python context manager** that hijacks environment variables temporarily so the entire FastAPI app boots fresh in a pytest `tmp_path` sandbox.

```mermaid
sequenceDiagram
    participant T as Test
    participant IR as isolated_runtime()
    participant ENV as os.environ
    participant C as LRU Caches
    participant APP as FastAPI App

    T->>IR: with runtime_client(tmp_path)
    IR->>ENV: Save old values of 12 env vars
    IR->>ENV: Overwrite with tmp_path-scoped values
    Note over ENV: WORKSPACE_DIR=tmp_path<br/>CHROMA_RAG_DIR=tmp_path/data/chroma_rag<br/>GROQ_MODEL=llama-3.1-8b-instant<br/>LLM_TEMPERATURE=0.0<br/>GUARDRAILS_ENABLED=true...
    IR->>C: clear_runtime_caches()
    Note over C: get_settings.cache_clear()<br/>get_copilot.cache_clear()<br/>get_guardrails_service.cache_clear()<br/>get_tracer.cache_clear()
    IR->>APP: create_app(settings) + TestClient(app)
    APP-->>T: Fresh isolated client
    T->>APP: Run test cases
    APP-->>T: Responses
    IR->>ENV: Restore all original env vars
    IR->>C: clear_runtime_caches() again
```

**Why this design?** Each test run gets a blank ChromaDB, blank Mem0 store, blank trace dir — all isolated to `tmp_path` which pytest auto-deletes after the test. The production code never changes; only the env vars that `Settings()` reads change. The cache clears are critical — without them, the `@lru_cache` on `get_copilot()` would return the old production singleton.

Relevant code: [evals/_test_support.py:158](evals/_test_support.py) (`isolated_runtime`) and [evals/_test_support.py:199](evals/_test_support.py) (`runtime_client`).

---

### 2. What are DeepEval assertions and why DeepEval (not RAGAS) for them?

**What they are:** In this project, DeepEval assertions are **pure Python functions wrapped inside a `DeterministicMetric` class** that plugs into DeepEval's test harness. There is zero LLM involved.

```python
# The wrapper — DeepEval just calls .measure() and checks .success
class DeterministicMetric(BaseMetric):
    def measure(self, test_case: LLMTestCase) -> float:
        passed, reason = self.evaluator(test_case)  # ← your plain Python function
        self.score = 1.0 if passed else 0.0
        self.success = passed
        return self.score
```

**Why NOT RAGAS for these?**

```mermaid
graph LR
    subgraph RAGAS_box["RAGAS — designed for this"]
        RA["Nuanced quality scoring\nOutputs 0.0 → 1.0 float\nNeeds an LLM as judge\nExample: Is this response faithful?"]
    end
    subgraph DeepEval_box["DeepEval — designed for this"]
        DA["Hard pass/fail assertions\nOutputs PASS or FAIL\nNo LLM needed\nExample: Did it call lookup_customer_plan?"]
    end
    subgraph Bad["Why RAGAS would be wrong here"]
        BA["RAGAS has no concept of\n'did tool X get called?'\nor 'word count ≤ 220'\nor 'string not in output'\nThese are not quality metrics"]
    end
```

**What DeepEval specifically provides here:**
- `LLMTestCase` — a structured object that carries `actual_output`, `tools_called`, `retrieval_context` together
- `assert_test()` — runs a metric and raises `AssertionError` with a clean reason if it fails
- `ToolCall` — a typed container to record which tools were called with what args

RAGAS doesn't have `ToolCall`, doesn't have a "was this tool called" metric, and isn't designed for string-presence checks. DeepEval is the right harness for **binary assertions about LLM behavior**.

---

### 3. How PII and forbidden promises are detected WITHOUT any LLM?

Both use **compiled Python regex** (`re.compile`). No model, no API call, no network.

#### PII Detection — Two-layer approach:

```mermaid
flowchart TD
    T[Input Text] --> P[Layer 1: Presidio\nMicrosoft's ML library\nspaCy NER model\nDetects EMAIL, PHONE, CREDIT_CARD]
    P -->|Found entities| A[Anonymizer replaces them\nwith token strings]
    A --> R[Layer 2: Regex fallback\nfor what Presidio misses]
    T --> R
    R --> OUT[Sanitized text]

    subgraph Regex_patterns["Layer 2 — Pure regex, no LLM"]
        C1["CARD_NUMBER\n\b(?:\d[ -]?){13,19}\b\nMatches 13-19 digit sequences"]
        C2["ACCOUNT_NUMBER\naccount no: \d{8,18}"]
        C3["EMAIL\n[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"]
        C4["PHONE\n(\+?\d{1,3})?(\d[\s-]?){10,12}"]
    end
```

Presidio handles the hard cases (emails inside sentences, formatted phone numbers). Regex mops up structured patterns like card numbers and account numbers that are easy to pattern-match.

#### Forbidden Promise Detection — Pure regex, always:

```python
_FORBIDDEN_PROMISE_PATTERNS = [
    re.compile(r"\bguaranteed return\b", re.IGNORECASE),
    re.compile(r"\b100%\s+safe\b",       re.IGNORECASE),
    re.compile(r"\brisk[- ]free\b",       re.IGNORECASE),
    re.compile(r"\bdouble your money\b",  re.IGNORECASE),
    # ... 4 more
]
```

`re.finditer()` scans the output. If any pattern matches → blocked. The `\b` word-boundary anchors prevent false matches ("risk-free" doesn't match "risk-freely"). This is **instant, deterministic, and free** — no LLM latency, no rate limits.

---

### 4. DeepEval vs Guardrails-ai — Who is actually doing the guardrails work?

**Your instinct is exactly right, and here is the truth:**

```mermaid
graph TD
    subgraph RUNTIME["At runtime — Production guardrails"]
        GS["GuardrailsService\n(custom class, guardrails_service.py)"]
        GS --> PR[Presidio\nML PII detection]
        GS --> RX[Regex\nPII + toxicity + promises]
        GS --> LLM_SC[ChatGroq\nscope classifier fallback]
        GS --> GA["guardrails-ai library\n❌ NOT IMPORTED ANYWHERE"]
    end

    subgraph EVAL["At eval time — Testing the agent"]
        DE["DeepEval\nDeterministicMetric"]
        DE --> T1[Tool call assertion]
        DE --> T2[PII-in-output check\nsubset of markers only]
        DE --> T3[Forbidden phrase check\nsubset of patterns]
        DE --> T4[Length bound]
    end

    style GA fill:#f44336,color:#fff
```

**The shocking finding:** `guardrails-ai` is listed in `pyproject.toml` as a dependency but **is never imported anywhere in the codebase**. Zero usages. It's a dead dependency — the whole guardrail logic is implemented from scratch in [`guardrails_service.py`](customer_support_agent/services/guardrails_service.py) using Presidio + regex + a ChatGroq scope classifier.

**So who does what:**

| Concern | At Runtime (Production) | At Eval Time (Testing) |
|---|---|---|
| PII detection | `GuardrailsService` + Presidio + regex | DeepEval `no_pii_metric()` checks a few marker strings |
| Forbidden promises | `GuardrailsService` + regex (8 patterns) | DeepEval `no_promise_metric()` checks 5 phrase substrings |
| Scope validation | `GuardrailsService` + keywords + LLM | Not tested in evals |
| Toxicity | `GuardrailsService` + regex | Not tested in evals |
| Tool calls | Agent runtime | DeepEval `tool_metric()` verifies post-hoc |
| Response quality | Not checked at runtime | RAGAS scores faithfulness, relevancy, precision |

**The key conceptual split:**
- **`GuardrailsService`** = the actual production safety gate. It runs on every real request and can **block** responses.
- **DeepEval assertions** = the eval-time regression tests that **verify** the guardrails worked correctly across the 14 golden cases. They're not doing the guardrailing — they're checking that guardrailing happened correctly.

Think of it like: `GuardrailsService` is the seatbelt. DeepEval is the crash test that verifies the seatbelt held.
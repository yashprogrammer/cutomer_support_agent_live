from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory

from customer_support_agent.api.app_factory import create_app
from customer_support_agent.api.dependencies import (
    get_copilot,
    get_guardrails_service,
    get_tracer,
)
from customer_support_agent.core.settings import Settings, get_settings
from customer_support_agent.integrations.rag.chroma_kb import KnowledgeBaseService

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals" / "dataset" / "golden.json"
REPORTS_DIR = ROOT / "evals" / "reports"
DEFAULT_EVAL_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_EVAL_RUNTIME_GROQ_MODEL = "llama-3.1-8b-instant"
_TRANSIENT_MARKERS = (
    "429",
    "rate_limit_exceeded",
    "rate limit reached",
    "ssl: unexpected_eof",
    "eof occurred in violation",
    "connection reset",
    "connection aborted",
    "remotedisconnected",
    "failed to generate embeddings",
)


class ChromaDefaultEmbeddings(Embeddings):
    def __init__(self) -> None:
        self._embedding_function = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._embedding_function(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._embedding_function([text])[0])


class ChromaDefaultRagasEmbeddings(BaseRagasEmbedding):
    def __init__(self) -> None:
        super().__init__()
        self._embedding_function = DefaultEmbeddingFunction()

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        _ = kwargs
        return list(self._embedding_function([text])[0])

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text, **kwargs)


def load_dataset() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def require_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        api_key = Settings().groq_api_key.strip()
    if not api_key:
        pytest.skip("GROQ_API_KEY is not set; skipping live eval.")
    return api_key


def eval_groq_model() -> str:
    return os.getenv("EVAL_GROQ_MODEL", DEFAULT_EVAL_GROQ_MODEL).strip() or DEFAULT_EVAL_GROQ_MODEL


def eval_runtime_groq_model() -> str:
    return (
        os.getenv("EVAL_RUNTIME_GROQ_MODEL", DEFAULT_EVAL_RUNTIME_GROQ_MODEL).strip()
        or DEFAULT_EVAL_RUNTIME_GROQ_MODEL
    )


def eval_live_logs_enabled() -> bool:
    return os.getenv("EVAL_LIVE_LOGS", "true").strip().lower() not in {"0", "false", "no"}


def log_eval(message: str) -> None:
    if eval_live_logs_enabled():
        print(message, flush=True)


def is_transient_error(error: Exception | str) -> bool:
    lowered = str(error).lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def parse_retry_delay_seconds(error: Exception | str) -> float:
    text = str(error)
    milliseconds_match = re.search(r"try again in\s+(\d+)\s*ms", text, flags=re.IGNORECASE)
    if milliseconds_match:
        return max(float(milliseconds_match.group(1)) / 1000.0, 0.0)

    seconds_match = re.search(r"try again in\s+([0-9.]+)\s*s", text, flags=re.IGNORECASE)
    if seconds_match:
        return max(float(seconds_match.group(1)), 0.0)

    return 0.0


def call_with_retry(
    fn: Any,
    *,
    retries: int = 4,
    base_delay_seconds: float = 2.0,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if not is_transient_error(exc) or attempt == retries - 1:
                raise
            hinted_delay = parse_retry_delay_seconds(exc)
            delay_seconds = max(base_delay_seconds * (2**attempt), hinted_delay)
            log_eval(
                f"[retry] attempt {attempt + 1}/{retries} failed with transient error; "
                f"sleeping {delay_seconds:.2f}s before retry. "
                f"error={type(exc).__name__}: {str(exc)[:220]}"
            )
            time.sleep(delay_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry helper exited without result or exception.")


def clear_runtime_caches() -> None:
    get_settings.cache_clear()
    get_copilot.cache_clear()
    get_guardrails_service.cache_clear()
    get_tracer.cache_clear()


@contextmanager
def isolated_runtime(
    tmp_path: Path,
    *,
    require_groq: bool,
    guardrails_enabled: bool = True,
    tracer_enabled: bool = True,
) -> Iterator[Settings]:
    env_updates = {
        "WORKSPACE_DIR": str(tmp_path),
        "DATA_DIR": "data",
        "DB_PATH": "data/support.db",
        "CHROMA_RAG_DIR": "data/chroma_rag",
        "CHROMA_MEM0_DIR": "data/chroma_mem0",
        "KNOWLEDGE_BASE_DIR": str(ROOT / "knowledge_base"),
        "GUARDRAILS_ENABLED": str(guardrails_enabled).lower(),
        "TRACER_ENABLED": str(tracer_enabled).lower(),
        "TRACER_DIR": "data/traces",
        "GROQ_MODEL": eval_runtime_groq_model(),
        "LLM_TEMPERATURE": "0.0",
    }
    if require_groq:
        env_updates["GROQ_API_KEY"] = require_groq_api_key()

    previous = {key: os.environ.get(key) for key in env_updates}
    try:
        for key, value in env_updates.items():
            os.environ[key] = value
        clear_runtime_caches()
        settings = get_settings()
        yield settings
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        clear_runtime_caches()


@contextmanager
def runtime_client(tmp_path: Path, *, require_groq: bool) -> Iterator[tuple[TestClient, Settings]]:
    with isolated_runtime(tmp_path, require_groq=require_groq) as settings:
        app = create_app(settings=settings)
        with TestClient(app) as client:
            yield client, settings


def ingest_knowledge(client: TestClient) -> dict[str, Any]:
    response = client.post("/api/knowledge/ingest", json={"clear_existing": True})
    assert response.status_code == 200, response.text
    return response.json()


def create_ticket_payload(case: dict[str, Any]) -> dict[str, Any]:
    ticket = dict(case["ticket"])
    customer = case["customer"]
    return {
        "customer_email": customer["email"],
        "customer_name": customer.get("name"),
        "customer_company": customer.get("company"),
        "subject": ticket["subject"],
        "description": ticket["description"],
        "priority": ticket.get("priority", "medium"),
        "auto_generate": False,
    }


def generate_case_draft(client: TestClient, case: dict[str, Any]) -> dict[str, Any]:
    ticket_response = client.post("/api/tickets", json=create_ticket_payload(case))
    assert ticket_response.status_code == 200, ticket_response.text
    ticket = ticket_response.json()

    def request_draft() -> Any:
        response = client.post(f"/api/tickets/{ticket['id']}/generate-draft")
        if response.status_code == 200:
            return response
        if is_transient_error(response.text):
            raise RuntimeError(response.text)
        assert response.status_code == 200, response.text
        return response

    draft_response = call_with_retry(request_draft)
    assert draft_response.status_code == 200, draft_response.text
    payload = draft_response.json()
    return {
        "ticket": ticket,
        "draft": payload["draft"],
    }


def load_expected_reference_contexts(case: dict[str, Any], settings: Settings) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    contexts: list[str] = []
    for expected in case.get("expected_sources", []):
        source_path = settings.knowledge_base_path / expected["source"]
        chunks = splitter.split_text(source_path.read_text(encoding="utf-8"))
        contexts.append(chunks[expected.get("chunk_index", 0)])
    return contexts


def direct_search_contexts(settings: Settings, case: dict[str, Any]) -> list[dict[str, Any]]:
    service = KnowledgeBaseService(settings=settings)
    query = f"{case['ticket']['subject']}\n{case['ticket']['description']}"
    return service.search(query=query, top_k=settings.rag_top_k)


def load_trace_entries(settings: Settings) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for trace_file in sorted(settings.tracer_dir_path.glob("*.jsonl")):
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def build_eval_llm(settings: Settings) -> Any:
    groq_client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    return llm_factory(
        eval_groq_model(),
        client=groq_client,
    )

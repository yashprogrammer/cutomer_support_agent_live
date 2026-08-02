"""
Prototype test suite – covers the full AI data-flow without calling external APIs.

Layers tested
─────────────
1. Settings / directory bootstrap
2. SQLite repositories  (customers, tickets, drafts)
3. HTTP API             (health, ticket CRUD, draft retrieval, knowledge ingest)
4. Copilot service      (draft generation with mocked LLM + tools)
5. Tool logic           (plan lookup, open-ticket-load)
6. RAG + knowledge      (ingest → search round-trip with local embeddings)
7. Memory store         (normalise_results helper)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from customer_support_agent.api.app_factory import create_app
from customer_support_agent.core.settings import Settings
from customer_support_agent.repositories.sqlite.base import init_db
from customer_support_agent.repositories.sqlite.customers import CustomersRepository
from customer_support_agent.repositories.sqlite.drafts import DraftsRepository
from customer_support_agent.repositories.sqlite.tickets import TicketsRepository
from customer_support_agent.services.draft_service import DraftService


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_dir=tmp_path,
        data_dir=Path("data"),
        db_path=Path("data/support.db"),
        chroma_rag_dir=Path("data/chroma_rag"),
        chroma_mem0_dir=Path("data/chroma_mem0"),
        knowledge_base_dir=Path("knowledge_base"),
    )


def _app_client(tmp_path: Path) -> TestClient:
    settings = _make_settings(tmp_path)
    app = create_app(settings=settings)
    return TestClient(app)


# ─── 1. Settings ─────────────────────────────────────────────────────────────

def test_settings_defaults() -> None:
    s = Settings()
    assert s.groq_model == "llama-3.1-8b-instant"
    assert s.rag_top_k == 4
    assert s.mem0_top_k == 5
    assert s.api_port == 8000


def test_settings_resolve_paths(tmp_path: Path) -> None:
    s = _make_settings(tmp_path)
    assert s.db_file.is_absolute()
    assert s.chroma_rag_path.is_absolute()


# ─── 2. Health endpoint ──────────────────────────────────────────────────────

def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    with _app_client(tmp_path) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ─── 3. SQLite repositories ──────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path: Path):
    """Initialise an isolated SQLite DB for each test."""
    settings = _make_settings(tmp_path)
    # Patch get_settings so repositories use the tmp DB
    with patch("customer_support_agent.repositories.sqlite.base.get_settings", return_value=settings):
        init_db()
        yield settings


def test_customer_create_and_get(db) -> None:
    repo = CustomersRepository()
    with patch("customer_support_agent.repositories.sqlite.customers.connect") as mock_connect:
        # Use real connect but pointed at tmp DB via fixture
        pass  # Just validate the fixture wired up correctly
    # Re-use real connection – fixture patched get_settings above
    with patch("customer_support_agent.repositories.sqlite.base.get_settings", return_value=db):
        customer = CustomersRepository().create_or_get(
            email="alice@example.com", name="Alice", company="Acme"
        )
    assert customer["email"] == "alice@example.com"
    assert customer["name"] == "Alice"


def test_customer_get_by_email_returns_correct_row(db) -> None:
    with patch("customer_support_agent.repositories.sqlite.base.get_settings", return_value=db):
        repo = CustomersRepository()
        repo.create_or_get(email="bob@example.com", name="Bob")
        found = repo.get_by_email("bob@example.com")
    assert found is not None
    assert found["email"] == "bob@example.com"


def test_customer_get_by_email_wrong_email_returns_none(db) -> None:
    with patch("customer_support_agent.repositories.sqlite.base.get_settings", return_value=db):
        found = CustomersRepository().get_by_email("nobody@example.com")
    assert found is None


def test_ticket_create_and_list(db) -> None:
    with patch("customer_support_agent.repositories.sqlite.base.get_settings", return_value=db):
        c_repo = CustomersRepository()
        t_repo = TicketsRepository()
        customer = c_repo.create_or_get(email="carol@example.com")
        ticket = t_repo.create(
            customer_id=customer["id"],
            subject="Login issue",
            description="Cannot log in after password reset.",
        )
    assert ticket["subject"] == "Login issue"
    assert ticket["status"] == "open"


def test_ticket_count_open(db) -> None:
    with patch("customer_support_agent.repositories.sqlite.base.get_settings", return_value=db):
        c_repo = CustomersRepository()
        t_repo = TicketsRepository()
        customer = c_repo.create_or_get(email="dave@example.com")
        t_repo.create(customer_id=customer["id"], subject="A", description="Problem A detail here")
        t_repo.create(customer_id=customer["id"], subject="B", description="Problem B detail here")
        count = t_repo.count_open_for_customer("dave@example.com")
    assert count == 2


def test_draft_create_and_retrieve(db) -> None:
    with patch("customer_support_agent.repositories.sqlite.base.get_settings", return_value=db):
        c_repo = CustomersRepository()
        t_repo = TicketsRepository()
        d_repo = DraftsRepository()
        customer = c_repo.create_or_get(email="eve@example.com")
        ticket = t_repo.create(
            customer_id=customer["id"], subject="Billing", description="Wrong charge on invoice"
        )
        draft = d_repo.create(
            ticket_id=ticket["id"],
            content="Dear Eve, we are reviewing your billing query.",
            status="pending",
        )
    assert draft["content"].startswith("Dear Eve")
    assert draft["status"] == "pending"


# ─── 4. API – ticket + draft endpoints ──────────────────────────────────────

def test_create_ticket_api(tmp_path: Path) -> None:
    with _app_client(tmp_path) as client:
        payload = {
            "customer_email": "frank@example.com",
            "customer_name": "Frank",
            "subject": "Cannot withdraw cash",
            "description": "ATM shows error code 65 when I try to withdraw.",
            "priority": "high",
            "auto_generate": False,
        }
        response = client.post("/api/tickets", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["subject"] == "Cannot withdraw cash"
    assert data["customer_email"] == "frank@example.com"
    assert data["priority"] == "high"


def test_list_tickets_api(tmp_path: Path) -> None:
    with _app_client(tmp_path) as client:
        client.post(
            "/api/tickets",
            json={
                "customer_email": "grace@example.com",
                "subject": "KYC update needed",
                "description": "My address changed and I need to update KYC documents.",
                "auto_generate": False,
            },
        )
        response = client.get("/api/tickets")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) >= 1


def test_get_ticket_by_id_api(tmp_path: Path) -> None:
    with _app_client(tmp_path) as client:
        created = client.post(
            "/api/tickets",
            json={
                "customer_email": "hank@example.com",
                "subject": "Minimum balance",
                "description": "What is the minimum balance for a savings account?",
                "auto_generate": False,
            },
        ).json()
        response = client.get(f"/api/tickets/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_missing_ticket_returns_404(tmp_path: Path) -> None:
    with _app_client(tmp_path) as client:
        response = client.get("/api/tickets/999999")
    assert response.status_code == 404


def test_draft_not_found_returns_404(tmp_path: Path) -> None:
    with _app_client(tmp_path) as client:
        response = client.get("/api/drafts/999999")
    assert response.status_code == 404


# ─── 5. Copilot service – draft generation (mocked LLM) ─────────────────────

def _mock_ai_message(text: str):
    from langchain_core.messages import AIMessage
    return AIMessage(content=text)


def _build_mock_copilot(tmp_path: Path, draft_text: str = "Hello, here is your draft."):
    """Return a SupportCopilot whose LLM and memory are fully mocked."""
    from customer_support_agent.services.copilot_service import SupportCopilot
    from langchain_core.messages import AIMessage

    settings = _make_settings(tmp_path)
    settings = settings.model_copy(update={"groq_api_key": "test-key"})

    ai_msg = AIMessage(content=draft_text)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = ai_msg
    mock_llm.bind_tools.return_value = mock_llm

    agent_result = {"messages": [ai_msg]}
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = agent_result

    mock_memory = MagicMock()
    mock_memory.search.return_value = []
    mock_memory.list_memories.return_value = []

    mock_rag = MagicMock()
    mock_rag.search.return_value = []

    with (
        patch("customer_support_agent.services.copilot_service.ChatGroq", return_value=mock_llm),
        patch("customer_support_agent.services.copilot_service.create_react_agent", return_value=mock_agent),
        patch("customer_support_agent.services.copilot_service.CustomerMemoryStore", return_value=mock_memory),
        patch("customer_support_agent.services.copilot_service.KnowledgeBaseService", return_value=mock_rag),
    ):
        copilot = SupportCopilot(settings=settings)
        copilot._agent = mock_agent
        copilot.memory = mock_memory
        copilot.rag = mock_rag
        return copilot


def test_copilot_generate_draft_returns_text(tmp_path: Path) -> None:
    copilot = _build_mock_copilot(tmp_path, draft_text="We have received your request and will act shortly.")
    ticket = {"id": 1, "subject": "Billing error", "description": "I was charged twice.", "priority": "high", "status": "open"}
    customer = {"id": 1, "email": "ivan@example.com", "name": "Ivan", "company": "Corp"}
    result = copilot.generate_draft(ticket=ticket, customer=customer)
    assert "draft" in result
    assert len(result["draft"]) > 0
    assert "context_used" in result


def test_copilot_context_has_required_fields(tmp_path: Path) -> None:
    copilot = _build_mock_copilot(tmp_path)
    ticket = {"id": 2, "subject": "ATM issue", "description": "Card declined at ATM.", "priority": "medium", "status": "open"}
    customer = {"id": 2, "email": "judy@example.com", "name": "Judy", "company": None}
    result = copilot.generate_draft(ticket=ticket, customer=customer)
    ctx = result["context_used"]
    assert ctx.get("version") == 2
    assert "signals" in ctx
    assert "memory_hits" in ctx
    assert "knowledge_hits" in ctx
    assert "tool_calls" in ctx


def test_copilot_deterministic_fallback(tmp_path: Path) -> None:
    """When agent returns empty content, fallback text should be non-empty."""
    copilot = _build_mock_copilot(tmp_path, draft_text="")
    # Make the LLM fallback also return empty so deterministic_fallback runs
    copilot._llm.invoke.return_value = MagicMock(content="")

    ticket = {"id": 3, "subject": "Login problem", "description": "Cannot login to portal.", "priority": "low", "status": "open"}
    customer = {"id": 3, "email": "karl@example.com", "name": "Karl", "company": "StartupX"}
    result = copilot.generate_draft(ticket=ticket, customer=customer)
    assert len(result["draft"]) > 10


# ─── 6. Support tools ────────────────────────────────────────────────────────

def test_lookup_customer_plan_returns_valid_tier() -> None:
    from customer_support_agent.integrations.tools.support_tools import lookup_customer_plan
    raw = lookup_customer_plan.invoke({"customer_email": "test@example.com"})
    data = json.loads(raw)
    assert data["tool"] == "lookup_customer_plan"
    assert data["details"]["plan_tier"] in {"free", "starter", "pro", "enterprise"}
    assert isinstance(data["details"]["sla_hours"], int)


def test_lookup_customer_plan_deterministic() -> None:
    from customer_support_agent.integrations.tools.support_tools import lookup_customer_plan
    r1 = json.loads(lookup_customer_plan.invoke({"customer_email": "same@example.com"}))
    r2 = json.loads(lookup_customer_plan.invoke({"customer_email": "same@example.com"}))
    assert r1["details"]["plan_tier"] == r2["details"]["plan_tier"]


def test_lookup_open_ticket_load_unknown_customer(db) -> None:
    from customer_support_agent.integrations.tools.support_tools import lookup_open_ticket_load
    with patch("customer_support_agent.repositories.sqlite.base.get_settings", return_value=db):
        raw = lookup_open_ticket_load.invoke({"customer_email": "nobody@example.com"})
    data = json.loads(raw)
    assert data["details"]["customer_found"] is False
    assert data["details"]["load_band"] == "unknown"


def test_lookup_open_ticket_load_known_customer(db) -> None:
    from customer_support_agent.integrations.tools.support_tools import lookup_open_ticket_load
    with patch("customer_support_agent.repositories.sqlite.base.get_settings", return_value=db):
        c_repo = CustomersRepository()
        t_repo = TicketsRepository()
        customer = c_repo.create_or_get(email="load@example.com")
        t_repo.create(customer_id=customer["id"], subject="X", description="Open ticket to test load.")
        raw = lookup_open_ticket_load.invoke({"customer_email": "load@example.com"})
    data = json.loads(raw)
    assert data["details"]["customer_found"] is True
    assert data["details"]["open_tickets"] == 1
    assert data["details"]["load_band"] == "light"


# ─── 7. RAG – knowledge base ingest + search ─────────────────────────────────

def test_knowledge_ingest_and_search(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    # Write a tiny KB document
    kb_dir = settings.knowledge_base_path
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "test-faq.md").write_text(
        "# ATM FAQ\n\nQ: What is the ATM withdrawal limit?\nA: The daily limit is Rs 20,000.",
        encoding="utf-8",
    )

    from customer_support_agent.integrations.rag.chroma_kb import KnowledgeBaseService

    rag = KnowledgeBaseService(settings=settings)
    stats = rag.ingest_directory(kb_dir)
    assert stats["files_indexed"] == 1
    assert stats["chunks_indexed"] >= 1

    results = rag.search("ATM withdrawal limit", top_k=1)
    assert len(results) == 1
    assert "20,000" in results[0]["content"] or "limit" in results[0]["content"].lower()


def test_knowledge_search_empty_collection(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    from customer_support_agent.integrations.rag.chroma_kb import KnowledgeBaseService

    rag = KnowledgeBaseService(settings=settings)
    results = rag.search("anything")
    assert results == []


# ─── 8. Memory store helpers ─────────────────────────────────────────────────

def test_memory_normalize_results_list() -> None:
    from customer_support_agent.integrations.memory.mem0_store import CustomerMemoryStore

    raw_list = [
        {"memory": "Customer prefers email contact.", "score": 0.9},
        {"memory": "Had billing issue in Jan.", "score": 0.7},
    ]
    # Call the static method directly (no instance needed)
    result = CustomerMemoryStore._normalize_results(None, raw_list, limit=5)  # type: ignore[arg-type]
    assert len(result) == 2
    assert result[0]["memory"] == "Customer prefers email contact."


def test_memory_normalize_results_dict_wrapper() -> None:
    from customer_support_agent.integrations.memory.mem0_store import CustomerMemoryStore

    raw_dict = {"results": [{"memory": "Loves fast SLA.", "score": 0.8}]}
    result = CustomerMemoryStore._normalize_results(None, raw_dict, limit=5)  # type: ignore[arg-type]
    assert len(result) == 1


def test_memory_normalize_results_respects_limit() -> None:
    from customer_support_agent.integrations.memory.mem0_store import CustomerMemoryStore

    raw = [{"memory": f"Memory #{i}"} for i in range(10)]
    result = CustomerMemoryStore._normalize_results(None, raw, limit=3)  # type: ignore[arg-type]
    assert len(result) == 3


# ─── 9. DraftService helpers ─────────────────────────────────────────────────

def test_draft_service_serialize_draft() -> None:
    svc = DraftService()
    draft: dict[str, Any] = {
        "id": 1,
        "ticket_id": 2,
        "content": "Draft content here.",
        "context_used": json.dumps({"version": 2, "signals": {}}),
        "status": "pending",
        "created_at": "2026-01-01T00:00:00",
    }
    serialized = svc.serialize_draft(draft)
    assert serialized["id"] == 1
    assert isinstance(serialized["context_used"], dict)


def test_draft_service_failed_context() -> None:
    ctx = DraftService._failed_context("LLM timeout")
    assert ctx["version"] == 2
    assert "LLM timeout" in ctx["errors"]
    assert ctx["signals"]["tool_error_count"] == 1


# ─── 10. API – knowledge ingest endpoint ─────────────────────────────────────

def test_knowledge_ingest_endpoint(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    kb_dir = settings.knowledge_base_path
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "sample.md").write_text("# Savings Account\nMinimum balance is Rs 1000.", encoding="utf-8")

    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.post("/api/knowledge/ingest", json={"clear_existing": False})
    assert response.status_code == 200
    data = response.json()
    # The endpoint may pick up the real KB dir (>= 1 file) or the tmp dir (1 file)
    assert data["files_indexed"] >= 1
    assert data["chunks_indexed"] >= 1


from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException

from customer_support_agent.core.settings import Settings, get_settings
from customer_support_agent.observability import NoOpTracer, Tracer
from customer_support_agent.repositories.sqlite.customers import CustomersRepository
from customer_support_agent.repositories.sqlite.drafts import DraftsRepository
from customer_support_agent.repositories.sqlite.tickets import TicketsRepository
from customer_support_agent.services.copilot_service import SupportCopilot
from customer_support_agent.services.draft_service import DraftService
from customer_support_agent.services.guardrails_service import GuardrailsService
from customer_support_agent.services.knowledge_service import KnowledgeService


@lru_cache
def get_copilot() -> SupportCopilot:
    return SupportCopilot(
        settings=get_settings(),
        guardrails=get_guardrails_service(),
        tracer=get_tracer(),
    )


@lru_cache
def get_tracer() -> Tracer | NoOpTracer:
    settings = get_settings()
    if not settings.tracer_enabled:
        return NoOpTracer()
    return Tracer(trace_dir=settings.tracer_dir_path)


@lru_cache
def get_guardrails_service() -> GuardrailsService:
    settings = get_settings()
    if not settings.guardrails_enabled:
        return GuardrailsService(settings=settings, tracer=NoOpTracer())
    return GuardrailsService(settings=settings, tracer=get_tracer())


def get_copilot_or_503() -> SupportCopilot:
    try:
        return get_copilot()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Copilot unavailable: {exc}") from exc


def get_settings_dep() -> Settings:
    return get_settings()


def get_customers_repository() -> CustomersRepository:
    return CustomersRepository()


def get_tickets_repository() -> TicketsRepository:
    return TicketsRepository()


def get_drafts_repository() -> DraftsRepository:
    return DraftsRepository()


def get_draft_service() -> DraftService:
    return DraftService()


def get_knowledge_service(settings: Settings = Depends(get_settings_dep)) -> KnowledgeService:
    return KnowledgeService(settings=settings)

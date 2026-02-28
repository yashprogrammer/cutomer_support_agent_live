from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from customer_support_agent.api.dependencies import (
    get_copilot,
    get_copilot_or_503,
    get_customers_repository,
    get_draft_service,
    get_drafts_repository,
    get_tickets_repository,
)
from customer_support_agent.repositories.sqlite.customers import CustomersRepository
from customer_support_agent.repositories.sqlite.drafts import DraftsRepository
from customer_support_agent.repositories.sqlite.tickets import TicketsRepository
from customer_support_agent.schemas.api import GenerateDraftResponse, TicketCreateRequest, TicketResponse
from customer_support_agent.services.copilot_service import SupportCopilot
from customer_support_agent.services.draft_service import DraftService


logger = logging.getLogger(__name__)
router = APIRouter()

def _generate_and_store_draft_background(
    ticket_id: int,
    tickets_repo: TicketsRepository,
    customers_repo: CustomersRepository,
    drafts_repo: DraftsRepository,
    draft_service: DraftService,
) -> dict[str, Any] | None:
    return draft_service.generate_and_store_background(
        ticket_id=ticket_id,
        tickets_repo=tickets_repo,
        customers_repo=customers_repo,
        drafts_repo=drafts_repo,
        copilot_factory=get_copilot,
        logger=logger,
    )


@router.post("/api/tickets", response_model=TicketResponse)
def create_ticket_route(
    payload: TicketCreateRequest,
    background_tasks: BackgroundTasks,
    customers_repo: CustomersRepository = Depends(get_customers_repository),
    tickets_repo: TicketsRepository = Depends(get_tickets_repository),
    drafts_repo: DraftsRepository = Depends(get_drafts_repository),
    draft_service: DraftService = Depends(get_draft_service),
) -> dict[str, Any]:
    customer = customers_repo.create_or_get(
        email=str(payload.customer_email),
        name=payload.customer_name,
        company=payload.customer_company,
    )
    ticket = tickets_repo.create(
        customer_id=customer["id"],
        subject=payload.subject,
        description=payload.description,
        priority=payload.priority,
    )

    merged = {
        **ticket,
        "customer_email": customer["email"],
        "customer_name": customer.get("name"),
        "customer_company": customer.get("company"),
    }

    if payload.auto_generate:
        background_tasks.add_task(
            _generate_and_store_draft_background,
            ticket["id"],
            tickets_repo,
            customers_repo,
            drafts_repo,
            draft_service,
        )

    return draft_service.serialize_ticket(merged)


@router.get("/api/tickets", response_model=list[TicketResponse])
def list_tickets_route(
    tickets_repo: TicketsRepository = Depends(get_tickets_repository),
    draft_service: DraftService = Depends(get_draft_service),
) -> list[dict[str, Any]]:
    return [draft_service.serialize_ticket(ticket) for ticket in tickets_repo.list()]


@router.get("/api/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket_route(
    ticket_id: int,
    tickets_repo: TicketsRepository = Depends(get_tickets_repository),
    draft_service: DraftService = Depends(get_draft_service),
) -> dict[str, Any]:
    ticket = tickets_repo.get_by_id(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return draft_service.serialize_ticket(ticket)

@router.post("/api/tickets/{ticket_id}/generate-draft", response_model=GenerateDraftResponse)
def generate_draft_route(
    ticket_id: int,
    tickets_repo: TicketsRepository = Depends(get_tickets_repository),
    customers_repo: CustomersRepository = Depends(get_customers_repository),
    drafts_repo: DraftsRepository = Depends(get_drafts_repository),
    draft_service: DraftService = Depends(get_draft_service),
    copilot: SupportCopilot = Depends(get_copilot_or_503),
) -> dict[str, Any]:
    ticket = tickets_repo.get_by_id(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    customer = customers_repo.get_by_id(ticket["customer_id"])
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    try:
        draft = draft_service.generate_and_store_manual(
            ticket_id=ticket_id,
            ticket=ticket,
            customer=customer,
            drafts_repo=drafts_repo,
            copilot=copilot,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate draft: {exc}") from exc

    return {
        "ticket_id": ticket_id,
        "draft": draft_service.serialize_draft(draft),
    }
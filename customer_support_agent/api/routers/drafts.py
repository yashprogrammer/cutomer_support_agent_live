from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from customer_support_agent.api.dependencies import (
    get_copilot,
    get_draft_service,
    get_drafts_repository,
    get_tickets_repository,
)
from customer_support_agent.repositories.sqlite.drafts import DraftsRepository
from customer_support_agent.repositories.sqlite.tickets import TicketsRepository
from customer_support_agent.schemas.api import DraftResponse, DraftUpdateRequest
from customer_support_agent.services.draft_service import DraftService


router = APIRouter()

@router.get("/api/drafts/{ticket_id}", response_model=DraftResponse)
def get_draft_route(
    ticket_id: int,
    drafts_repo: DraftsRepository = Depends(get_drafts_repository),
    draft_service: DraftService = Depends(get_draft_service),
) -> dict:
    draft = drafts_repo.get_latest_for_ticket(ticket_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft_service.serialize_draft(draft)


@router.patch("/api/drafts/{draft_id}", response_model=DraftResponse)
def update_draft_route(
    draft_id: int,
    payload: DraftUpdateRequest,
    drafts_repo: DraftsRepository = Depends(get_drafts_repository),
    tickets_repo: TicketsRepository = Depends(get_tickets_repository),
    draft_service: DraftService = Depends(get_draft_service),
) -> dict:
    existing = drafts_repo.get_by_id(draft_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Draft not found")

    updated = drafts_repo.update(draft_id=draft_id, content=payload.content, status=payload.status)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update draft")

    if payload.status == "accepted":
        relation = drafts_repo.get_ticket_and_customer_by_draft(draft_id)
        if relation:
            tickets_repo.set_status(relation["ticket_id"], "resolved")
            try:
                context_used = draft_service.parse_context_used(updated.get("context_used"))
                get_copilot().save_accepted_resolution(
                    customer_email=relation["customer_email"],
                    customer_company=relation.get("customer_company"),
                    ticket_subject=relation["subject"],
                    ticket_description=relation["description"],
                    draft_content=updated["content"],
                    context_used=context_used,
                )
            except Exception:
                # Draft acceptance should still succeed even if memory save fails.
                pass

    return draft_service.serialize_draft(updated)
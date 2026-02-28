from __future__ import annotations

from typing import Any

from customer_support_agent.repositories.sqlite.base import init_db
from customer_support_agent.repositories.sqlite.customers import CustomersRepository
from customer_support_agent.repositories.sqlite.drafts import DraftsRepository
from customer_support_agent.repositories.sqlite.tickets import TicketsRepository

_customers = CustomersRepository()
_tickets = TicketsRepository()
_drafts = DraftsRepository()

def create_or_get_customer(email: str, name: str | None = None, company: str | None = None) -> dict[str, Any]:
    return _customers.create_or_get(email=email, name=name, company=company)


def get_customer_by_id(customer_id: int) -> dict[str, Any] | None:
    return _customers.get_by_id(customer_id)


def get_customer_by_email(email: str) -> dict[str, Any] | None:
    return _customers.get_by_email(email)

def create_ticket(
    customer_id: int,
    subject: str,
    description: str,
    priority: str = "medium",
    status: str = "open",
) -> dict[str, Any]:
    return _tickets.create(
        customer_id=customer_id,
        subject=subject,
        description=description,
        priority=priority,
        status=status,
    )


def list_tickets(limit: int = 100) -> list[dict[str, Any]]:
    return _tickets.list(limit=limit)


def get_ticket_by_id(ticket_id: int) -> dict[str, Any] | None:
    return _tickets.get_by_id(ticket_id)


def set_ticket_status(ticket_id: int, status: str) -> dict[str, Any] | None:
    return _tickets.set_status(ticket_id=ticket_id, status=status)


def count_open_tickets_for_customer(customer_email: str) -> int:
    return _tickets.count_open_for_customer(customer_email)

def create_draft(
    ticket_id: int,
    content: str,
    context_used: str | None = None,
    status: str = "pending",
) -> dict[str, Any]:
    return _drafts.create(ticket_id=ticket_id, content=content, context_used=context_used, status=status)


def get_latest_draft_for_ticket(ticket_id: int) -> dict[str, Any] | None:
    return _drafts.get_latest_for_ticket(ticket_id)


def get_draft_by_id(draft_id: int) -> dict[str, Any] | None:
    return _drafts.get_by_id(draft_id)


def update_draft(
    draft_id: int,
    content: str | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    return _drafts.update(draft_id=draft_id, content=content, status=status)


def get_ticket_and_customer_by_draft(draft_id: int) -> dict[str, Any] | None:
    return _drafts.get_ticket_and_customer_by_draft(draft_id)


__all__ = [
    "CustomersRepository",
    "TicketsRepository",
    "DraftsRepository",
    "init_db",
    "create_or_get_customer",
    "get_customer_by_id",
    "get_customer_by_email",
    "create_ticket",
    "list_tickets",
    "get_ticket_by_id",
    "set_ticket_status",
    "count_open_tickets_for_customer",
    "create_draft",
    "get_latest_draft_for_ticket",
    "get_draft_by_id",
    "update_draft",
    "get_ticket_and_customer_by_draft",
]
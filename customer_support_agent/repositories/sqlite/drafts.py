from __future__ import annotations

from typing import Any

from customer_support_agent.repositories.sqlite.base import connect, row_to_dict


class DraftsRepository:
    def create(
        self,
        ticket_id: int,
        content: str,
        context_used: str | None = None,
        status: str = "pending",
    ) -> dict[str, Any]:
        with connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO drafts (ticket_id, content, context_used, status)
                VALUES (?, ?, ?, ?)
                """,
                (ticket_id, content, context_used, status),
                )

            draft_id = cursor.lastrowid
            row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
            return row_to_dict(row) or {}
    

    def get_latest_for_ticket(self, ticket_id: int) -> dict[str, Any] | None:
        with connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM drafts
                WHERE ticket_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (ticket_id,),
            ).fetchone()
            return row_to_dict(row)

    def get_by_id(self, draft_id: int) -> dict[str, Any] | None:
        with connect() as conn:
            row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
            return row_to_dict(row)


    def update(
        self,
        draft_id: int,
        content: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        updates: list[str] = []
        values: list[Any] = []

        if content is not None:
            updates.append("content = ?")
            values.append(content)
        if status is not None:
            updates.append("status = ?")
            values.append(status)

        if not updates:
            return self.get_by_id(draft_id)

        with connect() as conn:
            values.append(draft_id)
            conn.execute(f"UPDATE drafts SET {', '.join(updates)} WHERE id = ?", values)
            row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
            return row_to_dict(row)


    def get_ticket_and_customer_by_draft(self, draft_id: int) -> dict[str, Any] | None:
        with connect() as conn:
            row = conn.execute(
                """
                SELECT
                    d.id AS draft_id,
                    d.ticket_id,
                    d.content AS draft_content,
                    d.status AS draft_status,
                    t.subject,
                    t.description,
                    t.status AS ticket_status,
                    c.id AS customer_id,
                    c.email AS customer_email,
                    c.name AS customer_name,
                    c.company AS customer_company
                FROM drafts d
                JOIN tickets t ON t.id = d.ticket_id
                JOIN customers c ON c.id = t.customer_id
                WHERE d.id = ?
                """,
                (draft_id,),
            ).fetchone()
            return row_to_dict(row)
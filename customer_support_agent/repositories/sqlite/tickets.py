from __future__ import annotations

from typing import Any

from customer_support_agent.repositories.sqlite.base import connect, row_to_dict

class TicketsRepository:
    def create(
        self,
        customer_id: int,
        subject: str,
        description: str,
        priority: str = "medium",
        status: str = "open",
    ) -> dict[str, Any]:
        with connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tickets (customer_id, subject, description, priority, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (customer_id, subject, description, priority, status),
            )
            ticket_id = cursor.lastrowid
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            return row_to_dict(row) or {}

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    t.*,
                    c.email AS customer_email,
                    c.name AS customer_name,
                    c.company AS customer_company
                FROM tickets t
                JOIN customers c ON c.id = t.customer_id
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_by_id(self, ticket_id: int) -> dict[str, Any] | None:
        with connect() as conn:
            row = conn.execute(
                """
                SELECT
                    t.*,
                    c.email AS customer_email,
                    c.name AS customer_name,
                    c.company AS customer_company
                FROM tickets t
                JOIN customers c ON c.id = t.customer_id
                WHERE t.id = ?
                """,
                (ticket_id,),
            ).fetchone()
            return row_to_dict(row)
    
    def set_status(self, ticket_id: int, status: str) -> dict[str, Any] | None:
        with connect() as conn:
            conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            return row_to_dict(row)

    def count_open_for_customer(self, customer_email: str) -> int:
        with connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS open_count
                FROM tickets t
                JOIN customers c ON c.id = t.customer_id
                WHERE c.email = ? AND t.status = 'open'
                """,
                (customer_email,),
            ).fetchone()
            return int(row["open_count"]) if row else 0

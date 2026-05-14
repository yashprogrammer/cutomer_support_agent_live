from __future__ import annotations

import json
import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from customer_support_agent.core.settings import Settings
from customer_support_agent.integrations.memory.mem0_store import (
    CustomerMemoryStore,
)
from customer_support_agent.integrations.rag.chroma_kb import KnowledgeBaseService
from customer_support_agent.integrations.tools.support_tools import get_support_tools
from customer_support_agent.observability import NoOpTracer, Tracer
from customer_support_agent.services.guardrails_service import GuardrailsService



class SupportCopilot:
    def __init__(
        self,
        settings: Settings,
        guardrails: GuardrailsService | None = None,
        tracer: Tracer | NoOpTracer | None = None,
    ):
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it in .env before generating drafts."
            )
        self._settings = settings
        self._guardrails = guardrails or GuardrailsService(settings=settings)
        self._tracer = tracer or NoOpTracer()
        self._llm = ChatGroq(
            model=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
        )
        self._tools = get_support_tools()
        self._agent = create_agent(
            model=self._llm,
            tools=self._tools,
            checkpointer=InMemorySaver(),
            name="support_copilot_agent",
        )

        self._memory_error: str | None = None

        try:
            self.memory = CustomerMemoryStore(settings=settings, llm=self._llm)
        except Exception as exc:
            self._memory_error = str(exc)
        self.rag = KnowledgeBaseService(settings=settings)

    
    def generate_draft(self, ticket: dict[str, Any], customer: dict[str, Any]) -> dict[str, Any]:
        input_result = self._guardrails.check_input(
            f"{ticket['subject']}\n{ticket['description']}"
        )
        guardrail_outcomes: dict[str, Any] = {
            "input": input_result.to_dict(),
            "output": None,
        }
        if not input_result.passed:
            context_used = self._build_context(
                ticket=ticket,
                customer=customer,
                memory_hits=[],
                kb_hits=[],
                tool_calls=[],
                guardrail_outcomes=guardrail_outcomes,
            )
            context_used.setdefault("errors", []).append(
                "Input guardrail blocked draft generation before the model call."
            )
            context_used["agent_runtime"] = "guardrail_blocked"
            return {
                "draft": self._guardrails.ESCALATION_MESSAGE,
                "context_used": context_used,
            }

        safe_ticket = self._build_guarded_ticket(ticket)
        trace_customer = self._build_trace_customer(customer)
        query = input_result.sanitized_text
        customer_email = customer["email"]

        memory_hits = self._search_memory_scopes(
            query=query,
            customer_email=customer_email,
            customer_company=customer.get("company"),
            limit=self._settings.mem0_top_k,
        )
        kb_hits = self.rag.search(query=query, top_k=self._settings.rag_top_k)
        requires_tool_checks = self._requires_tool_checks(ticket=safe_ticket)
        prefetched_tool_calls = (
            self._prefetch_tool_calls(ticket=safe_ticket, customer=customer)
            if requires_tool_checks
            else []
        )

        system_prompt = self._build_system_prompt(memory_hits=memory_hits, kb_hits=kb_hits)
        user_prompt = self._build_user_prompt(
            ticket=safe_ticket,
            customer=customer,
            tool_calls=prefetched_tool_calls,
        )
        trace_user_prompt = self._build_user_prompt(
            ticket=safe_ticket,
            customer=trace_customer,
            tool_calls=prefetched_tool_calls,
        )

        draft_text = ""
        agent_error: str | None = None
        tool_calls: list[dict[str, Any]] = list(prefetched_tool_calls)
        used_direct_llm = False
        if tool_calls:
            draft_text = self._direct_generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                trace_user_prompt=trace_user_prompt,
                ticket=safe_ticket,
                kb_hits=kb_hits,
                tool_calls=tool_calls,
                guardrail_outcomes=guardrail_outcomes,
            )
            used_direct_llm = bool(draft_text)
        elif requires_tool_checks:
            with self._tracer.start_span(
                "agent_invoke",
                ticket_id=ticket.get("id"),
                thread_id=self._thread_id_for_ticket(ticket=ticket, customer=customer),
            ) as span:
                span["prompt"] = {
                    "system": system_prompt,
                    "user": trace_user_prompt,
                }
                span["knowledge_hits"] = self._sanitize_for_trace(kb_hits)
                try:
                    agent_result = self._agent.invoke(
                        {
                            "messages": [
                                SystemMessage(content=system_prompt),
                                HumanMessage(content=user_prompt),
                            ]
                        },
                        config={
                            "configurable": {
                                "thread_id": self._thread_id_for_ticket(ticket=ticket, customer=customer),
                            },
                            "recursion_limit": 40,
                        },
                    )
                    draft_text, tool_calls = self._extract_agent_draft_and_tool_calls(agent_result)
                except Exception as exc:
                    agent_error = f"{type(exc).__name__}: {exc}"
                    span["error"] = agent_error
                    draft_text = ""
                    tool_calls = []
                span["tool_calls"] = self._sanitize_for_trace(tool_calls)
                if draft_text:
                    draft_text, output_result = self._apply_output_guardrails(draft_text)
                    guardrail_outcomes["output"] = output_result
                    span["response"] = draft_text
                    span["guardrail_outcomes"] = guardrail_outcomes
        else:
            draft_text = self._direct_generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                trace_user_prompt=trace_user_prompt,
                ticket=safe_ticket,
                kb_hits=kb_hits,
                tool_calls=tool_calls,
                guardrail_outcomes=guardrail_outcomes,
            )
            used_direct_llm = bool(draft_text)

        used_fallback = False
        if requires_tool_checks and not draft_text:
            draft_text = self._fallback_generate_text(
                ticket=safe_ticket,
                customer=customer,
                trace_customer=trace_customer,
                memory_hits=memory_hits,
                kb_hits=kb_hits,
                tool_calls=tool_calls,
                guardrail_outcomes=guardrail_outcomes,
            )
            used_fallback = True
        if not draft_text:
            draft_text = self._deterministic_fallback(ticket=ticket, customer=customer, tool_calls=tool_calls)
            used_fallback = True
            guardrail_outcomes["output"] = {
                "passed": True,
                "sanitized_text": draft_text,
                "violations": [],
                "pii_redacted": False,
            }

        context_used = self._build_context(
            ticket=ticket,
            customer=customer,
            memory_hits=memory_hits,
            kb_hits=kb_hits,
            tool_calls=tool_calls,
            guardrail_outcomes=guardrail_outcomes,
        )
        if self._memory_error:
            context_used.setdefault("errors", []).append(f"Memory disabled: {self._memory_error}")
        if used_fallback:
            context_used.setdefault("errors", []).append(
                "Primary tool-call response had empty content; fallback synthesis was used."
            )
            if agent_error:
                context_used.setdefault("errors", []).append(
                    f"Agent invocation raised: {agent_error}"
                )
        if guardrail_outcomes.get("output") and not guardrail_outcomes["output"]["passed"]:
            context_used.setdefault("errors", []).append(
                "Output guardrail replaced the model response with a deterministic escalation draft."
            )
        context_used["agent_runtime"] = (
            "langchain_create_agent"
            if requires_tool_checks
            else "direct_llm_context_synthesis"
        )

        return {
            "draft": draft_text,
            "context_used": context_used,
        }

    def save_accepted_resolution(
        self,
        customer_email: str,
        customer_company: str | None,
        ticket_subject: str,
        ticket_description: str,
        draft_content: str,
        context_used: dict[str, Any] | None = None,
    ) -> None:
        entity_links = self._extract_entity_links(
            ticket_subject=ticket_subject,
            ticket_description=ticket_description,
            draft_content=draft_content,
            context_used=context_used or {},
        )
        for scope_user_id in self._memory_scope_ids(
            customer_email=customer_email,
            customer_company=customer_company,
        ):
            self.memory.add_resolution(
                user_id=scope_user_id,
                ticket_subject=ticket_subject,
                ticket_description=ticket_description,
                accepted_draft=draft_content,
                entity_links=entity_links,
            )

    def list_customer_memories(
        self,
        customer_email: str,
        customer_company: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        scope_user_ids = self._memory_scope_ids(
            customer_email=customer_email,
            customer_company=customer_company,
        )
        raw_hits: list[dict[str, Any]] = []
        for scope_user_id in scope_user_ids:
            hits = self.memory.list_memories(user_id=scope_user_id, limit=max(1, limit))
            raw_hits.extend(self._annotate_memory_scope(hits=hits, scope_user_id=scope_user_id))
        return self._dedupe_memory_hits(raw_hits, limit=max(1, limit))

    def search_customer_memories(
        self,
        customer_email: str,
        query: str,
        customer_company: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self._search_memory_scopes(
            query=query,
            customer_email=customer_email,
            customer_company=customer_company,
            limit=limit,
        )


    def _search_memory_scopes(
        self,
        query: str,
        customer_email: str,
        customer_company: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        per_scope_limit = max(1, limit)
        scope_user_ids = self._memory_scope_ids(
            customer_email=customer_email,
            customer_company=customer_company,
        )
        raw_hits: list[dict[str, Any]] = []
        for scope_user_id in scope_user_ids:
            hits = self.memory.search(query=query, user_id=scope_user_id, limit=per_scope_limit)
            raw_hits.extend(self._annotate_memory_scope(hits=hits, scope_user_id=scope_user_id))
        return self._dedupe_memory_hits(raw_hits, limit=per_scope_limit * len(scope_user_ids))
    
    def _memory_scope_ids(self, customer_email: str, customer_company: str | None) -> list[str]:
        scope_user_ids = [customer_email.strip().lower()]
        company_scope = self._company_scope_user_id(customer_company)
        if company_scope:
            scope_user_ids.append(company_scope)
        return self._unique_ordered(scope_user_ids)

    @staticmethod
    def _company_scope_user_id(customer_company: str | None) -> str | None:
        if not customer_company:
            return None
        lowered = customer_company.strip().lower()
        if not lowered:
            return None
        normalized = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
        if not normalized:
            return None
        return f"company::{normalized}"
    
    @staticmethod
    def _annotate_memory_scope(
        hits: list[dict[str, Any]],
        scope_user_id: str,
    ) -> list[dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        scope = "company" if scope_user_id.startswith("company::") else "customer"
        for hit in hits:
            item = dict(hit)
            metadata = dict(item.get("metadata") or {})
            metadata.setdefault("scope", scope)
            metadata.setdefault("scope_user_id", scope_user_id)
            item["metadata"] = metadata
            annotated.append(item)
        return annotated


    @staticmethod
    def _dedupe_memory_hits(hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            memory_text = str(hit.get("memory", "")).strip()
            if not memory_text:
                continue
            key = memory_text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
            if len(deduped) >= max(1, limit):
                break
        return deduped

    @staticmethod
    def _extract_content(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    @staticmethod
    def _format_memory(memory_hits: list[dict[str, Any]]) -> str:
        if not memory_hits:
            return "- No prior customer memories found."

        lines = []
        for item in memory_hits:
            lines.append(f"- {item.get('memory', '').strip()}")
        return "\n".join(lines)

    @staticmethod
    def _format_kb(kb_hits: list[dict[str, Any]]) -> str:
        if not kb_hits:
            return "- No relevant knowledge-base chunks found."

        lines = []
        for item in kb_hits:
            source = item.get("source", "unknown")
            snippet = item.get("content", "").strip()
            lines.append(f"- [{source}] {snippet}")
        return "\n".join(lines)

    def _build_system_prompt(self, memory_hits: list[dict[str, Any]], kb_hits: list[dict[str, Any]]) -> str:
        return (
            "You are an AI copilot for customer support agents. "
            "Write concise, empathetic, and actionable draft replies. "
            "Only call tools when the ticket explicitly asks about plan benefits, support priority/SLA, "
            "billing/account-specific status, or open ticket load. "
            "For general banking FAQ or policy tickets, answer from the knowledge base alone.\n\n"
            "Customer Memory Context:\n"
            f"{self._format_memory(memory_hits)}\n\n"
            "Knowledge Base Context:\n"
            f"{self._format_kb(kb_hits)}\n\n"
            "Output rules:\n"
            "1) Start with empathy and direct acknowledgement.\n"
            "2) Only state facts that are directly supported by the KB or tool outputs.\n"
            "3) If the context does not provide a specific next step, keep follow-up language generic rather than inventing process details.\n"
            "4) Mention plan/SLA/tool details only when they are directly relevant to the customer's question.\n"
            "5) Keep response under 180 words unless more detail is necessary."
        )

    @staticmethod
    def _build_user_prompt(
        ticket: dict[str, Any],
        customer: dict[str, Any],
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        verified_findings = ""
        if tool_calls:
            summaries = [
                str(item.get("summary") or item.get("output_text") or "").strip()
                for item in tool_calls
                if str(item.get("summary") or item.get("output_text") or "").strip()
            ]
            if summaries:
                verified_findings = (
                    "\n\nVerified tool findings:\n"
                    + "\n".join(f"- {item}" for item in summaries)
                )

        return (
            f"Customer: {customer.get('name') or 'Unknown'} ({customer['email']})\n"
            f"Company: {customer.get('company') or 'Unknown'}\n"
            f"Ticket Subject: {ticket['subject']}\n"
            f"Ticket Priority: {ticket.get('priority', 'medium')}\n"
            f"Ticket Description:\n{ticket['description']}\n\n"
            "Create a draft response for the support agent. "
            "Use tools only if the customer explicitly asks about plan priority/SLA, plan benefits, "
            "billing/account-specific status, or open ticket load. "
            "Otherwise answer directly from the KB context."
            f"{verified_findings}"
        )

    def _requires_tool_checks(self, ticket: dict[str, Any]) -> bool:
        return bool(self._tool_names_for_ticket(ticket))

    @staticmethod
    def _tool_names_for_ticket(ticket: dict[str, Any]) -> list[str]:
        text = f"{ticket.get('subject', '')}\n{ticket.get('description', '')}".lower()
        tool_names: list[str] = []
        plan_markers = (
            "plan",
            "priority",
            "sla",
            "faster support",
            "response priority",
            "support tier",
            "support level",
        )
        load_markers = (
            "open ticket",
            "ticket load",
            "multiple tickets",
        )
        if any(marker in text for marker in plan_markers):
            tool_names.append("lookup_customer_plan")
        if any(marker in text for marker in load_markers):
            tool_names.append("lookup_open_ticket_load")
        return list(dict.fromkeys(tool_names))

    def _prefetch_tool_calls(
        self,
        ticket: dict[str, Any],
        customer: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        arguments = {"customer_email": customer["email"]}
        for tool_name in self._tool_names_for_ticket(ticket):
            tool_calls.append(self._invoke_tool_for_trace(tool_name=tool_name, arguments=arguments))
        return tool_calls

    def _invoke_tool_for_trace(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        trace: dict[str, Any] = {
            "tool_name": tool_name,
            "tool_call_id": f"prefetch::{tool_name}",
            "arguments": dict(arguments),
        }
        tool = next((item for item in self._tools if getattr(item, "name", None) == tool_name), None)
        if tool is None:
            trace.update(
                {
                    "status": "error",
                    "summary": f"Tool '{tool_name}' is not available.",
                    "output": None,
                    "output_text": f"Tool '{tool_name}' is not available.",
                }
            )
            return trace

        try:
            raw_output = tool.invoke(arguments)
            output_text = self._extract_content(raw_output)
            parsed_output, output_text = self._parse_tool_output(output_text)
            trace.update(
                {
                    "status": "ok",
                    "summary": self._tool_summary(parsed_output=parsed_output, output_text=output_text),
                    "output": parsed_output,
                    "output_text": output_text,
                }
            )
            return trace
        except Exception as exc:
            trace.update(
                {
                    "status": "error",
                    "summary": f"Tool '{tool_name}' failed: {exc}",
                    "output": None,
                    "output_text": str(exc),
                }
            )
            return trace

    @staticmethod
    def _thread_id_for_ticket(ticket: dict[str, Any], customer: dict[str, Any]) -> str:
        ticket_id = ticket.get("id")
        if ticket_id is not None:
            return f"ticket::{ticket_id}"

        customer_email = str(customer.get("email") or "").strip().lower()
        if customer_email:
            return f"ticket::{customer_email}"
        return "ticket::unknown"

    def _build_guarded_ticket(self, ticket: dict[str, Any]) -> dict[str, Any]:
        safe_subject, _ = self._guardrails.sanitize_text(str(ticket.get("subject") or ""))
        safe_description, _ = self._guardrails.sanitize_text(str(ticket.get("description") or ""))
        return {
            **ticket,
            "subject": safe_subject,
            "description": safe_description,
        }

    def _build_trace_customer(self, customer: dict[str, Any]) -> dict[str, Any]:
        safe_email, _ = self._guardrails.sanitize_text(str(customer.get("email") or ""))
        return {
            **customer,
            "email": safe_email,
        }

    def _apply_output_guardrails(self, text: str) -> tuple[str, dict[str, Any]]:
        result = self._guardrails.check_output(text)
        if not result.passed:
            return self._guardrails.ESCALATION_MESSAGE, result.to_dict()
        return result.sanitized_text, result.to_dict()

    def _sanitize_for_trace(self, value: Any) -> Any:
        if isinstance(value, str):
            sanitized, _ = self._guardrails.sanitize_text(value)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_for_trace(item) for item in value]
        if isinstance(value, dict):
            return {key: self._sanitize_for_trace(item) for key, item in value.items()}
        return value

    def _extract_agent_draft_and_tool_calls(
        self, agent_result: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        raw_messages: Any
        if isinstance(agent_result, dict):
            raw_messages = agent_result.get("messages") or []
        else:
            raw_messages = getattr(agent_result, "messages", []) or []

        messages = [item for item in raw_messages if isinstance(item, BaseMessage)]

        draft_text = ""
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            candidate = self._extract_content(message).strip()
            if candidate:
                draft_text = candidate
                break

        tool_messages_by_id: dict[str, ToolMessage] = {}
        for message in messages:
            if isinstance(message, ToolMessage) and message.tool_call_id:
                tool_messages_by_id[message.tool_call_id] = message

        tool_calls: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue

            pending_calls = getattr(message, "tool_calls", None) or []
            for call in pending_calls:
                tool_name = call.get("name")
                tool_id = call.get("id")
                args = call.get("args")
                safe_tool_name = tool_name or "unknown_tool"

                trace: dict[str, Any] = {
                    "tool_name": safe_tool_name,
                    "tool_call_id": tool_id,
                    "arguments": args if isinstance(args, dict) else {},
                }

                tool_message = tool_messages_by_id.get(str(tool_id)) if tool_id is not None else None
                if not tool_message:
                    trace.update(
                        {
                            "status": "skipped",
                            "summary": f"Tool '{safe_tool_name}' was requested but no result was returned.",
                            "output": None,
                            "output_text": f"Tool '{safe_tool_name}' produced no output.",
                        }
                    )
                    tool_calls.append(trace)
                    continue

                output_text = self._extract_content(tool_message)
                parsed_output, output_text = self._parse_tool_output(output_text)
                summary = self._tool_summary(parsed_output=parsed_output, output_text=output_text)
                status = "error" if getattr(tool_message, "status", None) == "error" else "ok"
                trace.update(
                    {
                        "status": status,
                        "summary": summary,
                        "output": parsed_output,
                        "output_text": output_text,
                    }
                )
                tool_calls.append(trace)

        return draft_text, tool_calls

    @staticmethod
    def _parse_tool_output(raw_output: Any) -> tuple[dict[str, Any] | None, str]:
        if isinstance(raw_output, dict):
            return raw_output, json.dumps(raw_output)

        output_text = str(raw_output)
        try:
            parsed = json.loads(output_text)
            if isinstance(parsed, dict):
                return parsed, output_text
        except json.JSONDecodeError:
            pass
        return None, output_text

    
    @staticmethod
    def _tool_summary(parsed_output: dict[str, Any] | None, output_text: str) -> str:
        if parsed_output:
            summary = parsed_output.get("summary")
            if summary:
                return str(summary)
        return output_text

    def _build_context(
        self,
        ticket: dict[str, Any],
        customer: dict[str, Any],
        memory_hits: list[dict[str, Any]],
        kb_hits: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        guardrail_outcomes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        knowledge_sources = self._unique_ordered(
            [str(item.get("source")) for item in kb_hits if item.get("source")]
        )
        tool_errors = [item for item in tool_calls if item.get("status") != "ok"]

        return {
            "version": 2,
            "ticket": {
                "id": ticket.get("id"),
                "subject": ticket.get("subject"),
                "priority": ticket.get("priority"),
                "status": ticket.get("status"),
            },
            "customer": {
                "id": customer.get("id"),
                "email": customer.get("email"),
                "name": customer.get("name"),
                "company": customer.get("company"),
            },
            "signals": {
                "memory_hit_count": len(memory_hits),
                "knowledge_hit_count": len(kb_hits),
                "tool_call_count": len(tool_calls),
                "tool_error_count": len(tool_errors),
                "knowledge_sources": knowledge_sources,
            },
            "highlights": {
                "memory": [self._trim_text(item.get("memory", "")) for item in memory_hits[:3]],
                "knowledge": [
                    self._trim_text(
                        f"[{item.get('source', 'unknown')}] {item.get('content', '')}"
                    )
                    for item in kb_hits[:3]
                ],
                "tools": [self._trim_text(item.get("summary", "")) for item in tool_calls[:3]],
            },
            "memory_hits": memory_hits,
            "knowledge_hits": kb_hits,
            "tool_calls": tool_calls,
            "guardrail_outcomes": guardrail_outcomes or {},
        }

    
    @staticmethod
    def _unique_ordered(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    @staticmethod
    def _trim_text(text: Any, limit: int = 180) -> str:
        clean = str(text or "").strip()
        if len(clean) <= limit:
            return clean
        return f"{clean[: limit - 3]}..."

    def _extract_entity_links(
        self,
        ticket_subject: str,
        ticket_description: str,
        draft_content: str,
        context_used: dict[str, Any],
    ) -> list[str]:
        merged_text = f"{ticket_subject}\n{ticket_description}\n{draft_content}"
        merged_lower = merged_text.lower()
        links: list[str] = []

        endpoints = re.findall(r"/[a-zA-Z0-9][a-zA-Z0-9/_-]{2,}", merged_text)
        for endpoint in self._unique_ordered(endpoints)[:3]:
            links.append(f"endpoint:{endpoint}")

        status_codes = re.findall(r"\b([45]\d\d)\b", merged_text)
        for code in self._unique_ordered(status_codes)[:4]:
            links.append(f"http_status:{code}")

        regions = [
            ("EU", [" eu ", "europe", "emea"]),
            ("US", [" us ", "united states", "na "]),
            ("APAC", [" apac ", "asia pacific"]),
            ("India", [" india ", " in "]),
        ]
        padded = f" {merged_lower} "
        for region, markers in regions:
            if any(marker in padded for marker in markers):
                links.append(f"region:{region}")

        integrations = ["shopify", "stripe", "salesforce", "slack", "quickbooks", "hubspot", "zendesk"]
        for integration in integrations:
            if integration in merged_lower:
                links.append(f"integration:{integration}")

        for tool_call in context_used.get("tool_calls", []):
            output = tool_call.get("output") or {}
            details = output.get("details") if isinstance(output, dict) else None
            if not isinstance(details, dict):
                continue
            plan = details.get("plan_tier")
            if plan:
                links.append(f"plan:{plan}")
            risk = details.get("risk_level")
            if risk:
                links.append(f"billing_risk:{risk}")

        return self._unique_ordered([item for item in links if item])[:12]



    def _fallback_generate_text(
        self,
        ticket: dict[str, Any],
        customer: dict[str, Any],
        trace_customer: dict[str, Any],
        memory_hits: list[dict[str, Any]],
        kb_hits: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        guardrail_outcomes: dict[str, Any],
    ) -> str:
        tool_summaries = [
            self._trim_text(item.get("summary") or item.get("output_text", ""))
            for item in tool_calls
            if item.get("summary") or item.get("output_text")
        ]
        memory_summaries = [self._trim_text(item.get("memory", "")) for item in memory_hits[:3]]
        kb_summaries = [
            self._trim_text(f"[{item.get('source', 'unknown')}] {item.get('content', '')}")
            for item in kb_hits[:3]
        ]

        fallback_system = (
            "You are an AI support copilot. Produce only the final customer-facing draft reply. "
            "Ground every factual statement in the provided knowledge and tool context. "
            "Do not invent unsupported process steps or timelines. "
            "Only mention tool findings when they are directly relevant to the customer's question. "
            "No tool calls."
        )
        fallback_user = (
            f"Customer: {customer.get('name') or 'Unknown'} ({customer.get('email', 'unknown')})\n"
            f"Company: {customer.get('company') or 'Unknown'}\n"
            f"Ticket subject: {ticket.get('subject', '')}\n"
            f"Ticket description: {ticket.get('description', '')}\n\n"
            "Memory highlights:\n"
            f"{chr(10).join('- ' + item for item in memory_summaries) if memory_summaries else '- none'}\n\n"
            "Knowledge highlights:\n"
            f"{chr(10).join('- ' + item for item in kb_summaries) if kb_summaries else '- none'}\n\n"
            "Tool findings:\n"
            f"{chr(10).join('- ' + item for item in tool_summaries) if tool_summaries else '- none'}\n\n"
            "Write a concise, empathetic draft. "
            "Answer the customer's actual question first, and only include next steps that are directly supported by the context."
        )
        trace_fallback_user = (
            f"Customer: {trace_customer.get('name') or 'Unknown'} ({trace_customer.get('email', 'unknown')})\n"
            f"Company: {trace_customer.get('company') or 'Unknown'}\n"
            f"Ticket subject: {ticket.get('subject', '')}\n"
            f"Ticket description: {ticket.get('description', '')}\n\n"
            "Memory highlights:\n"
            f"{chr(10).join('- ' + item for item in memory_summaries) if memory_summaries else '- none'}\n\n"
            "Knowledge highlights:\n"
            f"{chr(10).join('- ' + item for item in kb_summaries) if kb_summaries else '- none'}\n\n"
            "Tool findings:\n"
            f"{chr(10).join('- ' + item for item in tool_summaries) if tool_summaries else '- none'}\n\n"
            "Write a concise, empathetic draft. "
            "Answer the customer's actual question first, and only include next steps that are directly supported by the context."
        )

        try:
            with self._tracer.start_span(
                "draft_fallback_invoke",
                ticket_id=ticket.get("id"),
            ) as span:
                span["prompt"] = {
                    "system": fallback_system,
                    "user": trace_fallback_user,
                }
                span["knowledge_hits"] = self._sanitize_for_trace(kb_hits)
                span["tool_calls"] = self._sanitize_for_trace(tool_calls)
                response = self._llm.invoke(
                    [
                        SystemMessage(content=fallback_system),
                        HumanMessage(content=fallback_user),
                    ]
                )
                draft_text = self._extract_content(response).strip()
                if draft_text:
                    draft_text, output_result = self._apply_output_guardrails(draft_text)
                    guardrail_outcomes["output"] = output_result
                    span["response"] = draft_text
                    span["guardrail_outcomes"] = guardrail_outcomes
                return draft_text
        except Exception:
            return ""

    def _direct_generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        trace_user_prompt: str,
        ticket: dict[str, Any],
        kb_hits: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        guardrail_outcomes: dict[str, Any],
    ) -> str:
        try:
            with self._tracer.start_span(
                "draft_direct_invoke",
                ticket_id=ticket.get("id"),
            ) as span:
                span["prompt"] = {
                    "system": system_prompt,
                    "user": trace_user_prompt,
                }
                span["knowledge_hits"] = self._sanitize_for_trace(kb_hits)
                span["tool_calls"] = self._sanitize_for_trace(tool_calls)
                response = self._llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                draft_text = self._extract_content(response).strip()
                if draft_text:
                    draft_text, output_result = self._apply_output_guardrails(draft_text)
                    guardrail_outcomes["output"] = output_result
                    span["response"] = draft_text
                    span["guardrail_outcomes"] = guardrail_outcomes
                return draft_text
        except Exception:
            return ""

    def _deterministic_fallback(
        self,
        ticket: dict[str, Any],
        customer: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> str:
        customer_name = customer.get("name") or customer.get("email") or "there"
        best_tool_summary = ""
        for item in tool_calls:
            summary = str(item.get("summary") or "").strip()
            if summary:
                best_tool_summary = summary
                break

        action_line = (
            best_tool_summary
            if best_tool_summary
            else "Our support team is reviewing your account and issue details now."
        )

        return (
            f"Hi {customer_name},\n\n"
            f"Thanks for reaching out about \"{ticket.get('subject', 'your issue')}\". "
            "I understand how disruptive this can be.\n\n"
            f"{action_line}\n\n"
            "Next, we will continue investigating and share an update with concrete steps shortly.\n\n"
            "Best,\nSupport Team"
        )

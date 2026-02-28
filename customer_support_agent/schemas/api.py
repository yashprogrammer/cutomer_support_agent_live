from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, EmailStr, Field


class TicketCreateRequest(BaseModel):
    customer_email: EmailStr
    customer_name: str | None = None
    customer_company: str | None = None
    subject: str = Field(min_length=3)
    description: str = Field(min_length=10)
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    auto_generate: bool = True


class TicketResponse(BaseModel):
    id: int
    customer_id: int
    customer_email: EmailStr
    customer_name: str | None = None
    customer_company: str | None = None
    subject: str
    description: str
    status: str
    priority: str
    created_at: str
    updated_at: str


class DraftSignals(BaseModel):
    memory_hit_count: int = 0
    knowledge_hit_count: int = 0
    tool_call_count: int = 0
    tool_error_count: int = 0
    knowledge_sources: list[str] = Field(default_factory=list)


class DraftHighlights(BaseModel):
    memory: list[str] = Field(default_factory=list)
    knowledge: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class DraftToolCall(BaseModel):
    tool_name: str
    tool_call_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str
    summary: str | None = None
    output: dict[str, Any] | None = None
    output_text: str

class StructuredDraftContext(BaseModel):
    version: int = 2
    ticket: dict[str, Any] | None = None
    customer: dict[str, Any] | None = None
    signals: DraftSignals | dict[str, Any] | None = None
    highlights: DraftHighlights | dict[str, Any] | None = None
    memory_hits: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_hits: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[DraftToolCall | dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

class DraftResponse(BaseModel):
    id: int
    ticket_id: int
    content: str
    context_used: StructuredDraftContext | dict[str, Any] | None = None
    status: str
    created_at: str

class DraftUpdateRequest(BaseModel):
    content: str | None = None
    status: Literal["pending", "accepted", "discarded"] | None = None

class GenerateDraftResponse(BaseModel):
    ticket_id: int
    draft: DraftResponse


class KnowledgeIngestRequest(BaseModel):
    clear_existing: bool = False

class KnowledgeIngestResponse(BaseModel):
    files_indexed: int
    chunks_indexed: int
    collection_count: int


class CustomerMemoriesResponse(BaseModel):
    customer_id: int
    customer_email: EmailStr
    memories: list[dict[str, Any]]



class CustomerMemorySearchResponse(BaseModel):
    customer_id: int
    customer_email: EmailStr
    query: str
    results: list[dict[str, Any]]
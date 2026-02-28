from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from customer_support_agent.api.dependencies import get_knowledge_service
from customer_support_agent.schemas.api import KnowledgeIngestRequest, KnowledgeIngestResponse
from customer_support_agent.services.knowledge_service import KnowledgeService

router = APIRouter()

@router.post("/api/knowledge/ingest", response_model=KnowledgeIngestResponse)
def ingest_knowledge_route(
    payload: KnowledgeIngestRequest,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> dict[str, int]:
    try:
        return knowledge_service.ingest(clear_existing=payload.clear_existing)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

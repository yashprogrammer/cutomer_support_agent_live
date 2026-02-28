from __future__ import annotations

from customer_support_agent.core.settings import Settings
from customer_support_agent.integrations.rag.chroma_kb import KnowledgeBaseService


class KnowledgeService:
    def __init__(self, settings: Settings):
        self._settings = settings

    def ingest(self, clear_existing: bool = False) -> dict[str, int]:
        rag_service = KnowledgeBaseService(settings=self._settings)
        return rag_service.ingest_directory(
            directory=self._settings.knowledge_base_path,
            clear_existing=clear_existing,
        )
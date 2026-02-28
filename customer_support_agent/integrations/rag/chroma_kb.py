from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from customer_support_agent.core.settings import Settings


class KnowledgeBaseService:
    def __init__(self, settings:Settings):
        self._settings = settings
        self._client = chromadb.PersistentClient(path=str(settings.chroma_rag_path))
        self._collection_name = "support_kb"
        self._collection = self._client.get_or_create_collection(name=self._collection_name)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

    def ingest_directory(self, directory: Path, clear_existing: bool = False) -> dict[str, int]:
        if clear_existing:
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
        
        source_files = sorted(
            [
                *directory.glob("*.md"),
                *directory.glob("*.txt"),
            ]
        )

        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for file_path in source_files:
            text = file_path.read_text(encoding="utf-8")
            chunks = self._splitter.split_text(text)

            for index, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:10]
                doc_id = f"{file_path.stem}-{index}-{chunk_hash}"
                docs.append(chunk)
                ids.append(doc_id)
                metadatas.append(
                    {
                        "source": file_path.name,
                        "chunk_index": index,
                    }
                )
            
        if docs:
            # upsert prevents duplicate-id failures when re-ingesting.
            self._collection.upsert(documents=docs, ids=ids, metadatas=metadatas)

        return {
            "files_indexed": len(source_files),
            "chunks_indexed": len(docs),
            "collection_count": self._collection.count(),
        }

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k or self._settings.rag_top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        combined: list[dict[str, Any]] = []
        for i, document in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else None
            combined.append(
                {
                    "content": document,
                    "source": metadata.get("source", "unknown"),
                    "distance": distance,
                }
            )

        return combined



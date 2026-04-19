from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


class Tracer:
    def __init__(self, trace_dir: Path):
        self._trace_dir = trace_dir
        self._trace_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def start_span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        started_at = perf_counter()
        span: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "span_id": uuid.uuid4().hex,
            "name": name,
            "latency_ms": None,
            "attrs": attrs,
            "prompt": None,
            "response": None,
            "knowledge_hits": [],
            "tool_calls": [],
            "guardrail_outcomes": {},
            "error": None,
        }

        try:
            yield span
        except Exception as exc:
            span["error"] = str(exc)
            raise
        finally:
            span["latency_ms"] = round((perf_counter() - started_at) * 1000, 2)
            self._append(span)

    def _append(self, span: dict[str, Any]) -> None:
        trace_file = self._trace_dir / f"{datetime.now(UTC).date().isoformat()}.jsonl"
        with trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(span, ensure_ascii=True, default=str))
            handle.write("\n")


class NoOpTracer:
    @contextmanager
    def start_span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        _ = (name, attrs)
        yield {
            "timestamp": None,
            "span_id": None,
            "name": name,
            "latency_ms": None,
            "attrs": attrs,
            "prompt": None,
            "response": None,
            "knowledge_hits": [],
            "tool_calls": [],
            "guardrail_outcomes": {},
            "error": None,
        }

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from customer_support_agent.api.app_factory import create_app
from customer_support_agent.core.settings import Settings


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    settings = Settings(
        workspace_dir=tmp_path,
        data_dir=Path("data"),
        db_path=Path("data/support.db"),
        chroma_rag_dir=Path("data/chroma_rag"),
        chroma_mem0_dir=Path("data/chroma_mem0"),
        knowledge_base_dir=Path("knowledge_base"),
    )

    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

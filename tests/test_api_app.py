from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest

from api.dependencies import ServiceContainer


class DummyRAGService:
    def health(self) -> Dict[str, Any]:
        return {"status": "ready"}

    def process_documents(self, pdf_dir=None, asynchronous=False) -> Dict[str, Any]:
        payload = {"pdf_dir": pdf_dir, "asynchronous": asynchronous}
        return {"results": [], "debug": payload}

    def list_documents(self) -> List[Dict[str, Any]]:
        return []

    def get_document_context(self, document_id, include_chunks=True, chunk_limit=None):
        return {
            "document": {"file_name": f"{document_id}.pdf"},
            "processing": {},
            "files": {},
            "statistics": {},
            "chunks": [] if include_chunks else None,
        }

    def get_job(self, job_id: str):
        return {"job_id": job_id, "status": "queued"}

    def search_similar(self, query: str, top_k: int = 5):
        return []


class DummyChatService:
    def run_chat(self, **kwargs):
        return {
            "answer": "stub-answer",
            "provider": kwargs.get("provider") or "gemini",
            "latency_ms": 1,
            "sources": [],
        }


@pytest.fixture
def test_app(monkeypatch):
    import api.app as app_module

    def fake_init_dependencies(flask_app):
        flask_app.extensions["services"] = ServiceContainer(
            rag_service=DummyRAGService(),
            chat_service=DummyChatService(),
        )

    monkeypatch.setenv("RAG_API_KEY_DISABLED", "1")
    monkeypatch.setattr(app_module, "init_dependencies", fake_init_dependencies)
    app = app_module.create_app()
    app.config.update(TESTING=True)
    yield app


@pytest.fixture
def client(test_app):
    return test_app.test_client()


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["data"]["status"] == "ok"


def test_chat_endpoint(client):
    resp = client.post(
        "/api/v1/chat",
        json={"query": "Xin chào"},
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["data"]["answer"] == "stub-answer"


def test_documents_process_endpoint(client):
    resp = client.post(
        "/api/v1/documents/process",
        json={"pdf_dir": "data/pdf", "asynchronous": False},
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "results" in payload["data"]

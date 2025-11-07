from __future__ import annotations

from dataclasses import dataclass
from flask import Flask, current_app

from services import ChatService, RAGService


@dataclass
class ServiceContainer:
    rag_service: RAGService
    chat_service: ChatService


def init_dependencies(app: Flask) -> None:
    """
    Instantiate shared services and store them on the Flask app extensions dict.
    """
    rag_service = RAGService()
    chat_service = ChatService(rag_service=rag_service)
    app.extensions["services"] = ServiceContainer(
        rag_service=rag_service,
        chat_service=chat_service,
    )


def get_services() -> ServiceContainer:
    """
    Retrieve the service container for the current request context.
    """
    app = current_app._get_current_object()
    container: ServiceContainer | None = app.extensions.get("services")
    if not container:  # pragma: no cover - defensive branch
        raise RuntimeError("Services not initialized. Call init_dependencies() first.")
    return container

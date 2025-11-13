from __future__ import annotations

from flask import Blueprint, request

from api.dependencies import get_services
from api.schemas import ChatRequest
from api.utils import api_response

chat_bp = Blueprint("chat", __name__, url_prefix="/api/v1")


@chat_bp.post("/chat")
def chat_endpoint():
    payload = ChatRequest.model_validate(request.get_json(silent=True) or {})
    services = get_services()

    history = [msg.model_dump() for msg in payload.history] if payload.history else None
    retrieval_options = payload.resolve_retrieval()
    retrieval_config = retrieval_options.model_dump(exclude={"api_tokens"})
    reranker_tokens = retrieval_options.tokens_dict()

    result = services.chat_service.run_chat(
        query=payload.query,
        history=history,
        provider=payload.provider,
        top_k=payload.top_k,
        documents=payload.documents,
        include_sources=payload.include_sources,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        retrieval_config=retrieval_config,
        reranker_tokens=reranker_tokens,
    )

    return api_response(data=result)

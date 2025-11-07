from __future__ import annotations

from flask import Blueprint

from api.dependencies import get_services
from api.utils import api_response

health_bp = Blueprint("health", __name__)


@health_bp.get("/api/health")
def health():
    services = get_services()
    payload = {
        "status": "ok",
        "components": {
            "rag_service": services.rag_service.health(),
        },
    }
    return api_response(data=payload)

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from flask import Response, g, jsonify


def ensure_request_context() -> None:
    """
    Populate request-scoped metadata (request_id, start time) if missing.
    Should be invoked via Flask before_request hook.
    """
    if not hasattr(g, "request_id"):
        g.request_id = str(uuid.uuid4())
    if not hasattr(g, "request_start"):
        g.request_start = time.perf_counter()


def api_response(
    *,
    data: Optional[Any] = None,
    error: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
) -> Response:
    """
    Standardize envelope returned to clients.
    """
    duration_ms = None
    if hasattr(g, "request_start"):
        duration_ms = int((time.perf_counter() - g.request_start) * 1000)

    body = {
        "data": data,
        "error": error,
        "meta": {
            "request_id": getattr(g, "request_id", None),
            "duration_ms": duration_ms,
            "source": "rag-api",
        },
    }

    response = jsonify(body)
    response.status_code = status_code
    if hasattr(g, "request_id"):
        response.headers["X-Request-ID"] = g.request_id
    return response

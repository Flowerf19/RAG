from __future__ import annotations

import os

from flask import request

from api.errors import APIError

API_KEY_ENV = "RAG_API_KEY"
DISABLE_ENV = "RAG_API_KEY_DISABLED"


def authenticate_request() -> None:
    """
    Basic header-based authentication using X-API-Key.

    If `RAG_API_KEY` env var is not set (or DISABLE env is truthy) the check is skipped,
    making local development easier.
    """
    if os.getenv(DISABLE_ENV, "").lower() in {"1", "true", "yes"}:
        return

    expected_key = os.getenv(API_KEY_ENV)
    if not expected_key:
        # No key configured => skip auth (but warn for visibility)
        return

    provided_key = request.headers.get("X-API-Key")
    if not provided_key or provided_key != expected_key:
        raise APIError(
            "Invalid API key",
            status_code=401,
            code="auth_failed",
        )

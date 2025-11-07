from __future__ import annotations

from typing import Any, Dict, Optional

from flask import Flask
from pydantic import ValidationError

from api.utils import api_response


class APIError(Exception):
    """
    Custom exception for predictable API failures.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "bad_request",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


def register_error_handlers(app: Flask) -> None:
    """Attach JSON error handlers to the Flask app."""

    @app.errorhandler(APIError)
    def handle_api_error(err: APIError):
        return api_response(data=None, error=err.to_dict(), status_code=err.status_code)

    @app.errorhandler(ValidationError)
    def handle_validation_error(err: ValidationError):
        error_payload = {
            "code": "validation_error",
            "message": "Payload validation failed",
            "details": err.errors(),
        }
        return api_response(data=None, error=error_payload, status_code=422)

    @app.errorhandler(404)
    def handle_not_found(err):
        error_payload = {"code": "not_found", "message": "Resource not found"}
        return api_response(data=None, error=error_payload, status_code=404)

    @app.errorhandler(Exception)
    def handle_generic_error(err: Exception):
        error_payload = {
            "code": "internal_error",
            "message": str(err),
        }
        return api_response(data=None, error=error_payload, status_code=500)

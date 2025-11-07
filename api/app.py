from __future__ import annotations

import logging
import os

from flask import Flask
from flask_cors import CORS

from api.auth import authenticate_request
from api.dependencies import init_dependencies
from api.errors import register_error_handlers
from api.routes import register_routes
from api.utils import ensure_request_context

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    init_dependencies(app)
    register_error_handlers(app)
    register_routes(app)

    @app.before_request
    def _request_setup():
        ensure_request_context()
        authenticate_request()

    return app


def main() -> None:
    """Entry point for running the Flask development server."""
    app = create_app()
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") in {"1", "true", "True"}
    logger.info("Starting Flask API on %s:%s (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

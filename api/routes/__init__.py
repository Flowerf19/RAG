from __future__ import annotations

from flask import Flask

from api.routes.chat import chat_bp
from api.routes.documents import documents_bp
from api.routes.health import health_bp


def register_routes(app: Flask) -> None:
    app.register_blueprint(health_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(chat_bp)

"""
Flask API package exposing the RAG pipeline to external clients.
"""

from api.app import create_app

__all__ = ["create_app"]

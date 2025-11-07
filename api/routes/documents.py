from __future__ import annotations

from flask import Blueprint, request

from api.dependencies import get_services
from api.errors import APIError
from api.schemas import ProcessDocumentsRequest
from api.utils import api_response

documents_bp = Blueprint("documents", __name__, url_prefix="/api/v1")


@documents_bp.post("/documents/process")
def process_documents():
    payload = ProcessDocumentsRequest.model_validate(request.get_json(silent=True) or {})
    services = get_services()
    result = services.rag_service.process_documents(
        pdf_dir=payload.pdf_dir,
        asynchronous=payload.asynchronous,
    )
    return api_response(data=result)


@documents_bp.get("/documents")
def list_documents():
    services = get_services()
    docs = services.rag_service.list_documents()
    return api_response(data={"documents": docs})


@documents_bp.get("/documents/<string:document_id>/context")
def get_document_context(document_id: str):
    services = get_services()
    include_chunks = request.args.get("include_chunks", "true").lower() not in {"0", "false", "no"}
    chunk_limit_param = request.args.get("chunk_limit")
    chunk_limit = int(chunk_limit_param) if chunk_limit_param else None
    context = services.rag_service.get_document_context(
        document_id=document_id,
        include_chunks=include_chunks,
        chunk_limit=chunk_limit,
    )
    return api_response(data=context)


@documents_bp.get("/jobs/<string:job_id>")
def get_job(job_id: str):
    services = get_services()
    job = services.rag_service.get_job(job_id)
    if not job:
        raise APIError("Job not found", status_code=404, code="job_not_found")
    return api_response(data=job)

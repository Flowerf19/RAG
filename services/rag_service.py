from __future__ import annotations

import json
import logging
import pickle
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Enumeration of pipeline job states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class JobRecord:
    """Metadata captured for background document-processing jobs."""

    job_id: str
    status: JobStatus
    submitted_at: datetime
    payload: Dict[str, Any]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    future: Optional[Future] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Futures are not serializable / not needed in API responses
        data.pop("future", None)
        # Enum -> value for JSON
        data["status"] = self.status.value
        # Datetime serialization
        for field in ("submitted_at", "started_at", "completed_at"):
            if data[field]:
                data[field] = data[field].isoformat()
        return data


class RAGService:
    """
    Thin orchestration layer that keeps a single RAGPipeline instance warm,
    exposes helper methods for document processing, metadata access, and
    similarity search, and manages background jobs.
    """

    def __init__(
        self,
        pipeline: Optional[RAGPipeline] = None,
        *,
        max_workers: int = 2,
    ) -> None:
        self.pipeline = pipeline or RAGPipeline()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, JobRecord] = {}
        self._job_lock = threading.Lock()

        self.metadata_dir = self.pipeline.metadata_dir
        self.vectors_dir = self.pipeline.vectors_dir
        self._active_summary_path: Optional[Path] = None
        self._active_summary: Optional[Dict[str, Any]] = None

        self._load_latest_summary()

    # ------------------------------------------------------------------#
    # Job management helpers
    # ------------------------------------------------------------------#
    def process_documents(
        self,
        pdf_dir: Optional[str | Path] = None,
        *,
        asynchronous: bool = False,
    ) -> Dict[str, Any]:
        """
        Trigger document processing.

        Args:
            pdf_dir: Optional directory override.
            asynchronous: If True, dispatch to worker pool and return job_id.

        Returns:
            Dict containing immediate results or job metadata.
        """
        if asynchronous:
            record = self._submit_job(pdf_dir)
            return {"job": record.to_dict()}

        logger.info("Processing documents synchronously (pdf_dir=%s)", pdf_dir or "default")
        results = self.pipeline.process_directory(pdf_dir)
        self._refresh_active_summary_from_results(results)
        return {"results": results}

    def _submit_job(self, pdf_dir: Optional[str | Path]) -> JobRecord:
        job_id = f"job-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.QUEUED,
            submitted_at=datetime.utcnow(),
            payload={"pdf_dir": str(pdf_dir) if pdf_dir else None},
        )

        def _runner() -> List[Dict[str, Any]]:
            record.started_at = datetime.utcnow()
            record.status = JobStatus.RUNNING
            logger.info("Job %s started (pdf_dir=%s)", job_id, pdf_dir or "default")
            start = time.perf_counter()
            try:
                results = self.pipeline.process_directory(pdf_dir)
                record.result = results
                record.status = JobStatus.SUCCEEDED
                self._refresh_active_summary_from_results(results)
                return results
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Job %s failed: %s", job_id, exc)
                record.error = str(exc)
                record.status = JobStatus.FAILED
                raise
            finally:
                record.completed_at = datetime.utcnow()
                record.duration_ms = int((time.perf_counter() - start) * 1000)

        future = self.executor.submit(_runner)
        record.future = future

        with self._job_lock:
            self._jobs[job_id] = record

        return record

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return job metadata if present."""
        with self._job_lock:
            record = self._jobs.get(job_id)
            return record.to_dict() if record else None

    # ------------------------------------------------------------------#
    # Metadata helpers
    # ------------------------------------------------------------------#
    def get_document_context(
        self,
        document_id: Optional[str] = None,
        *,
        include_chunks: bool = True,
        chunk_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load metadata summary (and optionally chunk metadata) for a document.

        Args:
            document_id: File stem or filename. When omitted, uses latest summary.
            include_chunks: Whether to include chunk metadata map.
            chunk_limit: Optional cap for number of chunks returned.
        """
        summary_path = self._resolve_summary_path(document_id)
        if not summary_path:
            raise FileNotFoundError("No processed documents found. Run the pipeline first.")

        summary = self._read_json(summary_path)
        context: Dict[str, Any] = {
            "document": summary.get("document", {}),
            "processing": summary.get("processing", {}),
            "files": summary.get("files", {}),
            "statistics": summary.get("statistics", {}),
        }

        if include_chunks:
            metadata_map = summary.get("files", {}).get("metadata_map")
            context["chunks"] = self._load_chunk_metadata(metadata_map, limit=chunk_limit)
        else:
            context["chunks"] = []

        return context

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents discovered in the metadata directory."""
        documents: List[Dict[str, Any]] = []
        for summary_path in sorted(self.metadata_dir.glob("*_summary_*.json")):
            data = self._read_json(summary_path)
            documents.append(
                {
                    "document": data.get("document", {}),
                    "processing": data.get("processing", {}),
                    "files": data.get("files", {}),
                    "statistics": data.get("statistics", {}),
                }
            )
        return documents

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run similarity search against the latest FAISS index."""
        index_paths = self._get_active_index_paths()
        if not index_paths:
            logger.warning("No FAISS index available yet.")
            return []

        results = self.pipeline.search_similar(
            index_paths["faiss"], index_paths["metadata"], query_text=query, top_k=top_k
        )
        return results

    def health(self) -> Dict[str, Any]:
        """Return a lightweight readiness snapshot."""
        index_paths = self._get_active_index_paths()
        return {
            "rag_pipeline": "ready",
            "metadata_dir": str(self.metadata_dir),
            "index_available": bool(index_paths),
            "latest_document": (self._active_summary or {}).get("document", {}),
        }

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _refresh_active_summary_from_results(self, results: List[Dict[str, Any]]) -> None:
        summary_paths: List[Path] = []
        for item in results:
            summary_file = (item.get("files") or {}).get("summary")
            if summary_file:
                summary_paths.append(self._ensure_path(summary_file))

        if summary_paths:
            summary_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            self._set_active_summary(summary_paths[0])
        else:
            self._load_latest_summary()

    def _load_latest_summary(self) -> None:
        summary_files = sorted(
            self.metadata_dir.glob("*_summary_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if summary_files:
            self._set_active_summary(summary_files[0])
        else:
            self._active_summary = None
            self._active_summary_path = None

    def _set_active_summary(self, summary_path: Path) -> None:
        self._active_summary_path = summary_path
        self._active_summary = self._read_json(summary_path)
        logger.info("Active summary set to %s", summary_path.name)

    def _resolve_summary_path(self, document_id: Optional[str]) -> Optional[Path]:
        if document_id:
            stem = Path(document_id).stem
            candidates = sorted(
                self.metadata_dir.glob(f"{stem}_summary_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0]
            return None

        # Fallback to cached latest summary
        if self._active_summary_path and self._active_summary_path.exists():
            return self._active_summary_path

        self._load_latest_summary()
        return self._active_summary_path

    def _get_active_index_paths(self) -> Optional[Dict[str, Path]]:
        summary = self._active_summary
        if not summary:
            summary_path = self._resolve_summary_path(None)
            if not summary_path:
                return None
            summary = self._read_json(summary_path)

        files = summary.get("files") or {}
        faiss_path = files.get("faiss_index")
        metadata_map_path = files.get("metadata_map")
        if not faiss_path or not metadata_map_path:
            return None

        faiss = self._ensure_path(faiss_path)
        metadata = self._ensure_path(metadata_map_path)
        if faiss.exists() and metadata.exists():
            return {"faiss": faiss, "metadata": metadata}
        return None

    def _load_chunk_metadata(
        self,
        metadata_map_path: Optional[str],
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not metadata_map_path:
            return []

        path = self._ensure_path(metadata_map_path)
        if not path.exists():
            logger.warning("Metadata map missing at %s", path)
            return []

        with path.open("rb") as fh:
            metadata_map = pickle.load(fh)

        chunks: List[Dict[str, Any]] = []
        for idx, payload in sorted(metadata_map.items()):
            entry = {
                "chunk_id": payload.get("chunk_id") or f"chunk_{idx}",
                "text": payload.get("text", ""),
                "score": payload.get("similarity_score"),
                "metadata": {
                    "page_number": payload.get("page_number"),
                    "page_numbers": payload.get("page_numbers"),
                    "file_name": payload.get("file_name"),
                    "source_file": payload.get("source_file"),
                    "token_count": payload.get("token_count"),
                    "chunk_index": payload.get("chunk_index"),
                    "block_type": payload.get("block_type"),
                    "is_table": payload.get("is_table"),
                    "is_figure": payload.get("is_figure"),
                },
            }
            chunks.append(entry)
            if limit and len(chunks) >= limit:
                break

        return chunks

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _ensure_path(path_like: str | Path) -> Path:
        path = Path(path_like)
        return path if path.is_absolute() else path.resolve()


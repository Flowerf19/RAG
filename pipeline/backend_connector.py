"""
RAG Retrieval Service
=====================
Module chỉ phụ trách phần Retrieval (FAISS search) để UI có thể hiển thị nguồn
và/hoặc tự ghép context vào prompt. Không gọi LLM tại đây.

Sử dụng nhanh:
    from RAG_system.pipeline.rag_pipeline import RAGPipeline
    from RAG_system.pipeline.rag_qa_engine import RAGRetrievalService

    pipeline = RAGPipeline(output_dir="data")
    retriever = RAGRetrievalService(pipeline)
    results = retriever.retrieve("Tìm điểm chính?", top_k=5)
    context = retriever.build_context(results)
    ui_items = retriever.to_ui_items(results)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from pipeline.rag_pipeline import RAGPipeline


logger = logging.getLogger(__name__)


class RAGRetrievalService:
    """
    Dịch vụ Retrieval thuần: tìm kiếm Top-K đoạn liên quan từ FAISS index và
    cung cấp tiện ích build context + payload hiển thị cho UI.
    Không gọi LLM tại đây (UI hoặc lớp khác sẽ làm việc đó).
    """

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.vector_weight = 0.4  # default contribution from vector search
        self.bm25_weight = 0.6    # default contribution from BM25 search

    def _as_float(self, value: Any, default: float = 0.0) -> float:
        """
        Helper to safely cast values to float while tolerating None or bad types.
        """
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    # ---------- Retrieval utilities ----------
    def _match_metadata_for_vectors(self, vectors_file: Path) -> Optional[Path]:
        """
        Tìm file metadata_map tương ứng với vectors bằng pattern tên file.
        Ví dụ: mydoc_vectors_20250101_120000.faiss => mydoc_metadata_map_20250101_120000.pkl
        """
        name = vectors_file.name
        if "_vectors_" not in name:
            return None
        candidate = self.pipeline.vectors_dir / name.replace("_vectors_", "_metadata_map_").replace(".faiss", ".pkl")
        return candidate if candidate.exists() else None

    def get_all_index_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Lấy tất cả cặp (faiss_index, metadata_map) hợp lệ trong thư mục vectors.
        Trả về list các cặp, không chỉ latest.
        """
        index_pairs = []
        faiss_files = list(self.pipeline.vectors_dir.glob("*_vectors_*.faiss"))

        for vf in faiss_files:
            mf = self._match_metadata_for_vectors(vf)
            if mf is not None and mf.exists():
                # Test if FAISS file can be loaded
                if self._test_faiss_file(vf):
                    index_pairs.append((vf, mf))
                else:
                    logger.warning(f"Skipping corrupted FAISS file: {vf}")

        # Sort by modification time (newest first)
        index_pairs.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
        return index_pairs

    def _test_faiss_file(self, faiss_file: Path) -> bool:
        """
        Test if a FAISS file can be loaded without errors.
        """
        try:
            import faiss
            faiss.read_index(str(faiss_file))
            return True
        except Exception as e:
            logger.warning(f"FAISS file test failed for {faiss_file}: {e}")
            # Don't auto-cleanup for now - let user decide
            # self._cleanup_corrupted_file(faiss_file)
            return False

    def _cleanup_corrupted_file(self, faiss_file: Path) -> None:
        """
        Remove corrupted FAISS file and its corresponding metadata file.
        """
        try:
            # Remove FAISS file
            faiss_file.unlink()
            logger.info(f"Removed corrupted FAISS file: {faiss_file}")

            # Remove corresponding metadata file
            mf = self._match_metadata_for_vectors(faiss_file)
            if mf and mf.exists():
                mf.unlink()
                logger.info(f"Removed corresponding metadata file: {mf}")
        except Exception as e:
            logger.error(f"Failed to cleanup corrupted files: {e}")

    def build_context(self, results: List[Dict[str, Any]], max_chars: int = 8000) -> str:
        """
        Tạo chuỗi context gọn từ danh sách kết quả retrieval (top-k).
        Sử dụng provenance information để tạo source attribution chi tiết hơn.
        Cắt ngắn mỗi chunk để đảm bảo có chỗ cho nhiều sources.
        """
        parts: List[str] = []
        total = 0
        max_per_chunk = max(400, max_chars // 8)  # Mỗi chunk tối đa 400 ký tự để đảm bảo capture keywords
        
        for i, r in enumerate(results, 1):
            file_name = r.get("file_name", "?")
            page = r.get("page_number", "?")
            score = r.get("similarity_score", 0.0)
            text = r.get("text", "")
            
            # Cắt ngắn text
            if len(text) > max_per_chunk:
                text = text[:max_per_chunk] + "..."

            # Enhanced source attribution using provenance if available
            provenance = r.get("provenance")
            if provenance and isinstance(provenance, dict):
                # Use provenance for more detailed source info
                page_nums = provenance.get("page_numbers", [])
                if page_nums:
                    page_range = f"pages {min(page_nums)}-{max(page_nums)}" if len(page_nums) > 1 else f"page {page_nums[0]}"
                else:
                    page_range = f"page {page}"

                source_blocks = provenance.get("source_blocks", [])
                if source_blocks:
                    block_info = f", blocks {len(source_blocks)}"
                else:
                    block_info = ""

                source_info = f"{file_name} ({page_range}{block_info})"
            else:
                # Fallback to basic info
                source_info = f"{file_name} (page {page})"

            # Check for table data
            table_data = r.get("table_data")
            if table_data:
                table_note = " [TABLE DATA]"
            else:
                table_note = ""

            piece = f"[{i}] Source: {source_info}, score {score:.3f}{table_note}\n{text}"
            parts.append(piece)
            total += len(piece)
            if total > max_chars:
                break
        return "\n\n".join(parts)

    def to_ui_items(self, results: List[Dict[str, Any]], max_text_len: int = 500) -> List[Dict[str, Any]]:
        """
        Chuyển danh sách kết quả sang dạng dễ hiển thị ở UI.
        Mỗi item gồm: title, snippet, file_name, page_number, similarity_score.
        """
        ui_items: List[Dict[str, Any]] = []
        for r in results:
            file_name = r.get("file_name", "?")
            page = r.get("page_number", "?")
            score = self._as_float(r.get("similarity_score"))
            text = r.get("text", "") or ""
            snippet = (text[: max_text_len - 3] + "...") if len(text) > max_text_len else text
            ui_items.append(
                {
                    "title": f"{file_name} - trang {page}",
                    "snippet": snippet,
                    "file_name": file_name,
                    "page_number": page,
                    "similarity_score": round(score, 4),
                    "distance": self._as_float(r.get("distance")),
                }
            )
        return ui_items

    # ---------- Hybrid Retrieval ----------
    def _normalize_scores(
        self,
        results: List[Dict[str, Any]],
        score_key: str,
        normalized_key: str,
    ) -> None:
        """
        Convert raw scores to z-scores for stable weighting.
        """
        if not results:
            return

        scores = [self._as_float(r.get(score_key)) for r in results]
        if len(scores) < 2:
            for r in results:
                r[normalized_key] = self._as_float(r.get(score_key))
            return

        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            for r in results:
                r[normalized_key] = self._as_float(r.get(score_key))
            return

        for r in results:
            r[normalized_key] = (self._as_float(r.get(score_key)) - mean) / std_dev

    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        score_key: str,
    ) -> List[Dict[str, Any]]:
        """
        Keep the highest-scoring entry per chunk_id.
        """
        best_by_chunk: Dict[str, Dict[str, Any]] = {}
        for result in results:
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue
            current_best = best_by_chunk.get(chunk_id)
            result_score = self._as_float(result.get(score_key))
            current_score = self._as_float(current_best.get(score_key)) if current_best else None
            if current_best is None or result_score > current_score:
                best_by_chunk[chunk_id] = result
        return list(best_by_chunk.values())

    def _merge_vector_and_bm25(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        *,
        top_k: int,
        vector_weight: float,
        bm25_weight: float,
    ) -> List[Dict[str, Any]]:
        """
        Merge FAISS and BM25 results using weighted normalized scores.
        """
        combined: Dict[str, Dict[str, Any]] = {}

        for res in vector_results:
            chunk_id = res.get("chunk_id")
            if not chunk_id:
                continue
            combined.setdefault(chunk_id, {})["vector"] = res

        for res in bm25_results:
            chunk_id = res.get("chunk_id")
            if not chunk_id:
                continue
            combined.setdefault(chunk_id, {})["bm25"] = res

        merged_results: List[Dict[str, Any]] = []

        for chunk_id, data in combined.items():
            vec = data.get("vector")
            bm25 = data.get("bm25")

            vector_norm = vec.get("vector_normalized_score") if vec else None
            bm25_norm = bm25.get("bm25_normalized_score") if bm25 else None

            weighted_vector = self._as_float(vector_norm) * vector_weight if vector_norm is not None else 0.0
            weighted_bm25 = self._as_float(bm25_norm) * bm25_weight if bm25_norm is not None else 0.0

            total_weight = 0.0
            if vector_norm is not None:
                total_weight += vector_weight
            if bm25_norm is not None:
                total_weight += bm25_weight

            final_score = (weighted_vector + weighted_bm25) / total_weight if total_weight else 0.0

            # Prefer vector text as it includes provenance-rich payload; fallback to BM25 metadata
            text = ""
            if vec:
                text = vec.get("text") or ""
            if not text and bm25:
                text = bm25.get("text") or ""

            file_name = None
            source_path = None
            page_number = None
            page_numbers: List[int] = []

            if vec:
                file_name = vec.get("file_name") or file_name
                source_path = vec.get("file_path") or vec.get("source_path") or source_path
                page_number = vec.get("page_number") or page_number
                page_numbers = vec.get("page_numbers") or page_numbers
            if bm25:
                file_name = bm25.get("file_name") or file_name
                source_path = bm25.get("source_path") or source_path
                page_number = bm25.get("page_number") or page_number
                page_numbers = bm25.get("page_numbers") or page_numbers

            derived_file_name = file_name
            if not derived_file_name:
                if isinstance(source_path, Path):
                    derived_file_name = source_path.name
                elif isinstance(source_path, str):
                    derived_file_name = Path(source_path).name

            merged_results.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "file_name": derived_file_name,
                    "file_path": source_path,
                    "page_number": page_number,
                    "page_numbers": page_numbers,
                    "similarity_score": final_score,
                    "score_components": {
                        "vector_normalized": vector_norm,
                        "bm25_normalized": bm25_norm,
                        "vector_weight": vector_weight if vector_norm is not None else 0.0,
                        "bm25_weight": bm25_weight if bm25_norm is not None else 0.0,
                        "vector_contribution": weighted_vector,
                        "bm25_contribution": weighted_bm25,
                    },
                    "vector_similarity": vec.get("similarity_score") if vec else None,
                    "distance": self._as_float(vec.get("distance")) if vec else 0.0,
                    "bm25_raw_score": bm25.get("bm25_raw_score") if bm25 else None,
                    "keywords": bm25.get("keywords") if bm25 else vec.get("keywords", []),
                    "provenance": (vec or {}).get("provenance") or (bm25 or {}).get("metadata"),
                    "retrieval_mode": "hybrid",
                }
            )

        merged_results.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
        return merged_results[:top_k]

    def retrieve_hybrid(
        self,
        query_text: str,
        top_k: int = 5,
        *,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run FAISS and BM25 searches in parallel and merge results by weighted z-score.
        """
        if vector_weight is None:
            vector_weight = self.vector_weight
        if bm25_weight is None:
            bm25_weight = self.bm25_weight

        weight_sum = vector_weight + bm25_weight
        if weight_sum == 0:
            raise ValueError("At least one of vector_weight or bm25_weight must be greater than zero.")
        vector_weight /= weight_sum
        bm25_weight /= weight_sum

        index_pairs = self.get_all_index_pairs()
        vector_results: List[Dict[str, Any]] = []
        for faiss_file, metadata_file in index_pairs:
            try:
                results = self.pipeline.search_similar(
                    faiss_file=faiss_file,
                    metadata_map_file=metadata_file,
                    query_text=query_text,
                    top_k=top_k * 2,
                )
                vector_results.extend(results)
            except Exception as exc:
                logger.warning("Vector search failed for %s: %s", faiss_file, exc)

        vector_results = self._deduplicate_results(vector_results, score_key="similarity_score")
        self._normalize_scores(vector_results, "similarity_score", "vector_normalized_score")

        bm25_results = self.pipeline.search_bm25(
            query_text,
            top_k=top_k * 2,
        )

        merged = self._merge_vector_and_bm25(
            vector_results,
            bm25_results,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        return merged


def fetch_retrieval(query_text: str, top_k: int = 5, max_chars: int = 8000) -> Dict[str, Any]:
    """
    Hàm tiện ích để retrieval từ FAISS indexes.
    Tự động tìm FAISS index mới nhất và thực hiện search.

    Args:
        query_text: Câu hỏi cần tìm
        top_k: Số lượng kết quả trả về
        max_chars: Độ dài tối đa của context

    Returns:
        Dict với keys: "context" (str), "sources" (list)
    """
    try:
        # Khởi tạo pipeline và retriever
        from pipeline.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        retriever = RAGRetrievalService(pipeline)

        results = retriever.retrieve_hybrid(query_text, top_k=top_k)
        if not results:
            logger.warning("Không tìm thấy kết quả từ hybrid retrieval")
            return {"context": "", "sources": []}

        # Build context
        context = retriever.build_context(results, max_chars=max_chars)

        # Convert to UI format
        sources = retriever.to_ui_items(results)

        return {
            "context": context,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Lỗi trong fetch_retrieval: {e}")
        return {"context": "", "sources": []}

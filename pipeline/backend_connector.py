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

import numpy as np

from pipeline.rag_pipeline import RAGPipeline
from pipeline.query_enhancement import QueryEnhancementModule, load_qem_settings
from llm.config_loader import get_config
from embedders.embedder_type import EmbedderType


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
        self.bm25_weight = 0.6  # default contribution from BM25 search

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
        candidate = self.pipeline.vectors_dir / name.replace(
            "_vectors_", "_metadata_map_"
        ).replace(".faiss", ".pkl")
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

    def build_context(
        self, results: List[Dict[str, Any]], max_chars: int = 8000
    ) -> str:
        """
        Tạo chuỗi context gọn từ danh sách kết quả retrieval (top-k).
        Sử dụng provenance information để tạo source attribution chi tiết hơn.
        Cắt ngắn mỗi chunk để đảm bảo có chỗ cho nhiều sources.
        """
        parts: List[str] = []
        total = 0
        max_per_chunk = max(
            400, max_chars // 8
        )  # Mỗi chunk tối đa 400 ký tự để đảm bảo capture keywords

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
                    page_range = (
                        f"pages {min(page_nums)}-{max(page_nums)}"
                        if len(page_nums) > 1
                        else f"page {page_nums[0]}"
                    )
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

            piece = (
                f"[{i}] Source: {source_info}, score {score:.3f}{table_note}\n{text}"
            )
            parts.append(piece)
            total += len(piece)
            if total > max_chars:
                break
        return "\n\n".join(parts)

    def to_ui_items(
        self, results: List[Dict[str, Any]], max_text_len: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Chuyển danh sách kết quả sang dạng dễ hiển thị ở UI.
        Mỗi item gồm: title, snippet, file_name, page_number, similarity_score, vector_similarity, rerank_score.
        """
        ui_items: List[Dict[str, Any]] = []
        for r in results:
            file_name = r.get("file_name", "?")
            page = r.get("page_number", "?")
            score = self._as_float(r.get("similarity_score"))
            text = r.get("text", "") or ""
            snippet = (
                (text[: max_text_len - 3] + "...") if len(text) > max_text_len else text
            )

            item = {
                "title": f"{file_name} - trang {page}",
                "snippet": snippet,
                "file_name": file_name,
                "page_number": page,
                "similarity_score": round(score, 4),
                "distance": self._as_float(r.get("distance")),
            }

            # Include vector_similarity (raw cosine) if available
            if "vector_similarity" in r and r["vector_similarity"] is not None:
                item["vector_similarity"] = round(
                    self._as_float(r["vector_similarity"]), 4
                )

            # Include rerank_score if present
            if "rerank_score" in r:
                item["rerank_score"] = round(self._as_float(r.get("rerank_score")), 4)

            ui_items.append(item)
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
            current_score = (
                self._as_float(current_best.get(score_key)) if current_best else None
            )
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
            if bm25_norm is None and bm25:
                bm25_norm = self._as_float(bm25.get("bm25_raw_score"))

            weighted_vector = (
                self._as_float(vector_norm) * vector_weight
                if vector_norm is not None
                else 0.0
            )
            weighted_bm25 = (
                self._as_float(bm25_norm) * bm25_weight
                if bm25_norm is not None
                else 0.0
            )

            total_weight = 0.0
            if vector_norm is not None:
                total_weight += vector_weight
            if bm25_norm is not None:
                total_weight += bm25_weight

            final_score = (
                (weighted_vector + weighted_bm25) / total_weight
                if total_weight
                else 0.0
            )

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
                source_path = (
                    vec.get("file_path") or vec.get("source_path") or source_path
                )
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
                        "vector_weight": vector_weight
                        if vector_norm is not None
                        else 0.0,
                        "bm25_weight": bm25_weight if bm25_norm is not None else 0.0,
                        "vector_contribution": weighted_vector,
                        "bm25_contribution": weighted_bm25,
                    },
                    "vector_similarity": vec.get("similarity_score") if vec else None,
                    "distance": self._as_float(vec.get("distance")) if vec else 0.0,
                    "bm25_raw_score": bm25.get("bm25_raw_score") if bm25 else None,
                    "keywords": bm25.get("keywords")
                    if bm25
                    else vec.get("keywords", []),
                    "provenance": (vec or {}).get("provenance")
                    or (bm25 or {}).get("metadata"),
                    "retrieval_mode": "hybrid",
                }
            )

        merged_results.sort(
            key=lambda item: item.get("similarity_score", 0.0), reverse=True
        )
        return merged_results[:top_k]

    def retrieve_hybrid(
        self,
        query_text: str,
        top_k: int = 5,
        *,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        query_embedding: Optional[List[float]] = None,
        bm25_query: Optional[str] = None,
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
            raise ValueError(
                "At least one of vector_weight or bm25_weight must be greater than zero."
            )
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
                    query_embedding=query_embedding,
                )
                vector_results.extend(results)
            except Exception as exc:
                logger.warning("Vector search failed for %s: %s", faiss_file, exc)

        vector_results = self._deduplicate_results(
            vector_results, score_key="similarity_score"
        )
        self._normalize_scores(
            vector_results, "similarity_score", "vector_normalized_score"
        )

        bm25_input = bm25_query if bm25_query is not None else query_text
        if bm25_input:
            bm25_results = self.pipeline.search_bm25(
                bm25_input,
                top_k=top_k * 2,
            )
        else:
            bm25_results = []
        self._normalize_scores(bm25_results, "bm25_raw_score", "bm25_normalized_score")
        merged = self._merge_vector_and_bm25(
            vector_results,
            bm25_results,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        return merged


def _fuse_query_embeddings(
    queries: List[str],
    embedder,
    logger: logging.Logger,
) -> Optional[List[float]]:
    """
    Generate embeddings for each query and return their mean vector.
    """
    vectors: List[np.ndarray] = []
    for query in queries:
        try:
            embedding = embedder.embed(query)
            if embedding:
                vectors.append(np.asarray(embedding, dtype=np.float32))
        except Exception as exc:
            logger.warning("Failed to embed query '%s': %s", query, exc)

    if not vectors:
        return None

    stacked = np.stack(vectors, axis=0)
    mean_vector = stacked.mean(axis=0)
    return mean_vector.astype(np.float32).tolist()


def fetch_retrieval(
    query_text: str,
    top_k: int = 5,
    max_chars: int = 8000,
    embedder_type: str = "ollama",
    reranker_type: str = "none",
    use_query_enhancement: bool = True,
    api_tokens: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Enhanced retrieval function combining query enhancement and reranking.

    Flow: Query Enhancement -> Embedding Retrieval -> Reranking (optional) -> Final Results

    Args:
        query_text: Original query text
        top_k: Number of final results to return
        max_chars: Maximum context length
        embedder_type: Type of embedder ("ollama", "huggingface_local", "huggingface_api")
        reranker_type: Type of reranker ("none", "bge_local", "bge_m3_ollama", etc.)
        use_query_enhancement: Whether to use query enhancement module
        api_tokens: Dict of API tokens for rerankers (keys: "hf", "cohere", "jina")

    Returns:
        Dict with keys: "context" (str), "sources" (list), "queries" (list), "retrieval_info" (dict)
    """
    try:
        # Query Enhancement
        expanded_queries = [query_text]
        if use_query_enhancement:
            try:
                app_config = get_config()
                qem_settings = load_qem_settings()
                qem = QueryEnhancementModule(app_config, qem_settings, logger=logger)
                expanded_queries = qem.enhance(query_text)
                expanded_queries = [q for q in expanded_queries if q and q.strip()]
                if not expanded_queries:
                    expanded_queries = [query_text]
            except Exception as exc:
                logger.warning(
                    "Query enhancement failed, using original query: %s", exc
                )
                expanded_queries = [query_text]

        # Setup embedder
        embedder_enum = EmbedderType.OLLAMA
        use_api = None

        if embedder_type.lower() == "huggingface_local":
            embedder_enum = EmbedderType.HUGGINGFACE
            use_api = False
        elif embedder_type.lower() == "huggingface_api":
            embedder_enum = EmbedderType.HUGGINGFACE
            use_api = True
        elif embedder_type.lower() == "huggingface":
            embedder_enum = EmbedderType.HUGGINGFACE
            use_api = None
        elif embedder_type.lower() == "ollama":
            embedder_enum = EmbedderType.OLLAMA

        # Initialize pipeline
        pipeline = RAGPipeline(embedder_type=embedder_enum, hf_use_api=use_api)
        retriever = RAGRetrievalService(pipeline)

        # Create fused embedding for multiple queries
        fusion_inputs = expanded_queries
        fused_embedding = _fuse_query_embeddings(
            fusion_inputs, pipeline.embedder, logger
        )
        bm25_query = " ".join(fusion_inputs).strip()

        # Hybrid retrieval - get more results for potential reranking
        retrieval_top_k = top_k * 2 if reranker_type != "none" else top_k
        results = retriever.retrieve_hybrid(
            query_text=query_text,
            top_k=retrieval_top_k,
            query_embedding=fused_embedding,
            bm25_query=bm25_query,
        )

        if not results:
            logger.warning("No hybrid retrieval results found.")
            return {
                "context": "",
                "sources": [],
                "queries": expanded_queries,
                "retrieval_info": {
                    "total_retrieved": 0,
                    "reranked": False,
                    "embedder": embedder_type,
                    "reranker": reranker_type,
                    "query_enhanced": use_query_enhancement,
                },
            }

        initial_count = len(results)
        logger.info(f"Retrieved {initial_count} results from hybrid search")

        # Apply reranking if specified
        reranked = False
        if reranker_type and reranker_type != "none":
            try:
                from reranking.reranker_factory import RerankerFactory
                from reranking.reranker_type import RerankerType

                # Map reranker_type string to enum
                reranker_enum = None
                if reranker_type == "bge_local":
                    reranker_enum = RerankerType.BGE_RERANKER
                elif reranker_type == "bge_m3_ollama":
                    reranker_enum = RerankerType.BGE_M3_OLLAMA
                elif reranker_type == "bge_m3_hf_api":
                    reranker_enum = RerankerType.BGE_M3_HF_API
                elif reranker_type == "bge_m3_hf_local":
                    reranker_enum = RerankerType.BGE_M3_HF_LOCAL
                elif reranker_type == "cohere":
                    reranker_enum = RerankerType.COHERE
                elif reranker_type == "jina":
                    reranker_enum = RerankerType.JINA

                if reranker_enum:
                    # Get API token for API-based rerankers
                    api_token = None
                    if api_tokens:
                        if reranker_type == "bge_m3_hf_api":
                            api_token = api_tokens.get("hf")
                        elif reranker_type == "cohere":
                            api_token = api_tokens.get("cohere")
                        elif reranker_type == "jina":
                            api_token = api_tokens.get("jina")

                    reranker = RerankerFactory.create(
                        reranker_enum, api_token=api_token
                    )
                    doc_texts = [r.get("text", "") for r in results]

                    # Rerank and directly get top_k results (no need to rerank all then slice)
                    reranked_results = reranker.rerank(
                        query_text, doc_texts, top_k=top_k
                    )

                    # Reorder results based on reranking
                    reranked_indices = [rr.index for rr in reranked_results]
                    results = [results[i] for i in reranked_indices]

                    # Add rerank_score (keep original similarity_score intact)
                    for i, rr in enumerate(reranked_results):
                        results[i]["rerank_score"] = rr.score
                        logger.debug(
                            f"Result {i}: hybrid={results[i].get('similarity_score'):.4f}, rerank={rr.score:.4f}"
                        )

                    reranked = True
                    logger.info(
                        f"Applied {reranker_type} reranking: {initial_count} -> {len(results)} results"
                    )

            except Exception as e:
                logger.warning(
                    f"Reranking failed ({reranker_type}): {e}. Using top {top_k} from original results."
                )
                results = results[:top_k]
        else:
            results = results[:top_k]

        # Build final output
        context = retriever.build_context(results, max_chars=max_chars)
        sources = retriever.to_ui_items(results)

        return {
            "context": context,
            "sources": sources,
            "queries": expanded_queries,
            "retrieval_info": {
                "total_retrieved": initial_count,
                "final_count": len(results),
                "reranked": reranked,
                "embedder": embedder_type,
                "reranker": reranker_type if reranked else "none",
                "query_enhanced": use_query_enhancement,
            },
        }

    except Exception as exc:
        logger.error("Error in fetch_retrieval: %s", exc)
        return {
            "context": "",
            "sources": [],
            "queries": [query_text],
            "retrieval_info": {
                "error": str(exc),
                "embedder": embedder_type,
                "reranker": reranker_type,
                "query_enhanced": use_query_enhancement,
            },
        }

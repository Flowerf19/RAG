"""
Retrieval Service Module
========================
Manages RAG retrieval operations and result formatting.
Single Responsibility: FAISS + BM25 search coordination and result formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.retrieval.score_fusion import ScoreFusion

logger = logging.getLogger(__name__)


class RAGRetrievalService:
    """
    RAG retrieval service: searches FAISS index and BM25, provides context building utilities.
    Does NOT call LLM (UI or other layers handle that).
    """

    def __init__(self, pipeline):
        """
        Initialize retrieval service.
        
        Args:
            pipeline: RAGPipeline instance
        """
        self.pipeline = pipeline
        self.vector_weight = 0.4  # default contribution from vector search
        self.bm25_weight = 0.6  # default contribution from BM25 search
        self.score_fusion = ScoreFusion()

    def _match_metadata_for_vectors(self, vectors_file: Path) -> Optional[Path]:
        """
        Find metadata_map file corresponding to a vectors file by name pattern.
        Example: mydoc_vectors_20250101_120000.faiss => mydoc_metadata_map_20250101_120000.pkl
        
        Args:
            vectors_file: Path to FAISS vectors file
            
        Returns:
            Path to metadata file or None if not found
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
        Get all valid (faiss_index, metadata_map) pairs from vectors directory.
        
        Returns:
            List of (faiss_file, metadata_file) tuples, sorted by modification time (newest first)
        """
        index_pairs = []
        faiss_files = list(self.pipeline.vectors_dir.glob("*_vectors_*.faiss"))

        for vf in faiss_files:
            mf = self._match_metadata_for_vectors(vf)
            if mf is not None and mf.exists():
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
        
        Args:
            faiss_file: Path to FAISS file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            import faiss
            faiss.read_index(str(faiss_file))
            return True
        except Exception as e:
            logger.warning(f"FAISS file test failed for {faiss_file}: {e}")
            return False

    def build_context(
        self,
        results: List[Dict[str, Any]],
        max_chars: int = 8000,
    ) -> str:
        """
        Create concise context string from retrieval results (top-k).
        Uses provenance information for detailed source attribution.
        Truncates each chunk to ensure space for multiple sources.
        
        Args:
            results: List of result dictionaries
            max_chars: Maximum context length
            
        Returns:
            Formatted context string
        """
        parts: List[str] = []
        total = 0
        max_per_chunk = max(
            400, max_chars // 8
        )  # Each chunk max 400 chars to capture keywords

        for i, r in enumerate(results, 1):
            file_name = r.get("file_name", "?")
            page = r.get("page_number", "?")
            score = r.get("similarity_score", 0.0)
            text = r.get("text", "")

            # Truncate text
            if len(text) > max_per_chunk:
                text = text[:max_per_chunk] + "..."

            # Enhanced source attribution using provenance
            provenance = r.get("provenance")
            if provenance and isinstance(provenance, dict):
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
                block_info = f", blocks {len(source_blocks)}" if source_blocks else ""
                source_info = f"{file_name} ({page_range}{block_info})"
            else:
                source_info = f"{file_name} (page {page})"

            # Check for table data
            table_note = " [TABLE DATA]" if r.get("table_data") else ""

            piece = f"[{i}] Source: {source_info}, score {score:.3f}{table_note}\n{text}"
            parts.append(piece)
            total += len(piece)
            
            if total > max_chars:
                break
                
        return "\n\n".join(parts)

    def to_ui_items(
        self,
        results: List[Dict[str, Any]],
        max_text_len: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Convert results to UI-friendly format.
        Each item includes: title, snippet, file_name, page_number, similarity_score,
        vector_similarity, rerank_score.
        
        Args:
            results: List of result dictionaries
            max_text_len: Maximum snippet length
            
        Returns:
            List of UI-ready dictionaries
        """
        ui_items: List[Dict[str, Any]] = []
        
        for r in results:
            file_name = r.get("file_name", "?")
            page = r.get("page_number", "?")
            score = self.score_fusion._as_float(r.get("similarity_score"))
            text = r.get("text", "") or ""
            snippet = (
                (text[: max_text_len - 3] + "...")
                if len(text) > max_text_len
                else text
            )

            item = {
                "title": f"{file_name} - trang {page}",
                "snippet": snippet,
                "full_text": text,
                "file_name": file_name,
                "page_number": page,
                "similarity_score": round(score, 4),
                "distance": self.score_fusion._as_float(r.get("distance")),
            }

            # Include vector_similarity if available
            if "vector_similarity" in r and r["vector_similarity"] is not None:
                item["vector_similarity"] = round(
                    self.score_fusion._as_float(r["vector_similarity"]), 4
                )

            # Include rerank_score if present
            if "rerank_score" in r:
                item["rerank_score"] = round(
                    self.score_fusion._as_float(r.get("rerank_score")), 4
                )

            ui_items.append(item)
            
        return ui_items

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
        
        Args:
            query_text: Query string
            top_k: Number of final results
            vector_weight: Weight for vector scores (default: self.vector_weight)
            bm25_weight: Weight for BM25 scores (default: self.bm25_weight)
            query_embedding: Optional precomputed query embedding
            bm25_query: Optional query string for BM25 (default: query_text)
            
        Returns:
            Merged and sorted list of results
        """
        if vector_weight is None:
            vector_weight = self.vector_weight
        if bm25_weight is None:
            bm25_weight = self.bm25_weight

        # Normalize weights
        weight_sum = vector_weight + bm25_weight
        if weight_sum == 0:
            raise ValueError(
                "At least one of vector_weight or bm25_weight must be greater than zero."
            )
        vector_weight /= weight_sum
        bm25_weight /= weight_sum

        # Vector search across all indexes
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

        # Deduplicate and normalize vector results
        vector_results = self.score_fusion.deduplicate_results(
            vector_results, score_key="similarity_score"
        )
        self.score_fusion.normalize_scores(
            vector_results, "similarity_score", "vector_normalized_score"
        )

        # BM25 search
        bm25_input = bm25_query if bm25_query is not None else query_text
        if bm25_input:
            bm25_results = self.pipeline.search_bm25(bm25_input, top_k=top_k * 2)
        else:
            bm25_results = []
            
        self.score_fusion.normalize_scores(
            bm25_results, "bm25_raw_score", "bm25_normalized_score"
        )

        # Merge results
        merged = self.score_fusion.merge_vector_and_bm25(
            vector_results,
            bm25_results,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        
        return merged

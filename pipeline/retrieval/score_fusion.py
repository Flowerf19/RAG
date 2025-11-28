"""
Score Fusion Module
===================
Handles score normalization, deduplication, and merging for hybrid retrieval.
Single Responsibility: Score computation and result merging.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ScoreFusion:
    """
    Manages score normalization and fusion for hybrid retrieval (vector + BM25).
    """

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        """
        Helper to safely cast values to float while tolerating None or bad types.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Float value or default
        """
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def normalize_scores(
        results: List[Dict[str, Any]],
        score_key: str,
        normalized_key: str,
    ) -> None:
        """
        Convert raw scores to z-scores for stable weighting (in-place modification).
        
        Args:
            results: List of result dictionaries
            score_key: Key for raw scores
            normalized_key: Key to store normalized scores
        """
        if not results:
            return

        scores = [ScoreFusion._as_float(r.get(score_key)) for r in results]
        
        if len(scores) < 2:
            for r in results:
                r[normalized_key] = ScoreFusion._as_float(r.get(score_key))
            return

        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            for r in results:
                r[normalized_key] = ScoreFusion._as_float(r.get(score_key))
            return

        for r in results:
            r[normalized_key] = (ScoreFusion._as_float(r.get(score_key)) - mean) / std_dev

    @staticmethod
    def min_max_normalize_scores(
        results: List[Dict[str, Any]],
        score_key: str,
        normalized_key: str,
    ) -> None:
        """
        Convert raw scores to min-max normalized scores (0-1 range) for stable weighting (in-place modification).
        
        Args:
            results: List of result dictionaries
            score_key: Key for raw scores
            normalized_key: Key to store normalized scores
        """
        if not results:
            return

        scores = [ScoreFusion._as_float(r.get(score_key)) for r in results]
        
        if len(scores) < 2:
            for r in results:
                r[normalized_key] = ScoreFusion._as_float(r.get(score_key))
            return

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            for r in results:
                r[normalized_key] = 0.5  # All scores are the same, set to middle value
            return

        for r in results:
            raw_score = ScoreFusion._as_float(r.get(score_key))
            r[normalized_key] = (raw_score - min_score) / score_range

    @staticmethod
    def deduplicate_results(
        results: List[Dict[str, Any]],
        score_key: str,
    ) -> List[Dict[str, Any]]:
        """
        Keep the highest-scoring entry per chunk_id.
        
        Args:
            results: List of result dictionaries
            score_key: Key for score comparison
            
        Returns:
            Deduplicated list of results
        """
        best_by_chunk: Dict[str, Dict[str, Any]] = {}
        
        for result in results:
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue
                
            current_best = best_by_chunk.get(chunk_id)
            result_score = ScoreFusion._as_float(result.get(score_key))
            current_score = (
                ScoreFusion._as_float(current_best.get(score_key))
                if current_best
                else None
            )
            
            if current_best is None or result_score > current_score:
                best_by_chunk[chunk_id] = result
                
        return list(best_by_chunk.values())

    @staticmethod
    def merge_vector_and_bm25(
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        *,
        top_k: int,
        vector_weight: float,
        bm25_weight: float,
    ) -> List[Dict[str, Any]]:
        """
        Merge FAISS and BM25 results using weighted normalized scores.
        
        Args:
            vector_results: Vector search results
            bm25_results: BM25 search results
            top_k: Number of final results
            vector_weight: Weight for vector scores
            bm25_weight: Weight for BM25 scores
            
        Returns:
            Merged and sorted list of results
        """
        combined: Dict[str, Dict[str, Any]] = {}

        # Group results by chunk_id
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

        # Compute weighted scores
        for chunk_id, data in combined.items():
            vec = data.get("vector")
            bm25 = data.get("bm25")

            vector_norm = vec.get("vector_normalized_score") if vec else None
            bm25_norm = bm25.get("bm25_normalized_score") if bm25 else None
            
            if bm25_norm is None and bm25:
                bm25_norm = ScoreFusion._as_float(bm25.get("bm25_raw_score"))

            weighted_vector = (
                ScoreFusion._as_float(vector_norm) * vector_weight
                if vector_norm is not None
                else 0.0
            )
            weighted_bm25 = (
                ScoreFusion._as_float(bm25_norm) * bm25_weight
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

            # Extract metadata
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

            # Derive file name from path if needed
            derived_file_name = file_name
            if not derived_file_name:
                if isinstance(source_path, Path):
                    derived_file_name = source_path.name
                elif isinstance(source_path, str):
                    derived_file_name = Path(source_path).name

            merged_results.append({
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
                "distance": ScoreFusion._as_float(vec.get("distance")) if vec else 0.0,
                "bm25_raw_score": bm25.get("bm25_raw_score") if bm25 else None,
                "keywords": bm25.get("keywords") if bm25 else vec.get("keywords", []),
                "provenance": (vec or {}).get("provenance") or (bm25 or {}).get("metadata"),
                "retrieval_mode": "hybrid",
            })

        # Sort by final score
        merged_results.sort(
            key=lambda item: item.get("similarity_score", 0.0), reverse=True
        )
        
        # Apply min-max normalization to final similarity scores for consistent 0-1 range
        ScoreFusion.min_max_normalize_scores(
            merged_results, "similarity_score", "normalized_similarity_score"
        )
        
        # Update similarity_score to be the normalized version
        for r in merged_results:
            r["similarity_score"] = r["normalized_similarity_score"]
        
        return merged_results[:top_k]

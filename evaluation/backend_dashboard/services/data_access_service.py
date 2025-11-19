"""
Data access service for ground truth operations.
"""

from typing import Dict, List, Any, Optional


class DataAccessService:
    """Handles data access operations like ground truth management."""

    def __init__(self, db):
        self.db = db

    def insert_ground_truth_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Insert multiple ground-truth rows into DB."""
        return self.db.insert_ground_truth_rows(rows)

    def get_ground_truth_list(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return list of ground-truth QA pairs."""
        return self.db.get_ground_truth_list(limit)

    def evaluate_semantic_similarity_with_source(self, retrieved_sources: List[Dict[str, Any]],
                                                ground_truth_id: int, embedder) -> Dict[str, Any]:
        """
        Evaluate semantic similarity between retrieved content and true source.

        Args:
            retrieved_sources: List of retrieved source dictionaries from RAG
            ground_truth_id: ID of ground truth entry to get true source
            embedder: Embedder instance to use for similarity calculation

        Returns:
            Dict with semantic similarity scores and analysis
        """
        try:
            # Get ground truth entry with true source
            gt_rows = self.db.get_ground_truth_list(limit=10000)
            true_source = None

            for row in gt_rows:
                if row.get('id') == ground_truth_id:
                    true_source = row.get('source', '').strip()
                    break

            if not true_source:
                return {
                    'error': f'No source found for ground truth ID {ground_truth_id}',
                    'semantic_similarity': 0.0,
                    'best_match_score': 0.0,
                    'matched_chunks': []
                }

            # Get embeddings for true source
            true_source_embedding = embedder.embed(true_source)

            # Calculate semantic similarity for each retrieved chunk
            similarities = []
            matched_chunks = []

            for i, source in enumerate(retrieved_sources):
                chunk_text = source.get('text', source.get('content', source.get('snippet', '')))
                if chunk_text.strip():
                    try:
                        chunk_embedding = embedder.embed(chunk_text)

                        # Calculate cosine similarity
                        import numpy as np
                        similarity = np.dot(true_source_embedding, chunk_embedding) / (
                            np.linalg.norm(true_source_embedding) * np.linalg.norm(chunk_embedding)
                        )

                        similarities.append(float(similarity))

                        # Store match info if similarity > 0.5
                        if similarity > 0.5:
                            matched_chunks.append({
                                'chunk_index': i,
                                'similarity_score': round(float(similarity), 4),
                                'chunk_text': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text,
                                'file_name': source.get('file_name', ''),
                                'page_number': source.get('page_number', 0)
                            })

                    except Exception:
                        similarities.append(0.0)

            # Calculate overall metrics
            if similarities:
                best_match_score = max(similarities)
                avg_similarity = sum(similarities) / len(similarities)

                # Sort matched chunks by similarity
                matched_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)

                return {
                    'semantic_similarity': round(avg_similarity, 4),
                    'best_match_score': round(best_match_score, 4),
                    'matched_chunks': matched_chunks[:5],  # Top 5 matches
                    'total_chunks_evaluated': len(similarities),
                    'chunks_above_threshold': len([s for s in similarities if s > 0.5]),
                    'true_source_length': len(true_source),
                    'embedder_used': getattr(embedder, 'profile', {}).get('model_id', 'unknown')
                }
            else:
                return {
                    'semantic_similarity': 0.0,
                    'best_match_score': 0.0,
                    'matched_chunks': [],
                    'total_chunks_evaluated': 0,
                    'chunks_above_threshold': 0,
                    'true_source_length': len(true_source),
                    'embedder_used': getattr(embedder, 'profile', {}).get('model_id', 'unknown')
                }

        except Exception as e:
            return {
                'error': str(e),
                'semantic_similarity': 0.0,
                'best_match_score': 0.0,
                'matched_chunks': []
            }
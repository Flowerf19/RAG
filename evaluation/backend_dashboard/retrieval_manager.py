"""
Retrieval Manager for Backend Dashboard
Provides cached retrieval functionality for evaluation.
"""

from typing import Dict, Any
from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval


def get_or_fetch_retrieval(cache: Dict[str, Any],
                          question: str,
                          embedder_type: str,
                          reranker_type: str,
                          use_query_enhancement: bool,
                          top_k: int) -> Dict[str, Any]:
    """
    Get cached retrieval result or fetch new one.

    Args:
        cache: Cache dictionary to store results
        question: Query question
        embedder_type: Type of embedder
        reranker_type: Type of reranker
        use_query_enhancement: Whether to use QEM
        top_k: Number of results to return

    Returns:
        Retrieval result dictionary
    """
    # Create cache key
    cache_key = f"{question}_{embedder_type}_{reranker_type}_{use_query_enhancement}_{top_k}"

    # Check cache first
    if cache_key in cache:
        return cache[cache_key]

    try:
        # Use the retrieval orchestrator function
        result = fetch_retrieval(
            query_text=question,
            top_k=top_k,
            embedder_type=embedder_type,
            reranker_type=reranker_type,
            use_query_enhancement=use_query_enhancement,
            evaluate_response=False
        )

        # Format result to match expected structure
        formatted_result = {
            'context': result.get('context', ''),
            'sources': result.get('sources', []),
            'chunks': result.get('chunks', []),
            'query': question
        }

        # Cache the result
        cache[cache_key] = formatted_result

        return formatted_result

    except Exception as e:
        # Return error result
        error_result = {
            'context': '',
            'sources': [],
            'chunks': [],
            'query': question,
            'error': str(e)
        }
        cache[cache_key] = error_result
        return error_result
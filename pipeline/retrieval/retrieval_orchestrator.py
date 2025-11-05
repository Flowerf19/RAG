"""
Retrieval Orchestrator Module
==============================
Orchestrates complete retrieval flow: QEM → Embedding → Hybrid Search → Reranking → Results.
Single Responsibility: High-level retrieval workflow coordination.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from embedders.embedder_type import EmbedderType
from pipeline.rag_pipeline import RAGPipeline
from pipeline.retrieval.retrieval_service import RAGRetrievalService
from query_enhancement.query_processor import create_query_processor

logger = logging.getLogger(__name__)


# Global cache for pipeline instances to avoid reloading models
_PIPELINE_CACHE: Dict[str, RAGPipeline] = {}


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

    Flow: Query Enhancement → Embedding Retrieval → Reranking (optional) → Final Results

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
        logger.info(f"📥 Received query: {query_text[:100]}...")

        # Setup embedder
        embedder_enum, use_api = _parse_embedder_type(embedder_type)

        # Initialize pipeline with caching
        cache_key = f"{embedder_enum.value}_{use_api}"
        if cache_key not in _PIPELINE_CACHE:
            logger.info(
                f"🔄 Creating new pipeline instance for {cache_key} (first time may take 30-60s)"
            )
            _PIPELINE_CACHE[cache_key] = RAGPipeline(
                embedder_type=embedder_enum, hf_use_api=use_api
            )
            logger.info(f"✅ Pipeline cached for {cache_key}")
        else:
            logger.info(f"♻️ Using cached pipeline for {cache_key}")

        pipeline = _PIPELINE_CACHE[cache_key]
        retriever = RAGRetrievalService(pipeline)

        # Query Enhancement
        query_processor = create_query_processor(use_query_enhancement, pipeline.embedder)
        expanded_queries = query_processor.enhance_query(query_text, use_query_enhancement)

        # Create fused embedding
        logger.info(f"🔍 Embedding {len(expanded_queries)} queries...")
        fused_embedding = query_processor.fuse_query_embeddings(expanded_queries)
        bm25_query = " ".join(expanded_queries).strip()
        logger.info("✅ Query embeddings created")

        # Hybrid retrieval - get more results for potential reranking
        retrieval_top_k = top_k * 5 if reranker_type != "none" else top_k
        logger.info(
            f"📚 Searching documents (retrieving top {retrieval_top_k} candidates)..."
        )
        
        results = retriever.retrieve_hybrid(
            query_text=query_text,
            top_k=retrieval_top_k,
            query_embedding=fused_embedding,
            bm25_query=bm25_query,
        )

        if not results:
            logger.warning("No hybrid retrieval results found.")
            return _build_empty_response(
                expanded_queries, embedder_type, reranker_type, use_query_enhancement
            )

        initial_count = len(results)
        logger.info(f"Retrieved {initial_count} results from hybrid search")

        # Apply reranking if specified
        reranked = False
        if reranker_type and reranker_type != "none":
            results, reranked = _apply_reranking(
                results, query_text, reranker_type, top_k, initial_count, api_tokens
            )
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


def _parse_embedder_type(embedder_type: str) -> tuple[EmbedderType, Optional[bool]]:
    """
    Parse embedder type string to enum and API flag.
    
    Args:
        embedder_type: Embedder type string
        
    Returns:
        Tuple of (EmbedderType enum, use_api flag)
    """
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

    return embedder_enum, use_api


def _build_empty_response(
    queries: List[str],
    embedder_type: str,
    reranker_type: str,
    query_enhanced: bool,
) -> Dict[str, Any]:
    """
    Build empty response when no results found.
    
    Args:
        queries: Expanded query list
        embedder_type: Embedder type string
        reranker_type: Reranker type string
        query_enhanced: Whether QEM was used
        
    Returns:
        Empty response dictionary
    """
    return {
        "context": "",
        "sources": [],
        "queries": queries,
        "retrieval_info": {
            "total_retrieved": 0,
            "reranked": False,
            "embedder": embedder_type,
            "reranker": reranker_type,
            "query_enhanced": query_enhanced,
        },
    }


def _apply_reranking(
    results: List[Dict[str, Any]],
    query_text: str,
    reranker_type: str,
    top_k: int,
    initial_count: int,
    api_tokens: Optional[Dict[str, str]],
) -> tuple[List[Dict[str, Any]], bool]:
    """
    Apply reranking to results if reranker is specified.
    
    Args:
        results: Initial retrieval results
        query_text: Query string
        reranker_type: Reranker type string
        top_k: Number of final results
        initial_count: Initial result count
        api_tokens: API tokens for rerankers
        
    Returns:
        Tuple of (reranked results, reranked flag)
    """
    try:
        from reranking.reranker_factory import RerankerFactory
        from reranking.reranker_type import RerankerType

        # Map reranker_type string to enum
        reranker_enum = _parse_reranker_type(reranker_type)

        if reranker_enum:
            # Get API token for API-based rerankers
            api_token = _get_api_token(reranker_type, api_tokens)

            reranker = RerankerFactory.create(reranker_enum, api_token=api_token)
            doc_texts = [r.get("text", "") for r in results]

            # Rerank and get top_k results
            reranked_results = reranker.rerank(query_text, doc_texts, top_k=top_k)

            # Reorder results based on reranking
            reranked_indices = [rr.index for rr in reranked_results]
            results = [results[i] for i in reranked_indices]

            # Add rerank_score
            for i, rr in enumerate(reranked_results):
                results[i]["rerank_score"] = rr.score
                logger.debug(
                    f"Result {i}: hybrid={results[i].get('similarity_score'):.4f}, rerank={rr.score:.4f}"
                )

            logger.info(
                f"Applied {reranker_type} reranking: {initial_count} -> {len(results)} results"
            )
            return results, True

    except Exception as e:
        logger.warning(
            f"Reranking failed ({reranker_type}): {e}. Using top {top_k} from original results."
        )
        return results[:top_k], False

    return results[:top_k], False


def _parse_reranker_type(reranker_type: str):
    """
    Parse reranker type string to enum.
    
    Args:
        reranker_type: Reranker type string
        
    Returns:
        RerankerType enum or None
    """
    from reranking.reranker_type import RerankerType

    if reranker_type == "bge_m3_ollama":
        return RerankerType.BGE_M3_OLLAMA
    elif reranker_type == "bge_m3_hf_api":
        return RerankerType.BGE_M3_HF_API
    elif reranker_type == "bge_m3_hf_local":
        return RerankerType.BGE_M3_HF_LOCAL
    elif reranker_type == "cohere":
        return RerankerType.COHERE
    elif reranker_type == "jina":
        return RerankerType.JINA
    return None


def _get_api_token(reranker_type: str, api_tokens: Optional[Dict[str, str]]) -> Optional[str]:
    """
    Get API token for reranker if needed.
    
    Args:
        reranker_type: Reranker type string
        api_tokens: API tokens dictionary
        
    Returns:
        API token or None
    """
    if not api_tokens:
        return None

    if reranker_type == "bge_m3_hf_api":
        return api_tokens.get("hf")
    elif reranker_type == "cohere":
        return api_tokens.get("cohere")
    elif reranker_type == "jina":
        return api_tokens.get("jina")
        
    return None

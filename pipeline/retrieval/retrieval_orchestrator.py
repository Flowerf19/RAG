"""
Retrieval Orchestrator Module
==============================
Orchestrates complete retrieval flow: QEM → Embedding → Hybrid Search → Reranking → Results.
Single Responsibility: High-level retrieval workflow coordination.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
import threading

from embedders.embedder_type import EmbedderType
from pipeline.rag_pipeline import RAGPipeline
from pipeline.retrieval.retrieval_service import RAGRetrievalService
from query_enhancement.query_processor import create_query_processor

# Import evaluation components
try:
    from evaluation.metrics.logger import EvaluationLogger
    from evaluation.evaluators.auto_evaluator import AutoEvaluator  # noqa: F401
    from evaluation.metrics.token_counter import token_counter
    _EVALUATION_AVAILABLE = True
except ImportError:
    _EVALUATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Evaluation system not available, running without metrics logging")

logger = logging.getLogger(__name__)


# Global cache for pipeline instances to avoid reloading models
_PIPELINE_CACHE: Dict[str, RAGPipeline] = {}

# Lock to prevent concurrent pipeline initializations (double-checked locking)
_PIPELINE_LOCK = threading.Lock()


def fetch_retrieval(
    query_text: str,
    top_k: int = 10,
    max_chars: int = 8000,
    embedder_type: str = "ollama",
    reranker_type: str = "none",
    use_query_enhancement: bool = True,
    api_tokens: Optional[Dict[str, str]] = None,
    llm_model: Optional[str] = None,
    evaluate_response: bool = False,  # Disabled by default - evaluation should happen in dedicated methods
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
        llm_model: LLM model name for evaluation logging (e.g., "gemini", "lmstudio")
        evaluate_response: DISABLED - Response evaluation should happen in dedicated evaluation methods

    Returns:
        Dict with keys: "context" (str), "sources" (list), "queries" (list), "retrieval_info" (dict)
    """
    start_time = time.time() if _EVALUATION_AVAILABLE else None

    try:
        logger.info(f"[RECEIVED] Received query: {query_text[:100]}...")

        # Setup embedder
        embedder_enum, use_api = _parse_embedder_type(embedder_type)

        # Initialize pipeline with caching (thread-safe double-checked locking)
        cache_key = f"{embedder_enum.value}_{use_api}_{embedder_type}"

        if cache_key not in _PIPELINE_CACHE:
            # Acquire lock so only one thread/process in this interpreter creates the pipeline
            with _PIPELINE_LOCK:
                if cache_key not in _PIPELINE_CACHE:
                    logger.info(
                        f"[CREATING] Creating new pipeline instance for {cache_key} (first time may take 30-60s)"
                    )
                    _PIPELINE_CACHE[cache_key] = RAGPipeline(
                        embedder_type=embedder_enum, hf_use_api=use_api
                    )
                    logger.info(f"[CACHED] Pipeline cached for {cache_key}")
        else:
            logger.info(f"[USING] Using cached pipeline for {cache_key}")

        pipeline = _PIPELINE_CACHE[cache_key]

        pipeline = _PIPELINE_CACHE[cache_key]
        
        # Override embedder for specific multilingual models
        if embedder_type.lower() in ["e5_large_instruct", "e5_base", "gte_multilingual_base", 
                                   "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"]:
            from embedders.embedder_factory import EmbedderFactory
            factory = EmbedderFactory()
            
            if embedder_type.lower() == "e5_large_instruct":
                pipeline.embedder = factory.create_e5_large_instruct(device="cpu")
            elif embedder_type.lower() == "e5_base":
                pipeline.embedder = factory.create_e5_base(device="cpu")
            elif embedder_type.lower() == "gte_multilingual_base":
                pipeline.embedder = factory.create_gte_multilingual_base(device="cpu")
            elif embedder_type.lower() == "paraphrase_mpnet_base_v2":
                pipeline.embedder = factory.create_paraphrase_mpnet_base_v2(device="cpu")
            elif embedder_type.lower() == "paraphrase_minilm_l12_v2":
                pipeline.embedder = factory.create_paraphrase_minilm_l12_v2(device="cpu")
            
            logger.info(f"[SWITCHED] Switched to {embedder_type} embedder")

        retriever = RAGRetrievalService(pipeline)

        # Query Enhancement
        query_processor = create_query_processor(use_query_enhancement, pipeline.embedder)
        expanded_queries = query_processor.enhance_query(query_text, use_query_enhancement)
        logger.info(f"Expanded queries: {expanded_queries}")

        # Create fused embedding
        logger.info(f"[EMBEDDING] Embedding {len(expanded_queries)} queries...")
        fused_embedding = query_processor.fuse_query_embeddings(expanded_queries)
        bm25_query = " ".join(expanded_queries).strip()
        logger.info("[EMBEDDING] Query embeddings created")

        # Hybrid retrieval - get more results for potential reranking
        retrieval_top_k = top_k * 5 if reranker_type != "none" else top_k
        logger.info(
            f"📚 Searching documents (retrieving top {retrieval_top_k} candidates)..."
        )

        # Determine number of index pairs that will be searched for a helpful
        # consolidated log message. `RAGRetrievalService.get_all_index_pairs`
        # lists available FAISS indexes.
        index_pairs = retriever.get_all_index_pairs()
        num_indexes = len(index_pairs)

        results = retriever.retrieve_hybrid(
            query_text=query_text,
            top_k=retrieval_top_k,
            query_embedding=fused_embedding,
            bm25_query=bm25_query,
        )

        if not results:
            logger.warning("No hybrid retrieval results found.")
            empty_response = _build_empty_response(
                expanded_queries, embedder_type, reranker_type, use_query_enhancement
            )

            # Log failed evaluation if available
            if _EVALUATION_AVAILABLE:
                latency = time.time() - start_time if start_time else 0
                eval_logger = EvaluationLogger()
                
                # Get specific model names
                embedder_specific = getattr(pipeline.embedder, 'get_model_name', lambda: None)() if pipeline.embedder else None
                
                eval_logger.log_evaluation(
                    query=query_text,
                    model=llm_model or f"{embedder_type}_{reranker_type}",
                    latency=latency,
                    error=True,
                    error_message="No retrieval results found",
                    embedder_model=embedder_type,
                    llm_model=llm_model,
                    reranker_model=reranker_type,
                    embedder_specific_model=embedder_specific,
                    llm_specific_model=llm_model,  # LLM model name is already specific
                    reranker_specific_model=None,  # No reranker used
                        query_enhanced=use_query_enhancement,
                        embedding_tokens=0,
                        reranking_tokens=0,
                        llm_tokens=0,
                        total_tokens=0,
                        retrieval_chunks=0,
                        metadata={'evaluation_type': 'retrieval'}
                )

            return empty_response

        initial_count = len(results)
        # Consolidated info-level log: number of indexes searched and total
        # merged results. This replaces multiple per-index INFO logs.
        logger.info(
            "Hybrid retrieval completed: searched %d index(es), merged %d result(s)",
            num_indexes,
            initial_count,
        )

        # Apply reranking if specified
        reranked = False
        reranker_obj = None
        if reranker_type and reranker_type != "none":
            results, reranked, reranker_obj = _apply_reranking(
                results, query_text, reranker_type, top_k, initial_count, api_tokens
            )
        else:
            results = results[:top_k]

        # Build final output
        context = retriever.build_context(results, max_chars=max_chars)
        sources = retriever.to_ui_items(results)

        # Calculate evaluation metrics if available
        if _EVALUATION_AVAILABLE:
            latency = time.time() - start_time if start_time else 0

            # Track tokens used in different operations
            embedding_tokens = 0
            reranking_tokens = 0
            llm_tokens = 0

            try:
                # Count embedding tokens (query + expanded queries)
                embedding_texts = [query_text] + expanded_queries
                embedding_tokens = sum(token_counter.count_tokens(text, embedder_type) for text in embedding_texts)
                logger.info(f"Embedding tokens counted: {embedding_tokens}")

                # Count reranking tokens if reranking was used
                if reranked:
                    # Approximate reranking tokens (query + top candidates)
                    reranking_texts = [query_text] + [result.get('text', '')[:500] for result in results[:top_k]]
                    reranking_tokens = sum(token_counter.count_tokens(text, 'default') for text in reranking_texts)
                    logger.info(f"Reranking tokens counted: {reranking_tokens}")

                # Count LLM tokens for evaluation (if LLM evaluation is used)
                if llm_model:
                    eval_texts = [query_text, context, context]  # query, answer, context for evaluation
                    llm_tokens = sum(token_counter.count_tokens(text, llm_model) for text in eval_texts)
                    logger.info(f"LLM tokens counted: {llm_tokens}")

            except Exception as e:
                logger.error(f"Error counting tokens: {e}")
                # Continue with zero token counts

            # No joined_sources needed at retrieval-level; skip assigning unused variable

            # NOTE: Response quality evaluation should be done in dashboard modules where
            # actual LLM responses and ground truth are available. The current evaluation
            # incorrectly uses retrieval context as "answer" and compares it against itself.
            # Faithfulness/relevance scores are inflated, and recall is always None.
            # DISABLED: Evaluation should only happen in dedicated evaluation methods
            # if evaluate_response:
            #     evaluator = AutoEvaluator(embedder=pipeline.embedder)
            #     # We don't use the results since they're set to None in logging anyway
            #     evaluator.evaluate_response(query_text, context, joined_sources)

            # Log evaluation with token counts
            eval_logger = EvaluationLogger()
            
            # Get specific model names
            embedder_specific = getattr(pipeline.embedder, 'get_model_name', lambda: None)() if pipeline.embedder else None
            reranker_specific = getattr(reranker_obj, 'get_model_name', lambda: None)() if reranker_obj else None
            
            eval_logger.log_evaluation(
                query=query_text,
                model=llm_model or f"{embedder_type}_{reranker_type}",
                latency=latency,
                faithfulness=None,  # Not evaluated at retrieval level - use dedicated faithfulness evaluation
                relevance=None,     # Not evaluated at retrieval level - use dedicated relevance evaluation  
                recall=None,        # Not evaluated at retrieval level - use dedicated recall evaluation
                error=False,
                embedder_model=embedder_type,
                llm_model=llm_model,
                reranker_model=reranker_type,
                embedder_specific_model=embedder_specific,
                llm_specific_model=llm_model,  # LLM model name is already specific
                reranker_specific_model=reranker_specific,
                query_enhanced=use_query_enhancement,
                embedding_tokens=embedding_tokens,
                reranking_tokens=reranking_tokens,
                llm_tokens=llm_tokens,
                total_tokens=embedding_tokens + reranking_tokens + llm_tokens,
                retrieval_chunks=initial_count,
                metadata={'evaluation_type': 'retrieval'}
            )

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

        # Log failed evaluation if available
        if _EVALUATION_AVAILABLE:
            latency = time.time() - start_time if start_time else 0
            eval_logger = EvaluationLogger()
            eval_logger.log_evaluation(
                query=query_text,
                model=llm_model or f"{embedder_type}_{reranker_type}",
                latency=latency,
                error=True,
                error_message=str(exc),
                embedder_model=embedder_type,
                llm_model=llm_model,
                reranker_model=reranker_type,
                query_enhanced=use_query_enhancement
            )

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
    elif embedder_type.lower() in ["e5_large_instruct", "e5_base", "gte_multilingual_base", 
                                   "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"]:
        # New multilingual models - use HUGGINGFACE with specific model selection
        embedder_enum = EmbedderType.HUGGINGFACE
        use_api = False  # All new models are local
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
) -> tuple[List[Dict[str, Any]], bool, Optional[Any]]:
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
        Tuple of (reranked results, reranked flag, reranker object)
    """
    try:
        from reranking.reranker_factory import RerankerFactory

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
            return results, True, reranker

    except Exception as e:
        logger.warning(
            f"Reranking failed ({reranker_type}): {e}. Using top {top_k} from original results."
        )
        return results[:top_k], False, None

    return results[:top_k], False, None


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
    elif reranker_type == "jina_v2_multilingual":
        return RerankerType.JINA_V2_MULTILINGUAL
    elif reranker_type == "gte_multilingual":
        return RerankerType.GTE_MULTILINGUAL
    elif reranker_type == "bge_base":
        return RerankerType.BGE_BASE
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

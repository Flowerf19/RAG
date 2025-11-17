"""
Evaluation Logger
Logs metrics after each RAG pipeline execution for performance tracking and comparison.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .database import MetricsDB


logger = logging.getLogger(__name__)


class EvaluationLogger:
    """Logs evaluation metrics to database after RAG pipeline runs."""

    def __init__(self, db_path: str = "data/metrics.db"):
        """Initialize logger with database connection."""
        self.db = MetricsDB(db_path)

    def log_evaluation(self,
                      query: str,
                      model: str,
                      latency: float,
                      faithfulness: Optional[float] = None,
                      relevance: Optional[float] = None,
                      recall: Optional[float] = None,
                      error: bool = False,
                      error_message: Optional[str] = None,
                      embedder_model: Optional[str] = None,
                      llm_model: Optional[str] = None,
                      reranker_model: Optional[str] = None,
                      embedder_specific_model: Optional[str] = None,
                      llm_specific_model: Optional[str] = None,
                      reranker_specific_model: Optional[str] = None,
                      query_enhanced: bool = False,
                      embedding_tokens: int = 0,
                      reranking_tokens: int = 0,
                      llm_tokens: int = 0,
                      total_tokens: int = 0,
                      retrieval_chunks: int = 0,
                      metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Log a single evaluation metric.

        Args:
            query: The user query
            model: Model name/configuration used (legacy, kept for compatibility)
            latency: Response time in seconds
            faithfulness: Faithfulness score (0-1)
            relevance: Relevance score (0-1)
            recall: Recall score (0-1) for retrieval evaluation
            error: Whether the pipeline failed
            error_message: Error details if any
            embedder_model: Embedding model used (e.g., 'bge-m3', 'huggingface_local')
            llm_model: LLM model used (e.g., 'gemini', 'lmstudio')
            reranker_model: Reranker model used (e.g., 'bge_m3_hf_local', 'none')
            embedder_specific_model: Specific embedder model name (e.g., 'gemma-7b', 'bge-m3')
            llm_specific_model: Specific LLM model name (e.g., 'gemini-1.5-flash', 'llama-3.1-8b')
            reranker_specific_model: Specific reranker model name (e.g., 'bge-reranker-v2-m3', 'jina-reranker-v2-base-multilingual')
            query_enhanced: Whether query enhancement was used
            embedding_tokens: Number of tokens used for embedding operations
            reranking_tokens: Number of tokens used for reranking operations
            llm_tokens: Number of tokens used for LLM operations
            total_tokens: Total tokens used across all operations
            retrieval_chunks: Number of chunks retrieved from vector search
            metadata: Additional context data

        Returns:
            Database record ID
        """
        # Prepare metadata with specific model names
        final_metadata = metadata or {}
        if embedder_specific_model:
            final_metadata['embedder_specific_model'] = embedder_specific_model
        if llm_specific_model:
            final_metadata['llm_specific_model'] = llm_specific_model
        if reranker_specific_model:
            final_metadata['reranker_specific_model'] = reranker_specific_model
        
        metric = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'model': model,
            'embedder_model': embedder_model,
            'llm_model': llm_model,
            'reranker_model': reranker_model,
            'query_enhanced': query_enhanced,
            'latency': latency,
            'faithfulness': faithfulness,
            'relevance': relevance,
            'recall': recall,
            'error': error,
            'error_message': error_message,
            'embedding_tokens': embedding_tokens,
            'reranking_tokens': reranking_tokens,
            'llm_tokens': llm_tokens,
            'total_tokens': total_tokens,
            'retrieval_chunks': retrieval_chunks,
            'metadata': final_metadata
        }

        try:
            # Debug: print metric dict
            print(f"DEBUG: Metric dict keys: {list(metric.keys())}")
            for k, v in metric.items():
                print(f"  {k}: {type(v)} = {v}")
            record_id = self.db.insert_metric(metric_dict=metric)
            faithfulness_str = f"{faithfulness:.3f}" if isinstance(faithfulness, (int, float)) else str(faithfulness)
            relevance_str = f"{relevance:.3f}" if isinstance(relevance, (int, float)) else str(relevance)
            recall_str = f"{recall:.3f}" if isinstance(recall, (int, float)) else str(recall)
            logger.info(f"Logged metric for model {model}: latency={latency:.3f}s, "
                       f"faithfulness={faithfulness_str}, "
                       f"relevance={relevance_str}, "
                       f"recall={recall_str}")
            return record_id
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")
            raise

    def time_and_log(self, query: str, model: str):
        """
        Context manager to time a RAG pipeline execution and log results.

        Usage:
            with logger.time_and_log(query, model) as timer:
                # Run RAG pipeline
                result = rag_pipeline.run(query)
                timer.set_scores(faithfulness=0.9, relevance=0.8)
        """
        return _PipelineTimer(self, query, model)


class _PipelineTimer:
    """Context manager for timing and logging pipeline execution."""

    def __init__(self, logger: EvaluationLogger, query: str, model: str):
        self.logger = logger
        self.query = query
        self.model = model
        self.start_time = None
        self.scores = {}
        self.tokens = {
            'embedding': 0,
            'reranking': 0,
            'llm': 0,
            'total': 0
        }
        self.model_config = {
            'embedder_model': None,
            'llm_model': None,
            'reranker_model': None,
            'embedder_specific_model': None,
            'llm_specific_model': None,
            'reranker_specific_model': None,
            'query_enhanced': False,
            'retrieval_chunks': 0
        }

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time

        if exc_type is not None:
            # Pipeline failed
            self.logger.log_evaluation(
                query=self.query,
                model=self.model,
                latency=latency,
                error=True,
                error_message=str(exc_val),
                embedder_model=self.model_config['embedder_model'],
                llm_model=self.model_config['llm_model'],
                reranker_model=self.model_config['reranker_model'],
                embedder_specific_model=self.model_config['embedder_specific_model'],
                llm_specific_model=self.model_config['llm_specific_model'],
                reranker_specific_model=self.model_config['reranker_specific_model'],
                query_enhanced=self.model_config['query_enhanced'],
                retrieval_chunks=self.model_config['retrieval_chunks']
            )
        else:
            # Pipeline succeeded
            self.logger.log_evaluation(
                query=self.query,
                model=self.model,
                latency=latency,
                faithfulness=self.scores.get('faithfulness'),
                relevance=self.scores.get('relevance'),
                recall=self.scores.get('recall'),
                embedder_model=self.model_config['embedder_model'],
                llm_model=self.model_config['llm_model'],
                reranker_model=self.model_config['reranker_model'],
                embedder_specific_model=self.model_config['embedder_specific_model'],
                llm_specific_model=self.model_config['llm_specific_model'],
                reranker_specific_model=self.model_config['reranker_specific_model'],
                query_enhanced=self.model_config['query_enhanced'],
                embedding_tokens=self.tokens['embedding'],
                reranking_tokens=self.tokens['reranking'],
                llm_tokens=self.tokens['llm'],
                total_tokens=self.tokens['total'],
                retrieval_chunks=self.model_config['retrieval_chunks']
            )

    def set_scores(self, faithfulness: float = None, relevance: float = None, recall: float = None):
        """Set evaluation scores for this pipeline run."""
        if faithfulness is not None:
            self.scores['faithfulness'] = faithfulness
        if relevance is not None:
            self.scores['relevance'] = relevance
        if recall is not None:
            self.scores['recall'] = recall

    def add_embedding_tokens(self, tokens: int):
        """Add tokens used for embedding operations."""
        self.tokens['embedding'] += tokens
        self._update_total_tokens()

    def add_reranking_tokens(self, tokens: int):
        """Add tokens used for reranking operations."""
        self.tokens['reranking'] += tokens
        self._update_total_tokens()

    def add_llm_tokens(self, tokens: int):
        """Add tokens used for LLM operations."""
        self.tokens['llm'] += tokens
        self._update_total_tokens()

    def set_model_config(self,
                        embedder_model: str = None,
                        llm_model: str = None,
                        reranker_model: str = None,
                        embedder_specific_model: str = None,
                        llm_specific_model: str = None,
                        reranker_specific_model: str = None,
                        query_enhanced: bool = False,
                        retrieval_chunks: int = 0):
        """Set model configuration for this pipeline run."""
        self.model_config.update({
            'embedder_model': embedder_model,
            'llm_model': llm_model,
            'reranker_model': reranker_model,
            'embedder_specific_model': embedder_specific_model,
            'llm_specific_model': llm_specific_model,
            'reranker_specific_model': reranker_specific_model,
            'query_enhanced': query_enhanced,
            'retrieval_chunks': retrieval_chunks
        })
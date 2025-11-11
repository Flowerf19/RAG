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
                      error: bool = False,
                      error_message: Optional[str] = None,
                      embedder_model: Optional[str] = None,
                      llm_model: Optional[str] = None,
                      reranker_model: Optional[str] = None,
                      query_enhanced: bool = False,
                      metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Log a single evaluation metric.

        Args:
            query: The user query
            model: Model name/configuration used (legacy, kept for compatibility)
            latency: Response time in seconds
            faithfulness: Faithfulness score (0-1)
            relevance: Relevance score (0-1)
            error: Whether the pipeline failed
            error_message: Error details if any
            embedder_model: Embedding model used (e.g., 'bge-m3', 'huggingface_local')
            llm_model: LLM model used (e.g., 'gemini', 'lmstudio')
            reranker_model: Reranker model used (e.g., 'bge_m3_hf_local', 'none')
            query_enhanced: Whether query enhancement was used
            metadata: Additional context data

        Returns:
            Database record ID
        """
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
            'error': error,
            'error_message': error_message,
            'metadata': metadata or {}
        }

        try:
            record_id = self.db.insert_metric(metric)
            faithfulness_str = f"{faithfulness:.3f}" if isinstance(faithfulness, (int, float)) else str(faithfulness)
            relevance_str = f"{relevance:.3f}" if isinstance(relevance, (int, float)) else str(relevance)
            logger.info(f"Logged metric for model {model}: latency={latency:.3f}s, "
                       f"faithfulness={faithfulness_str}, "
                       f"relevance={relevance_str}")
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
                error_message=str(exc_val)
            )
        else:
            # Pipeline succeeded
            self.logger.log_evaluation(
                query=self.query,
                model=self.model,
                latency=latency,
                faithfulness=self.scores.get('faithfulness'),
                relevance=self.scores.get('relevance')
            )

    def set_scores(self, faithfulness: float = None, relevance: float = None):
        """Set evaluation scores for this pipeline run."""
        self.scores['faithfulness'] = faithfulness
        self.scores['relevance'] = relevance
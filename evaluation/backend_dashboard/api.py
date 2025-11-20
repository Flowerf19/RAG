"""
Backend Dashboard API
Provides data access for the evaluation dashboard from different system stages.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging
import sqlite3

from evaluation.metrics.database import MetricsDB


class BackendDashboard:
    """Backend API for evaluation dashboard data access."""

    def __init__(self, db_path: str = "data/metrics.db"):
        """Initialize with database connection."""
        self.db = MetricsDB(db_path)
        # Cache for embedder instances to avoid recreating them
        self._embedder_cache = {}
        # Cache for retrieval results to avoid redundant fetches
        self._retrieval_cache = {}

    def _get_or_create_embedder(self, embedder_type: str):
        """
        Get cached embedder instance or create new one if not exists.
        
        Args:
            embedder_type: Type of embedder ('ollama', 'huggingface_local', etc.)
            
        Returns:
            Cached embedder instance
        """
        cache_key = embedder_type.lower()
        
        if cache_key in self._embedder_cache:
            return self._embedder_cache[cache_key]
        
        # Create new embedder instance
        from embedders.embedder_factory import EmbedderFactory
        from embedders.embedder_type import EmbedderType
        
        factory = EmbedderFactory()
        
        # Parse embedder type
        if embedder_type.lower() in ["e5_large_instruct", "e5_base", "gte_multilingual_base", 
                                   "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"]:
            embedder_enum = EmbedderType.HUGGINGFACE
            use_api = False
        elif embedder_type == "huggingface_api":
            embedder_enum = EmbedderType.HUGGINGFACE
            use_api = True
        elif embedder_type == "huggingface_local":
            embedder_enum = EmbedderType.HUGGINGFACE
            use_api = False
        else:  # ollama and others
            embedder_enum = EmbedderType.OLLAMA
            use_api = False
        
        # Create embedder
        if embedder_enum == EmbedderType.HUGGINGFACE and not use_api:
            if embedder_type.lower() == "e5_large_instruct":
                embedder = factory.create_e5_large_instruct(device="cpu")
            elif embedder_type.lower() == "e5_base":
                embedder = factory.create_e5_base(device="cpu")
            elif embedder_type.lower() == "gte_multilingual_base":
                embedder = factory.create_gte_multilingual_base(device="cpu")
            elif embedder_type.lower() == "paraphrase_mpnet_base_v2":
                embedder = factory.create_paraphrase_mpnet_base_v2(device="cpu")
            elif embedder_type.lower() == "paraphrase_minilm_l12_v2":
                embedder = factory.create_paraphrase_minilm_l12_v2(device="cpu")
            else:
                # Default HuggingFace local embedder for "huggingface_local"
                embedder = factory.create_huggingface_local(device="cpu")
        else:
            from pipeline.rag_pipeline import RAGPipeline
            pipeline = RAGPipeline(embedder_type=embedder_enum, hf_use_api=use_api)
            embedder = pipeline.embedder
        
        # Cache the embedder
        self._embedder_cache[cache_key] = embedder
        return embedder

    def _get_or_fetch_retrieval(self, question: str, embedder_type: str, reranker_type: str, 
                               use_query_enhancement: bool, top_k: int) -> Dict[str, Any]:
        """
        Get cached retrieval result or fetch new one if not exists.
        
        Args:
            question: The query question
            embedder_type: Type of embedder
            reranker_type: Type of reranker
            use_query_enhancement: Whether to use QEM
            top_k: Number of top results to retrieve
            
        Returns:
            Cached or freshly fetched retrieval result
        """
        cache_key = (question, embedder_type, reranker_type, use_query_enhancement, top_k)
        
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]
        
        # Fetch new retrieval result
        from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval
        result = fetch_retrieval(
            query_text=question,
            embedder_type=embedder_type,
            reranker_type=reranker_type,
            use_query_enhancement=use_query_enhancement,
            top_k=top_k
        )
        
        # Cache the result
        self._retrieval_cache[cache_key] = result
        return result

    def _get_model_display_info(self, model_key: str, model_type: str) -> str:
        """Get display name for model, checking metadata and configs."""
        if not model_key or model_key == 'none':
            return 'None'

        # Try to get specific model info from recent metrics metadata
        try:
            recent_metrics = self.db.get_metrics(limit=10)
            for metric in recent_metrics:
                metadata = metric.get('metadata', '{}')
                if metadata and isinstance(metadata, str):
                    try:
                        meta_dict = json.loads(metadata)
                        if model_type == 'llm' and meta_dict.get('llm_model') == model_key:
                            specific_model = meta_dict.get('llm_specific_model')
                            if specific_model:
                                return f"{model_key.title()} ({specific_model})"
                        elif model_type == 'embedder' and meta_dict.get('embedder_model') == model_key:
                            specific_model = meta_dict.get('embedder_specific_model')
                            if specific_model:
                                return f"{model_key.title()} ({specific_model})"
                        elif model_type == 'reranker' and meta_dict.get('reranker_model') == model_key:
                            specific_model = meta_dict.get('reranker_specific_model')
                            if specific_model:
                                return f"{model_key.replace('_', ' ').title()} ({specific_model})"
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass

        # Fallback to formatted provider names
        return self._format_provider_name(model_key, model_type)

    def _format_provider_name(self, model_key: str, model_type: str) -> str:
        """Format provider name for better display."""
        if not model_key or model_key == 'none':
            return 'None'

        # Clean up the model key
        clean_key = model_key.strip().lower()

        # LLM models - use provider names with better formatting
        if model_type == 'llm':
            if clean_key == 'ollama':
                return 'Ollama'
            elif clean_key == 'gemini':
                return 'Google Gemini'
            elif clean_key == 'lmstudio':
                return 'LM Studio'
            elif clean_key == 'openai':
                return 'OpenAI GPT'
            else:
                return model_key.replace('_', ' ').title()

        # Embedding models - use provider names with better formatting
        elif model_type == 'embedder':
            if clean_key == 'ollama':
                return 'Ollama Embeddings'
            elif clean_key == 'huggingface_local':
                return 'HuggingFace Local'
            elif clean_key == 'huggingface_api':
                return 'HuggingFace API'
            else:
                return model_key.replace('_', ' ').title()

        # Reranking models - use descriptive names
        elif model_type == 'reranker':
            if clean_key == 'bge_m3_hf_local':
                return 'BGE-M3 (Local)'
            elif clean_key == 'bge_m3_ollama':
                return 'BGE-M3 (Ollama)'
            elif clean_key == 'bge_m3_hf_api':
                return 'BGE-M3 (API)'
            elif clean_key == 'jina_v2_multilingual':
                return 'Jina V2 Multilingual'
            elif clean_key == 'gte_multilingual':
                return 'GTE Multilingual'
            elif clean_key == 'bge_base':
                return 'BGE Base'
            elif clean_key == 'none':
                return 'No Reranking'
            else:
                return model_key.replace('_', ' ').title()

        # Query enhancement - use provider names
        elif model_type == 'qe':
            if clean_key in ['true', 'ollama', 'gemini', 'lmstudio']:
                return f'{model_key.title()} QE'
            else:
                return f'{model_key.replace("_", " ").title()} QE'

        return model_key

    def get_overview_stats(self) -> Dict[str, Any]:
        """Get overall system statistics for dashboard header."""
        llm_stats = self.db.get_llm_stats()

        if not llm_stats:
            return {
                'total_queries': 0,
                'avg_accuracy': 0.0,
                'avg_faithfulness': 0.0,
                'avg_relevance': 0.0,
                'avg_recall': 0.0,
                'avg_latency': 0.0,
                'error_rate': 0.0,
                'model_count': 0
            }

        # Calculate system-wide averages based on LLM models
        total_queries = sum(model['total_queries'] for model in llm_stats.values())
        weighted_accuracy = sum(model['avg_accuracy'] * model['total_queries'] for model in llm_stats.values())
        weighted_faithfulness = sum(model['avg_faithfulness'] * model['total_queries'] for model in llm_stats.values())
        weighted_relevance = sum(model['avg_relevance'] * model['total_queries'] for model in llm_stats.values())
        weighted_recall = sum(model['avg_recall'] * model['total_queries'] for model in llm_stats.values())
        weighted_latency = sum(model['avg_latency'] * model['total_queries'] for model in llm_stats.values())
        total_errors = sum(model['error_rate'] * model['total_queries'] / 100 for model in llm_stats.values())

        return {
            'total_queries': total_queries,
            'avg_accuracy': round(weighted_accuracy / total_queries, 3) if total_queries > 0 else 0.0,
            'avg_faithfulness': round(weighted_faithfulness / total_queries, 3) if total_queries > 0 else 0.0,
            'avg_relevance': round(weighted_relevance / total_queries, 3) if total_queries > 0 else 0.0,
            'avg_recall': round(weighted_recall / total_queries, 3) if total_queries > 0 else 0.0,
            'avg_latency': round(weighted_latency / total_queries, 3) if total_queries > 0 else 0.0,
            'error_rate': round(total_errors / total_queries * 100, 2) if total_queries > 0 else 0.0,
            'model_count': len(llm_stats)
        }

    def get_model_comparison_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get data for model comparison tables."""
        # Use unified model stats instead of component-specific stats
        # This ensures all evaluation types from the same configuration are aggregated
        model_stats = self.db.get_model_stats()

        # Convert to list format for display
        all_models = []

        for model_name, model_stats_data in model_stats.items():
            all_models.append({
                'model': model_name,  # Use the unified model name directly
                'queries': model_stats_data['total_queries'],
                'accuracy': model_stats_data['avg_accuracy'],
                'faithfulness': model_stats_data['avg_faithfulness'],
                'relevance': model_stats_data['avg_relevance'],
                'recall': model_stats_data['avg_recall'],
                'latency': model_stats_data['avg_latency'],
                'error_rate': model_stats_data['error_rate']
            })

        # For backward compatibility, return in the expected format
        # But now all models are in one combined list
        return {
            'reranking': [],  # Empty since we're using unified model names
            'embedding': [],  # Empty since we're using unified model names
            'llm': sorted(all_models, key=lambda x: x['accuracy'], reverse=True),  # Put all models here
            'query_enhancement': []  # Keep query enhancement separate
        }

    def get_latency_over_time(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get latency data for time series chart."""
        # Query actual latency data from database
        try:
            # Get metrics from the last N hours
            from datetime import datetime, timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            # Query metrics grouped by hour
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT
                        strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                        AVG(latency) as avg_latency,
                        COUNT(*) as query_count,
                        AVG(faithfulness) as avg_faithfulness,
                        AVG(relevance) as avg_relevance,
                        AVG(recall) as avg_recall
                    FROM metrics
                    WHERE timestamp >= ?
                    GROUP BY hour
                    ORDER BY hour
                """, (cutoff_time.isoformat(),))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        'timestamp': row['hour'],
                        'avg_latency': round(row['avg_latency'] or 0, 3),
                        'query_count': row['query_count'] or 0,
                        'avg_faithfulness': round(row['avg_faithfulness'] or 0, 3),
                        'avg_relevance': round(row['avg_relevance'] or 0, 3),
                        'avg_recall': round(row['avg_recall'] or 0, 3)
                    })

                return results

        except Exception:
            # Fallback to mock data if query fails
            return [
                {
                    'timestamp': (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                    'avg_latency': round(1.0 + 0.5 * (i % 3), 3),
                    'query_count': 10 + (i % 5),
                    'avg_faithfulness': 0.7,
                    'avg_relevance': 0.8,
                    'avg_recall': 0.6
                }
                for i in range(min(hours, 24))
            ]

    def get_accuracy_by_model(self) -> List[Dict[str, Any]]:
        """Get accuracy data for bar chart."""
        stats = self.db.get_model_stats()

        return [
            {
                'model': model_name,
                'accuracy': model_stats['avg_accuracy'],
                'faithfulness': model_stats['avg_faithfulness'],
                'relevance': model_stats['avg_relevance']
            }
            for model_name, model_stats in stats.items()
        ]

    def get_recent_metrics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent metric records for detailed view."""
        return self.db.get_metrics(limit=limit)

    def refresh_data(self) -> bool:
        """Force refresh of cached data if needed."""
        # Clear retrieval cache to force fresh fetches
        self._retrieval_cache.clear()
        # In a real implementation, this might also clear embedder cache if needed
        # self._embedder_cache.clear()
        return True

    def get_llm_accuracy(self) -> List[Dict[str, Any]]:
        """Get accuracy data for LLM models."""
        stats = self.db.get_llm_stats()
        return [
            {
                'model': self._get_model_display_info(model_name, 'llm'),
                'accuracy': model_stats['avg_accuracy'],
                'faithfulness': model_stats['avg_faithfulness'],
                'relevance': model_stats['avg_relevance']
            }
            for model_name, model_stats in stats.items()
        ]

    def get_embedder_accuracy(self) -> List[Dict[str, Any]]:
        """Get accuracy data for embedder models."""
        stats = self.db.get_embedder_stats()
        return [
            {
                'model': self._get_model_display_info(model_name, 'embedder'),
                'accuracy': model_stats['avg_accuracy'],
                'faithfulness': model_stats['avg_faithfulness'],
                'relevance': model_stats['avg_relevance']
            }
            for model_name, model_stats in stats.items()
        ]

    def get_reranker_accuracy(self) -> List[Dict[str, Any]]:
        """Get accuracy data for reranker models."""
        stats = self.db.get_reranker_stats()
        return [
            {
                'model': self._get_model_display_info(model_name, 'reranker'),
                'accuracy': model_stats['avg_accuracy'],
                'faithfulness': model_stats['avg_faithfulness'],
                'relevance': model_stats['avg_relevance']
            }
            for model_name, model_stats in stats.items()
        ]

    def get_query_enhancement_accuracy(self) -> List[Dict[str, Any]]:
        """Get accuracy data for query enhancement comparison."""
        stats = self.db.get_query_enhancement_comparison()
        return [
            {
                'model': self._get_model_display_info(qe_status, 'qe'),
                'accuracy': qe_stats['avg_accuracy'],
                'faithfulness': qe_stats['avg_faithfulness'],
                'relevance': qe_stats['avg_relevance']
            }
            for qe_status, qe_stats in stats.items()
        ]

    def get_token_usage(self) -> List[Dict[str, Any]]:
        """Get token usage data for models."""
        stats = self.db.get_token_usage_stats()
        return [
            {
                'model': model_name,
                'total_tokens': model_stats['total_tokens'],
                'prompt_tokens': model_stats['prompt_tokens'],
                'completion_tokens': model_stats['completion_tokens']
            }
            for model_name, model_stats in stats.items()
        ]

    def get_token_usage_overview(self) -> Dict[str, Any]:
        """Get overall token usage statistics."""
        token_stats = self.db.get_token_usage_stats()
        # Add retrieval chunks to the response
        retrieval_chunks_stats = self.db._get_retrieval_chunks_stats()
        token_stats.update(retrieval_chunks_stats)
        return token_stats

    def get_token_usage_by_embedder(self) -> List[Dict[str, Any]]:
        """Get token usage statistics by embedder model."""
        stats = self.db.get_token_usage_by_model('embedder')
        return [
            {
                'model': model_name,
                'total_queries': model_stats['total_queries'],
                'total_embedding_tokens': model_stats['total_embedding_tokens'],
                'total_reranking_tokens': model_stats['total_reranking_tokens'],
                'total_llm_tokens': model_stats['total_llm_tokens'],
                'total_tokens': model_stats['total_tokens'],
                'total_retrieval_chunks': model_stats['total_retrieval_chunks'],
                'avg_embedding_tokens': model_stats['avg_embedding_tokens'],
                'avg_reranking_tokens': model_stats['avg_reranking_tokens'],
                'avg_llm_tokens': model_stats['avg_llm_tokens'],
                'avg_total_tokens': model_stats['avg_total_tokens'],
                'avg_retrieval_chunks': model_stats['avg_retrieval_chunks']
            }
            for model_name, model_stats in stats.items()
        ]

    def get_token_usage_by_llm(self) -> List[Dict[str, Any]]:
        """Get token usage statistics by LLM model."""
        stats = self.db.get_token_usage_by_model('llm')
        return [
            {
                'model': model_name,
                'total_queries': model_stats['total_queries'],
                'total_embedding_tokens': model_stats['total_embedding_tokens'],
                'total_reranking_tokens': model_stats['total_reranking_tokens'],
                'total_llm_tokens': model_stats['total_llm_tokens'],
                'total_tokens': model_stats['total_tokens'],
                'avg_embedding_tokens': model_stats['avg_embedding_tokens'],
                'avg_reranking_tokens': model_stats['avg_reranking_tokens'],
                'avg_llm_tokens': model_stats['avg_llm_tokens'],
                'avg_total_tokens': model_stats['avg_total_tokens']
            }
            for model_name, model_stats in stats.items()
        ]

    def get_token_usage_by_reranker(self) -> List[Dict[str, Any]]:
        """Get token usage statistics by reranker model."""
        stats = self.db.get_token_usage_by_model('reranker')
        return [
            {
                'model': model_name,
                'total_queries': model_stats['total_queries'],
                'total_embedding_tokens': model_stats['total_embedding_tokens'],
                'total_reranking_tokens': model_stats['total_reranking_tokens'],
                'total_llm_tokens': model_stats['total_llm_tokens'],
                'total_tokens': model_stats['total_tokens'],
                'avg_embedding_tokens': model_stats['avg_embedding_tokens'],
                'avg_reranking_tokens': model_stats['avg_reranking_tokens'],
                'avg_llm_tokens': model_stats['avg_llm_tokens'],
                'avg_total_tokens': model_stats['avg_total_tokens']
            }
            for model_name, model_stats in stats.items()
        ]

    def get_token_usage_over_time(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get token usage data over time for time series chart."""
        return self.db.get_token_usage_over_time(hours)

    def get_token_costs(self, embedding_cost_per_1k: float = 0.0001, reranking_cost_per_1k: float = 0.001, llm_cost_per_1k: float = 0.002) -> Dict[str, Any]:
        """Calculate token costs based on provided rates per 1K tokens."""
        token_stats = self.db.get_token_usage_stats()
        total_embedding_cost = (token_stats.get('total_embedding_tokens', 0) / 1000) * embedding_cost_per_1k
        total_reranking_cost = (token_stats.get('total_reranking_tokens', 0) / 1000) * reranking_cost_per_1k
        total_llm_cost = (token_stats.get('total_llm_tokens', 0) / 1000) * llm_cost_per_1k

        total_cost = round(total_embedding_cost + total_reranking_cost + total_llm_cost, 6)

        cost_per_query = 0.0
        if token_stats.get('total_queries', 0) > 0:
            cost_per_query = round(total_cost / token_stats['total_queries'], 6)

        return {
            'embedding_cost': round(total_embedding_cost, 6),
            'reranking_cost': round(total_reranking_cost, 6),
            'llm_cost': round(total_llm_cost, 6),
            'total_cost': total_cost,
            'cost_per_query': cost_per_query
        }
    # Ground truth API
    def insert_ground_truth_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Insert multiple ground-truth rows into DB. Returns number inserted."""
        # Use the bulk insert helper on MetricsDB which accepts a list of dicts
        return self.db.insert_ground_truth_rows(rows)

    def get_ground_truth_list(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return list of ground-truth QA pairs."""
        return self.db.get_ground_truth_list(limit)

    def evaluate_ground_truth_with_ragas(self,
                                         llm_provider: str = 'ollama',
                                         model_name: Optional[str] = None,
                                         embedder_choice: str = 'ground_truth_evaluation',
                                         reranker_choice: str = 'none',
                                         api_key: str | None = None,
                                         limit: int | None = None,
                                         save_to_db: bool = True,
                                         generate_visualizations: bool = True) -> Dict[str, Any]:
        """
        Evaluate ground truth using Ragas framework with configurable LLM provider.

        This method provides standardized evaluation metrics:
        - faithfulness: How faithful the answer is to the context
        - context_recall: How much of the ground truth is covered by the context
        - context_relevance: How relevant the context is to the question

        Args:
            llm_provider: LLM provider to use ('ollama' or 'gemini')
            model_name: Model name (optional, defaults based on provider)
            api_key: Google API key for Gemini. If None, uses GOOGLE_API_KEY env var.
            limit: Limit number of ground truth items to evaluate
            save_to_db: Whether to save results to database

        Returns:
            Dictionary with Ragas evaluation results
        """
        try:
            from evaluation.backend_dashboard.ragas_evaluator import RagasEvaluator
            import json
            import os

            logger = logging.getLogger(__name__)
            logger.info(f"Starting Ragas evaluation with {llm_provider.upper()} LLM")

            # Get API key for Gemini if needed
            if llm_provider == 'gemini' and not api_key:
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    try:
                        import streamlit as st
                        api_key = st.secrets.get('GOOGLE_API_KEY')
                    except Exception:
                        pass

                if not api_key:
                    return {'error': 'GOOGLE_API_KEY must be provided via parameter, environment variable, or Streamlit secrets for Gemini provider'}

            # Initialize Ragas evaluator with specified provider
            ragas_evaluator = RagasEvaluator(llm_provider=llm_provider, model_name=model_name, api_key=api_key)

            # Load ground truth data from database instead of JSON file
            ground_truth_rows = self.get_ground_truth_list(limit=limit)
            if not ground_truth_rows:
                return {'error': 'No ground truth data found in database'}

            # Convert database format to Ragas format
            ground_truth_data = []
            for row in ground_truth_rows:
                # For Ragas evaluation, we need contexts. Try to use stored retrieved_context;
                # if missing, perform a fresh retrieval so ContextRelevance can be computed.
                contexts = []
                if row.get('retrieved_context'):
                    try:
                        # Try to parse as JSON list
                        contexts = json.loads(row['retrieved_context'])
                    except Exception:
                        # If not JSON, treat as single context string
                        contexts = [row['retrieved_context']]
                else:
                    # No retrieved_context stored — attempt to fetch retrieval now
                    try:
                        # Use a conservative top_k for evaluation contexts
                        retrieval_result = self._get_or_fetch_retrieval(
                            question=row['question'],
                            embedder_type=embedder_choice,
                            reranker_type=reranker_choice,
                            use_query_enhancement=False,
                            top_k=5,
                        )

                        # `fetch_retrieval` returns UI-style `sources`; each source has `full_text` or `snippet`
                        sources = retrieval_result.get('sources', []) if isinstance(retrieval_result, dict) else []
                        contexts = [s.get('full_text') or s.get('snippet') for s in sources if s]
                        logger.info(f"Fetched {len(contexts)} retrieval contexts for question: {row['question'][:60]}...")
                    except Exception as e:
                        logger.debug(f"Retrieval fetch failed for question '{row.get('question','')[:60]}...': {e}")
                        contexts = []

                ground_truth_data.append({
                    'question': row['question'],
                    'answer': row['predicted_answer'] if row['predicted_answer'] is not None else row['answer'],
                    'contexts': contexts,
                    'ground_truth': row['answer']  # The correct answer
                })

            logger.info(f"Evaluating {len(ground_truth_data)} ground truth samples with Ragas")

            # Prepare data for batch evaluation
            questions = []
            answers = []
            contexts_list = []
            ground_truths = []

            for item in ground_truth_data:
                questions.append(item['question'])
                answers.append(item['answer'])  # Back to 'answer'
                contexts_list.append(item['contexts'])
                ground_truths.append(item['ground_truth'])

            # Run batch evaluation
            results = ragas_evaluator.evaluate_batch(
                questions, answers, contexts_list, ground_truths
            )

            # Aggregate results
            faithfulness_scores = [r.faithfulness for r in results]
            context_recall_scores = [r.context_recall for r in results]
            context_relevance_scores = [r.context_relevance for r in results]
            answer_relevancy_scores = [r.answer_relevancy for r in results]

            summary = {
                'evaluation_type': 'ragas',
                'total_samples': len(results),
                'faithfulness': {
                    'mean': sum(faithfulness_scores) / len(faithfulness_scores),
                    'min': min(faithfulness_scores),
                    'max': max(faithfulness_scores),
                    'scores': faithfulness_scores
                },
                'context_recall': {
                    'mean': sum(context_recall_scores) / len(context_recall_scores),
                    'min': min(context_recall_scores),
                    'max': max(context_recall_scores),
                    'scores': context_recall_scores
                },
                'context_relevance': {
                    'mean': sum(context_relevance_scores) / len(context_relevance_scores) if context_relevance_scores else 0.0,
                    'min': min(context_relevance_scores) if context_relevance_scores else 0.0,
                    'max': max(context_relevance_scores) if context_relevance_scores else 0.0,
                    'scores': context_relevance_scores
                },
                'answer_relevancy': {
                    'mean': sum(answer_relevancy_scores) / len(answer_relevancy_scores) if answer_relevancy_scores else 0.0,
                    'min': min(answer_relevancy_scores) if answer_relevancy_scores else 0.0,
                    'max': max(answer_relevancy_scores) if answer_relevancy_scores else 0.0,
                    'scores': answer_relevancy_scores
                },
                'detailed_results': [
                    {
                        'question': r.question,
                        'answer': r.answer,
                        'contexts': r.contexts,
                        'ground_truth': r.ground_truth,
                        'faithfulness': r.faithfulness,
                        'context_recall': r.context_recall,
                        'context_relevance': r.context_relevance,
                        'answer_relevancy': r.answer_relevancy
                    }
                    for r in results
                ]
            }

            # Save to database if requested
            if save_to_db:
                try:
                    logger.info(f"Saving {len(results)} evaluation results to database...")

                    # Create a mapping of questions to database IDs for efficient lookup
                    gt_rows = self.get_ground_truth_list()
                    question_to_id = {gt['question']: gt['id'] for gt in gt_rows}

                    saved_count = 0
                    for result in results:
                        question = result.question
                        if question in question_to_id:
                            gt_id = question_to_id[question]

                            # Save evaluation results to the corresponding ground truth row
                            self.db.update_ground_truth_result(
                                gt_id=gt_id,
                                predicted_answer=result.answer,  # The generated answer
                                retrieved_context=json.dumps(result.contexts) if result.contexts else None,
                                retrieval_chunks=len(result.contexts) if result.contexts else 0,
                                faithfulness=result.faithfulness,
                                relevance=result.context_relevance,  # Ragas context_relevance maps to our relevance
                                evaluated_at=datetime.utcnow().isoformat()
                            )
                            saved_count += 1
                        else:
                            logger.warning(f"No matching ground truth found for question: {question[:50]}...")

                    logger.info(f"Successfully saved {saved_count}/{len(results)} evaluation results to database")

                    # Also save summary metrics to the metrics table for dashboard display
                    # Use the unified model name format: {embedder}_{reranker}_{llm}
                    model_name = f"{embedder_choice}_{reranker_choice}_{llm_provider}"
                    timestamp = datetime.utcnow().isoformat()

                    # Save one row per evaluated sample to metrics table
                    for i, result in enumerate(results):
                        self.db.insert_metric(
                            timestamp=timestamp,
                            query=result.question,
                            model=model_name,
                            embedder_model=embedder_choice,
                            llm_model=llm_provider,
                            reranker_model=reranker_choice,
                            query_enhanced=False,
                            latency=0.0,  # No latency for ground truth evaluation
                            faithfulness=result.faithfulness,
                            relevance=result.context_relevance,
                            recall=result.context_recall,
                            error=False,
                            error_message=None,
                            metadata=json.dumps({
                                "evaluation_type": "ragas_ground_truth",
                                "answer_relevancy": result.answer_relevancy,
                                "sample_index": i
                            })
                        )

                    logger.info(f"Saved {len(results)} summary metrics to metrics table for model '{model_name}'")

                except Exception as e:
                    logger.error(f"Failed to save evaluation results to database: {e}")
                    return {'error': f'Failed to save results: {str(e)}'}

            logger.info("Ragas evaluation completed successfully")

            # Generate visualizations if requested
            if generate_visualizations:
                try:
                    from evaluation.visualizations import RAGMetricsVisualizer
                    visualizer = RAGMetricsVisualizer("data/visualizations")

                    # Create config name based on parameters
                    config_name = f"{llm_provider.upper()}_{model_name or 'default'}"
                    if limit:
                        config_name += f"_limit_{limit}"

                    viz_results = visualizer.visualize_from_ragas_output(
                        summary,
                        config_name=config_name,
                        title_prefix=f"RAG Evaluation - {config_name}",
                        save_charts=True,
                        show_charts=False
                    )

                    if "error" not in viz_results:
                        summary["visualizations"] = viz_results
                        logger.info(f"Generated {len(viz_results)} visualizations")
                    else:
                        logger.warning(f"Failed to generate visualizations: {viz_results['error']}")

                except Exception as viz_error:
                    logger.warning(f"Visualization generation failed: {viz_error}")

            return summary

        except Exception as e:
            logger.error(f"Error in Ragas evaluation: {str(e)}")
            return {'error': str(e)}
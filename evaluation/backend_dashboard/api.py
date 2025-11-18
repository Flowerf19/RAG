"""
Backend Dashboard API
Provides data access for the evaluation dashboard from different system stages.
"""

from typing import Dict, List, Any
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

    def evaluate_ground_truth_with_semantic_similarity(self,
                                                      embedder_type: str = "ollama",
                                                      reranker_type: str = "none",
                                                      use_query_enhancement: bool = True,
                                                      top_k: int = 10,
                                                      api_tokens: dict | None = None,
                                                      llm_model: str | None = None,
                                                      limit: int | None = None,
                                                      save_to_db: bool = True) -> Dict[str, Any]:
        """
        Enhanced ground truth evaluation with semantic similarity to true source.
        
        Returns evaluation results including semantic similarity scores.
        """
        # Delegate to the refactored semantic module
        logger = logging.getLogger(__name__)
        logger.info("Starting semantic similarity evaluation (backend wrapper)")

        try:
            from evaluation.backend_dashboard.semantic import run_semantic_similarity
        except ModuleNotFoundError:
            # Fallback if module not found
            return {'error': 'Semantic similarity module not found'}

        res = run_semantic_similarity(
            self.db,
            self._get_or_fetch_retrieval,  # Use cached retrieval
            self._get_or_create_embedder,
            embedder_type=embedder_type,
            reranker_type=reranker_type,
            use_query_enhancement=use_query_enhancement,
            top_k=top_k,
            api_tokens=api_tokens,
            llm_model=llm_model,
            limit=limit
        )

        if save_to_db:
            self.save_evaluation_results_to_db(
                evaluation_type='semantic_similarity',
                results=res,
                embedder_type=embedder_type,
                reranker_type=reranker_type,
                use_query_enhancement=use_query_enhancement
            )

        logger.info("Completed semantic similarity evaluation")
        return res

    def save_evaluation_results_to_db(self,
                                     evaluation_type: str,
                                     results: Dict[str, Any],
                                     embedder_type: str = "ollama",
                                     reranker_type: str = "none",
                                     use_query_enhancement: bool = True,
                                     llm_model: str = None) -> None:
        """
        Save evaluation results to database for historical tracking.

        Args:
            evaluation_type: Type of evaluation ('semantic_similarity', 'recall', 'relevance', 'full_suite')
            results: Results dictionary from evaluation method
            embedder_type: Embedder model used
            reranker_type: Reranker model used
            use_query_enhancement: Whether QEM was used
            llm_model: LLM model used (for faithfulness evaluation)
        """
        try:
            summary = results.get('summary', {})
            detailed_results = results.get('results', [])

            # Create metadata for the evaluation run
            metadata = {
                'evaluation_type': evaluation_type,
                'embedder_used': embedder_type,
                'reranker_used': reranker_type,
                'query_enhancement_used': use_query_enhancement,
                'timestamp': datetime.utcnow().isoformat(),
                'total_questions': summary.get('total_questions', 0),
                'processed': summary.get('processed', 0),
                'errors': summary.get('errors', 0)
            }

            # Add LLM model if provided (for faithfulness evaluation)
            if llm_model:
                metadata['llm_used'] = llm_model

            # Add evaluation-specific metrics to metadata
            if evaluation_type == 'semantic_similarity':
                metadata.update({
                    'avg_semantic_similarity': summary.get('avg_semantic_similarity', 0),
                    'avg_best_match_score': summary.get('avg_best_match_score', 0),
                    'total_chunks_above_threshold': summary.get('total_chunks_above_threshold', 0)
                })
            elif evaluation_type == 'recall':
                metadata.update({
                    'overall_recall': summary.get('overall_recall', 0),
                    'overall_precision': summary.get('overall_precision', 0),
                    'overall_f1_score': summary.get('overall_f1_score', 0),
                    'total_true_positives': summary.get('total_true_positives', 0),
                    'total_false_positives': summary.get('total_false_positives', 0),
                    'total_false_negatives': summary.get('total_false_negatives', 0)
                })
            elif evaluation_type == 'relevance':
                metadata.update({
                    'avg_overall_relevance': summary.get('avg_overall_relevance', 0),
                    'avg_chunk_relevance': summary.get('avg_chunk_relevance', 0),
                    'global_avg_relevance': summary.get('global_avg_relevance', 0),
                    'global_high_relevance_ratio': summary.get('global_high_relevance_ratio', 0),
                    'global_relevant_ratio': summary.get('global_relevant_ratio', 0),
                    'relevance_distribution': summary.get('relevance_distribution', {})
                })
            elif evaluation_type == 'faithfulness':
                metadata.update({
                    'avg_faithfulness': summary.get('avg_faithfulness', 0),
                    'global_avg_faithfulness': summary.get('global_avg_faithfulness', 0),
                    'global_high_faithfulness_ratio': summary.get('global_high_faithfulness_ratio', 0),
                    'global_faithful_ratio': summary.get('global_faithful_ratio', 0),
                    'faithfulness_distribution': summary.get('faithfulness_distribution', {})
                })

            # Save summary metrics to metrics table
            self.db.insert_metric(
                query=f"Evaluation: {evaluation_type}",
                model=f"{embedder_type}_{reranker_type}",
                embedder_model=embedder_type,
                reranker_model=reranker_type,
                llm_model=llm_model,
                query_enhanced=use_query_enhancement,
                recall=summary.get('overall_recall') if evaluation_type == 'recall' else None,
                relevance=summary.get('avg_overall_relevance') if evaluation_type == 'relevance' else summary.get('avg_semantic_similarity') if evaluation_type == 'semantic_similarity' else None,
                faithfulness=summary.get('avg_faithfulness') if evaluation_type == 'faithfulness' else None,
                metadata=json.dumps(metadata)
            )

            # Save detailed results to ground truth table
            for result in detailed_results:
                if evaluation_type == 'semantic_similarity':
                    self.db.update_ground_truth_result(
                        gt_id=result.get('ground_truth_id'),
                        retrieval_chunks=result.get('retrieved_chunks', 0),
                        retrieval_recall_at_k=result.get('chunks_above_threshold', 0),
                        answer_token_recall=result.get('semantic_similarity', 0.0),
                        answer_f1=result.get('best_match_score', 0.0)
                    )
                elif evaluation_type == 'recall':
                    self.db.update_ground_truth_result(
                        gt_id=result.get('ground_truth_id'),
                        retrieval_chunks=result.get('retrieved_chunks', 0),
                        retrieval_recall_at_k=result.get('true_positives', 0),
                        answer_token_recall=result.get('recall', 0.0),
                        answer_token_precision=result.get('precision', 0.0),
                        answer_f1=result.get('f1_score', 0.0)
                    )
                elif evaluation_type == 'relevance':
                    self.db.update_ground_truth_result(
                        gt_id=result.get('ground_truth_id'),
                        retrieval_chunks=result.get('retrieved_chunks', 0),
                        answer_token_recall=result.get('overall_relevance', 0.0),
                        answer_token_precision=result.get('avg_chunk_relevance', 0.0)
                    )
                elif evaluation_type == 'faithfulness':
                    self.db.update_ground_truth_result(
                        gt_id=result.get('ground_truth_id'),
                        retrieval_chunks=result.get('context_length', 0),
                        answer_token_recall=result.get('faithfulness_score', 0.0)
                    )

        except Exception as e:
            print(f"Warning: Failed to save evaluation results to database: {e}")
            # Don't raise exception to avoid breaking the evaluation flow

    def evaluate_recall(self,
                        embedder_type: str = "huggingface_local",
                        reranker_type: str = "none",
                        use_query_enhancement: bool = True,
                        top_k: int = 10,
                        similarity_threshold: float = 0.5,
                        limit: int | None = None,
                        save_to_db: bool = True) -> Dict[str, Any]:
        """
        Evaluate recall metric for RAG system.

        Recall = True Positives / (True Positives + False Negatives)
        Where:
        - True Positives: Retrieved chunks that are semantically similar to ground truth source
        - False Negatives: Relevant chunks in database that were not retrieved

        Returns recall evaluation results.
        """
        # Delegate to the refactored recall module
        logger = logging.getLogger(__name__)
        logger.info("Starting recall evaluation (backend wrapper)")
        from evaluation.backend_dashboard.recall import run_recall

        res = run_recall(
            self.db,
            self._get_or_fetch_retrieval,  # Use cached retrieval
            self.evaluate_semantic_similarity_with_source,
            embedder_type=embedder_type,
            reranker_type=reranker_type,
            use_query_enhancement=use_query_enhancement,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            limit=limit
        )

        if save_to_db:
            self.save_evaluation_results_to_db(
                evaluation_type='recall',
                results=res,
                embedder_type=embedder_type,
                reranker_type=reranker_type,
                use_query_enhancement=use_query_enhancement
            )

        logger.info("Completed recall evaluation")

        return res

    def evaluate_relevance(self,
                           embedder_type: str = "huggingface_local",
                           reranker_type: str = "none",
                           use_query_enhancement: bool = True,
                           top_k: int = 10,
                           limit: int | None = None,
                           save_to_db: bool = True) -> Dict[str, Any]:
        """
        Evaluate relevance metric for RAG system.

        Relevance measures how well retrieved content matches the user's query.
        This includes both semantic similarity to ground truth source and
        query-chunk relevance scoring.

        Returns relevance evaluation results with detailed scoring.
        """
        # Delegate to the refactored relevance module
        logger = logging.getLogger(__name__)
        logger.info("Starting relevance evaluation (backend wrapper)")
        from evaluation.backend_dashboard.relevance import run_relevance

        res = run_relevance(
            self.db,
            self._get_or_fetch_retrieval,  # Use cached retrieval
            self.evaluate_semantic_similarity_with_source,
            embedder_type=embedder_type,
            reranker_type=reranker_type,
            use_query_enhancement=use_query_enhancement,
            top_k=top_k,
            limit=limit
        )

        if save_to_db:
            self.save_evaluation_results_to_db(
                evaluation_type='relevance',
                results=res,
                embedder_type=embedder_type,
                reranker_type=reranker_type,
                use_query_enhancement=use_query_enhancement
            )

        logger.info("Completed relevance evaluation")

        return res
        

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
        embedder_stats = self.db.get_embedder_stats()
        llm_stats = self.db.get_llm_stats()
        reranker_stats = self.db.get_reranker_stats()
        qe_comparison = self.db.get_query_enhancement_comparison()

        # Convert to list format for display
        embedding_models = []
        llm_models = []
        reranking_models = []
        query_enhancement = []

        for model_name, model_stats in embedder_stats.items():
            embedding_models.append({
                'model': self._get_model_display_info(model_name, 'embedder'),
                'queries': model_stats['total_queries'],
                'accuracy': model_stats['avg_accuracy'],
                'faithfulness': model_stats['avg_faithfulness'],
                'relevance': model_stats['avg_relevance'],
                'recall': model_stats['avg_recall'],
                'latency': model_stats['avg_latency'],
                'error_rate': model_stats['error_rate']
            })

        for model_name, model_stats in llm_stats.items():
            llm_models.append({
                'model': self._get_model_display_info(model_name, 'llm'),
                'queries': model_stats['total_queries'],
                'accuracy': model_stats['avg_accuracy'],
                'faithfulness': model_stats['avg_faithfulness'],
                'relevance': model_stats['avg_relevance'],
                'recall': model_stats['avg_recall'],
                'latency': model_stats['avg_latency'],
                'error_rate': model_stats['error_rate']
            })

        for model_name, model_stats in reranker_stats.items():
            reranking_models.append({
                'model': self._get_model_display_info(model_name, 'reranker'),
                'queries': model_stats['total_queries'],
                'accuracy': model_stats['avg_accuracy'],
                'faithfulness': model_stats['avg_faithfulness'],
                'relevance': model_stats['avg_relevance'],
                'recall': model_stats['avg_recall'],
                'latency': model_stats['avg_latency'],
                'error_rate': model_stats['error_rate']
            })

        for qe_status, qe_stats in qe_comparison.items():
            query_enhancement.append({
                'model': self._get_model_display_info(qe_status, 'qe'),
                'queries': qe_stats['total_queries'],
                'accuracy': qe_stats['avg_accuracy'],
                'latency': qe_stats['avg_latency'],
                'error_rate': qe_stats['error_rate']
            })

        return {
            'reranking': sorted(reranking_models, key=lambda x: x['accuracy'], reverse=True),
            'embedding': sorted(embedding_models, key=lambda x: x['accuracy'], reverse=True),
            'llm': sorted(llm_models, key=lambda x: x['accuracy'], reverse=True),
            'query_enhancement': sorted(query_enhancement, key=lambda x: x['accuracy'], reverse=True)
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

    def get_ground_truth_list(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Return list of ground-truth QA pairs."""
        return self.db.get_ground_truth_list(limit)

    def evaluate_faithfulness(self,
                              embedder_type: str = "huggingface_local",
                              reranker_type: str = "none",
                              llm_choice: str = "gemini",
                              use_query_enhancement: bool = True,
                              top_k: int = 10,
                              limit: int | None = None,
                              save_to_db: bool = True) -> Dict[str, Any]:
        """
        Evaluate faithfulness metric for RAG system.

        Faithfulness measures how well the generated answer is grounded in the retrieved context.
        This uses LLM-based evaluation to score answer faithfulness.

        Returns faithfulness evaluation results with detailed scoring.
        """
        # Delegate to the refactored faithfulness module
        logger = logging.getLogger(__name__)
        logger.info("Starting faithfulness evaluation (backend wrapper)")
        from llm.client_factory import LLMClientFactory
        from evaluation.evaluators.auto_evaluator import AutoEvaluator
        from evaluation.backend_dashboard.faithfulness import run_faithfulness

        res = run_faithfulness(
            self.db,
            self._get_or_fetch_retrieval,  # Use cached retrieval
            LLMClientFactory,
            AutoEvaluator,
            embedder_type=embedder_type,
            reranker_type=reranker_type,
            llm_choice=llm_choice,
            use_query_enhancement=use_query_enhancement,
            top_k=top_k,
            limit=limit
        )

        if save_to_db:
            self.save_evaluation_results_to_db(
                evaluation_type='faithfulness',
                results=res,
                embedder_type=embedder_type,
                reranker_type=reranker_type,
                llm_model=llm_choice,
                use_query_enhancement=use_query_enhancement
            )

        logger.info("Completed faithfulness evaluation")

        return res

    def evaluate_all_metrics_batch(self,
                                   embedder_type: str = "huggingface_local",
                                   reranker_type: str = "none",
                                   llm_choice: str = "gemini",
                                   use_query_enhancement: bool = True,
                                   top_k: int = 10,
                                   limit: int | None = None,
                                   save_to_db: bool = True) -> Dict[str, Any]:
        """
        Evaluate all metrics (semantic similarity, recall, relevance, faithfulness) in batch.
        Performs retrieval once per question and computes all metrics to avoid redundant fetches.
        
        Returns combined evaluation results for all metrics.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting batch evaluation of all metrics")
        
        # Get ground truth data
        ground_truth_rows = self.get_ground_truth_list(limit=limit)
        if not ground_truth_rows:
            return {'error': 'No ground truth data found'}
        
        # Perform retrieval once for all questions
        retrieval_results = {}
        for gt_row in ground_truth_rows:
            question = gt_row.get('question', '').strip()
            if question:
                try:
                    result = self._get_or_fetch_retrieval(
                        question=question,
                        embedder_type=embedder_type,
                        reranker_type=reranker_type,
                        use_query_enhancement=use_query_enhancement,
                        top_k=top_k
                    )
                    retrieval_results[gt_row['id']] = result
                except Exception as e:
                    logger.warning(f"Failed to retrieve for question '{question}': {e}")
                    retrieval_results[gt_row['id']] = {'error': str(e)}
        
        # Evaluate all metrics using cached retrieval results
        results = {
            'semantic_similarity': self._evaluate_semantic_similarity_batch(retrieval_results, ground_truth_rows),
            'recall': self._evaluate_recall_batch(retrieval_results, ground_truth_rows, embedder_type),
            'relevance': self._evaluate_relevance_batch(retrieval_results, ground_truth_rows, embedder_type),
            'faithfulness': self._evaluate_faithfulness_batch(retrieval_results, ground_truth_rows, embedder_type, reranker_type, llm_choice, use_query_enhancement, top_k)
        }
        
        # Save results to DB if requested
        if save_to_db:
            for metric_type, result in results.items():
                if isinstance(result, dict) and 'summary' in result:
                    self.save_evaluation_results_to_db(
                        evaluation_type=metric_type,
                        results=result,
                        embedder_type=embedder_type,
                        reranker_type=reranker_type,
                        llm_model=llm_choice if metric_type == 'faithfulness' else None,
                        use_query_enhancement=use_query_enhancement
                    )
        
        logger.info("Completed batch evaluation of all metrics")
        return results

    def _evaluate_semantic_similarity_batch(self, retrieval_results: Dict[int, Dict], ground_truth_rows: List[Dict]) -> Dict[str, Any]:
        """Batch evaluate semantic similarity using cached retrieval results."""
        from evaluation.backend_dashboard.semantic import run_semantic_similarity_batch
        
        return run_semantic_similarity_batch(
            self.db,
            retrieval_results,
            ground_truth_rows
        )

    def _evaluate_recall_batch(self, retrieval_results: Dict[int, Dict], ground_truth_rows: List[Dict], embedder_type: str) -> Dict[str, Any]:
        """Batch evaluate recall using cached retrieval results."""
        from evaluation.backend_dashboard.recall import run_recall_batch
        
        return run_recall_batch(
            self.db,
            retrieval_results,
            ground_truth_rows,
            self.evaluate_semantic_similarity_with_source,
            embedder_type=embedder_type
        )

    def _evaluate_relevance_batch(self, retrieval_results: Dict[int, Dict], ground_truth_rows: List[Dict], embedder_type: str) -> Dict[str, Any]:
        """Batch evaluate relevance using cached retrieval results."""
        from evaluation.backend_dashboard.relevance import run_relevance_batch
        
        return run_relevance_batch(
            self.db,
            retrieval_results,
            ground_truth_rows,
            self.evaluate_semantic_similarity_with_source,
            embedder_type=embedder_type
        )

    def _evaluate_faithfulness_batch(self, retrieval_results: Dict[int, Dict], ground_truth_rows: List[Dict], 
                                    embedder_type: str, reranker_type: str, llm_choice: str, 
                                    use_query_enhancement: bool, top_k: int) -> Dict[str, Any]:
        """Batch evaluate faithfulness using cached retrieval results."""
        from evaluation.backend_dashboard.faithfulness import run_faithfulness_batch
        
        return run_faithfulness_batch(
            self.db,
            retrieval_results,
            ground_truth_rows,
            embedder_type=embedder_type,
            reranker_type=reranker_type,
            llm_choice=llm_choice,
            use_query_enhancement=use_query_enhancement,
            top_k=top_k
        )

    def evaluate_semantic_similarity_with_source(self, 
                                                retrieved_sources: List[Dict[str, Any]], 
                                                ground_truth_id: int,
                                                embedder_type: str = "huggingface_local") -> Dict[str, Any]:
        """
        Evaluate semantic similarity between retrieved content and true source from database.
        
        Args:
            retrieved_sources: List of retrieved source dictionaries from RAG
            ground_truth_id: ID of ground truth entry to get true source
            embedder_type: Embedder type to use for semantic similarity
            
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
            
            # Get cached embedder instance
            embedder = self._get_or_create_embedder(embedder_type)
            
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
                    'embedder_used': embedder_type
                }
            else:
                return {
                    'semantic_similarity': 0.0,
                    'best_match_score': 0.0,
                    'matched_chunks': [],
                    'total_chunks_evaluated': 0,
                    'chunks_above_threshold': 0,
                    'true_source_length': len(true_source),
                    'embedder_used': embedder_type
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'semantic_similarity': 0.0,
                'best_match_score': 0.0,
                'matched_chunks': []
            }
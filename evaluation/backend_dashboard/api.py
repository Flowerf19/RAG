"""
Backend Dashboard API
Provides data access for the evaluation dashboard from different system stages.
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta

from evaluation.metrics.database import MetricsDB


class BackendDashboard:
    """Backend API for evaluation dashboard data access."""

    def __init__(self, db_path: str = "data/metrics.db"):
        """Initialize with database connection."""
        self.db = MetricsDB(db_path)

    def evaluate_ground_truth(self,
                              embedder_type: str = "ollama",
                              reranker_type: str = "none",
                              use_query_enhancement: bool = True,
                              top_k: int = 5,
                              api_tokens: dict | None = None,
                              llm_model: str | None = None,
                              limit: int | None = None) -> Dict[str, Any]:
        """Run RAG retrieval for all ground-truth rows and persist results.

        Returns a summary dict with counts of processed and errors.
        """
        # Lazy import to avoid heavy startup costs
        from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval

        rows = self.db.get_ground_truth_list(limit=limit or 10000)
        processed = 0
        errors = 0
        errors_list = []

        for r in rows:
            gt_id = r.get('id')
            question = r.get('question')
            try:
                result = fetch_retrieval(
                    query_text=question,
                    top_k=top_k,
                    embedder_type=embedder_type,
                    reranker_type=reranker_type,
                    use_query_enhancement=use_query_enhancement,
                    api_tokens=api_tokens,
                    llm_model=llm_model,
                )

                context = result.get('context', '')
                sources = result.get('sources', [])
                retrieval_info = result.get('retrieval_info', {})

                # Flatten sources to a string for storage
                if isinstance(sources, list):
                    try:
                        import json as _json
                        sources_text = _json.dumps(sources, ensure_ascii=False)
                    except Exception:
                        sources_text = str(sources)
                else:
                    sources_text = str(sources)

                chunks = retrieval_info.get('total_retrieved', retrieval_info.get('final_count', 0))

                # Compute simple evaluation metrics
                try:
                    import re

                    gt_answer = (r.get('answer') or '').strip()

                    def _norm_tokens(text: str) -> list:
                        if not text:
                            return []
                        # extract word-like tokens (unicode-aware)
                        toks = re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)
                        return toks

                    # Build texts from sources list (if available)
                    source_texts = []
                    if isinstance(sources, list):
                        for s in sources:
                            if isinstance(s, dict):
                                # common keys
                                txt = s.get('text') or s.get('content') or s.get('snippet') or s.get('title') or str(s)
                                source_texts.append(str(txt))
                            else:
                                source_texts.append(str(s))
                    else:
                        source_texts.append(str(sources))

                    joined_sources = "\n".join(source_texts)

                    # Normalized presence check for retrieval recall@k
                    gt_norm = " ".join(_norm_tokens(gt_answer))
                    found_in_sources = False
                    if gt_norm:
                        if gt_norm in " ".join(_norm_tokens(joined_sources)):
                            found_in_sources = True
                        elif gt_norm in " ".join(_norm_tokens(context)):
                            found_in_sources = True

                    retrieval_recall_at_k = 1 if found_in_sources else 0

                    # Token-overlap metrics between GT answer and predicted context
                    pred_norm_toks = _norm_tokens(context)
                    gt_norm_toks = _norm_tokens(gt_answer)

                    inter = set(gt_norm_toks) & set(pred_norm_toks)
                    recall_val = len(inter) / max(1, len(gt_norm_toks))
                    precision_val = len(inter) / max(1, len(pred_norm_toks)) if pred_norm_toks else 0.0
                    f1_val = 0.0
                    if (precision_val + recall_val) > 0:
                        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)

                except Exception:
                    # On any error, fallback to zeros
                    retrieval_recall_at_k = 0
                    recall_val = 0.0
                    precision_val = 0.0
                    f1_val = 0.0

                # Persist into ground_truth_qa (include evaluation metrics)
                self.db.update_ground_truth_result(
                    gt_id=gt_id,
                    predicted_answer=context[:4000],
                    retrieved_context=context,
                    retrieved_sources=sources_text,
                    retrieval_chunks=chunks,
                    retrieval_recall_at_k=retrieval_recall_at_k,
                    answer_token_recall=round(recall_val, 4),
                    answer_token_precision=round(precision_val, 4),
                    answer_f1=round(f1_val, 4),
                )

                # Compute embedding-based faithfulness/relevance using a local embedder
                faith_val = None
                rel_val = None
                try:
                    from embedders.embedder_factory import EmbedderFactory
                    factory = EmbedderFactory()

                    emb = None
                    et = embedder_type.lower() if isinstance(embedder_type, str) else ''
                    # Try to create matching embedder (best-effort)
                    if et == 'ollama' or 'gemma' in et:
                        try:
                            emb = factory.create_gemma()
                        except Exception:
                            emb = None
                    elif et in ('huggingface_local', 'huggingface-local', 'hf_local'):
                        try:
                            emb = factory.create_huggingface_local()
                        except Exception:
                            emb = None
                    elif et in ('huggingface_api', 'huggingface-api', 'hf_api'):
                        try:
                            emb = factory.create_huggingface_api()
                        except Exception:
                            emb = None
                    elif et == 'e5_large_instruct':
                        try:
                            emb = factory.create_e5_large_instruct()
                        except Exception:
                            emb = None
                    elif et == 'e5_base':
                        try:
                            emb = factory.create_e5_base()
                        except Exception:
                            emb = None
                    elif et == 'gte_multilingual_base':
                        try:
                            emb = factory.create_gte_multilingual_base()
                        except Exception:
                            emb = None

                    if emb is not None:
                        from evaluation.evaluators.auto_evaluator import AutoEvaluator
                        evaluator = AutoEvaluator(embedder=emb)
                        # Build joined_sources from sources list so we evaluate answer vs actual source text
                        try:
                            source_texts = []
                            if isinstance(sources, list):
                                for s in sources:
                                    if isinstance(s, dict):
                                        txt = s.get('text') or s.get('content') or s.get('snippet') or s.get('title') or str(s)
                                        source_texts.append(str(txt))
                                    else:
                                        source_texts.append(str(s))
                            else:
                                source_texts.append(str(sources))
                            joined_sources = "\n".join(source_texts)
                        except Exception:
                            joined_sources = context

                        # Evaluate predicted/context against the joined sources (correct comparison)
                        faith_val, rel_val = evaluator.evaluate_response(question, context, joined_sources)
                except Exception:
                    faith_val = None
                    rel_val = None

                # Persist faithfulness/relevance if computed
                if faith_val is not None or rel_val is not None:
                    try:
                        self.db.update_ground_truth_result(
                            gt_id=gt_id,
                            predicted_answer=context[:4000],
                            retrieved_context=context,
                            retrieved_sources=sources_text,
                            retrieval_chunks=chunks,
                            retrieval_recall_at_k=retrieval_recall_at_k,
                            answer_token_recall=round(recall_val, 4),
                            answer_token_precision=round(precision_val, 4),
                            answer_f1=round(f1_val, 4),
                            faithfulness=float(faith_val) if faith_val is not None else None,
                            relevance=float(rel_val) if rel_val is not None else None,
                        )
                    except Exception:
                        pass

                processed += 1

            except Exception as e:
                errors += 1
                errors_list.append({'id': gt_id, 'error': str(e)})

        return {
            'processed': processed,
            'errors': errors,
            'error_details': errors_list
        }

    def get_overview_stats(self) -> Dict[str, Any]:
        """Get overall system statistics for dashboard header."""
        llm_stats = self.db.get_llm_stats()

        if not llm_stats:
            return {
                'total_queries': 0,
                'avg_accuracy': 0.0,
                'avg_latency': 0.0,
                'error_rate': 0.0,
                'model_count': 0
            }

        # Calculate system-wide averages based on LLM models
        total_queries = sum(model['total_queries'] for model in llm_stats.values())
        weighted_accuracy = sum(model['avg_accuracy'] * model['total_queries'] for model in llm_stats.values())
        weighted_latency = sum(model['avg_latency'] * model['total_queries'] for model in llm_stats.values())
        total_errors = sum(model['error_rate'] * model['total_queries'] / 100 for model in llm_stats.values())

        return {
            'total_queries': total_queries,
            'avg_accuracy': round(weighted_accuracy / total_queries, 3) if total_queries > 0 else 0.0,
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
                'model': model_name,
                'queries': model_stats['total_queries'],
                'accuracy': model_stats['avg_accuracy'],
                'latency': model_stats['avg_latency'],
                'error_rate': model_stats['error_rate']
            })

        for model_name, model_stats in llm_stats.items():
            llm_models.append({
                'model': model_name,
                'queries': model_stats['total_queries'],
                'accuracy': model_stats['avg_accuracy'],
                'latency': model_stats['avg_latency'],
                'error_rate': model_stats['error_rate']
            })

        for model_name, model_stats in reranker_stats.items():
            reranking_models.append({
                'model': model_name,
                'queries': model_stats['total_queries'],
                'accuracy': model_stats['avg_accuracy'],
                'latency': model_stats['avg_latency'],
                'error_rate': model_stats['error_rate']
            })

        for qe_status, qe_stats in qe_comparison.items():
            query_enhancement.append({
                'model': qe_status,
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
        # This would require more complex SQL queries
        # For now, return mock data structure
        return [
            {
                'timestamp': (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                'avg_latency': round(1.0 + 0.5 * (i % 3), 3),  # Mock varying latency
                'model': 'all_models'
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
        # In a real implementation, this might clear caches or re-query data
        return True

    def get_llm_accuracy(self) -> List[Dict[str, Any]]:
        """Get accuracy data for LLM models."""
        stats = self.db.get_llm_stats()
        return [
            {
                'model': model_name,
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
                'model': model_name,
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
                'model': model_name,
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
                'model': qe_status,
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
        return self.db.get_ground_truth(limit)
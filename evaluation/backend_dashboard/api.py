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
"""
Metrics Database Manager
Handles SQLite database operations for storing and retrieving evaluation metrics.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class MetricsDB:
    """SQLite database manager for RAG evaluation metrics."""

    def __init__(self, db_path: str = "data/metrics.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _create_tables(self):
        """Create metrics table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedder_model TEXT,
                    llm_model TEXT,
                    reranker_model TEXT,
                    query_enhanced BOOLEAN DEFAULT FALSE,
                    latency REAL,
                    faithfulness REAL,
                    relevance REAL,
                    error BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            # Add new columns if they don't exist (migration)
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN embedder_model TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN llm_model TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN reranker_model TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN query_enhanced BOOLEAN DEFAULT FALSE")
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()

    def insert_metric(self, metric: Dict) -> int:
        """Insert a single metric record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO metrics
                (timestamp, query, model, embedder_model, llm_model, reranker_model, query_enhanced, latency, faithfulness, relevance, error, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.get('timestamp', datetime.utcnow().isoformat()),
                metric['query'],
                metric.get('model', ''),
                metric.get('embedder_model'),
                metric.get('llm_model'),
                metric.get('reranker_model'),
                metric.get('query_enhanced', False),
                metric.get('latency'),
                metric.get('faithfulness'),
                metric.get('relevance'),
                metric.get('error', False),
                metric.get('error_message'),
                json.dumps(metric.get('metadata', {}))
            ))
            conn.commit()
            return cursor.lastrowid

    def get_metrics(self, model: Optional[str] = None, limit: int = 1000) -> List[Dict]:
        """Retrieve metrics, optionally filtered by model."""
        query = "SELECT * FROM metrics"
        params = []

        if model:
            query += " WHERE model = ?"
            params.append(model)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_model_stats(self) -> Dict[str, Dict]:
        """Get aggregated statistics for each model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    model,
                    COUNT(*) as total_queries,
                    AVG(latency) as avg_latency,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(relevance) as avg_relevance,
                    SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) as error_count
                FROM metrics
                GROUP BY model
            """)

            stats = {}
            for row in cursor.fetchall():
                model = row['model']
                total = row['total_queries']
                stats[model] = {
                    'total_queries': total,
                    'avg_latency': round(row['avg_latency'] or 0, 3),
                    'avg_faithfulness': round(row['avg_faithfulness'] or 0, 3),
                    'avg_relevance': round(row['avg_relevance'] or 0, 3),
                    'avg_accuracy': round(((row['avg_faithfulness'] or 0) + (row['avg_relevance'] or 0)) / 2, 3),
                    'error_rate': round((row['error_count'] or 0) / total * 100, 2) if total > 0 else 0
                }

            return stats

    def get_embedder_stats(self) -> Dict[str, Dict]:
        """Get aggregated statistics for each embedder model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    embedder_model,
                    COUNT(*) as total_queries,
                    AVG(latency) as avg_latency,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(relevance) as avg_relevance,
                    SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) as error_count
                FROM metrics
                WHERE embedder_model IS NOT NULL
                GROUP BY embedder_model
            """)

            stats = {}
            for row in cursor.fetchall():
                model = row['embedder_model']
                total = row['total_queries']
                stats[model] = {
                    'total_queries': total,
                    'avg_latency': round(row['avg_latency'] or 0, 3),
                    'avg_faithfulness': round(row['avg_faithfulness'] or 0, 3),
                    'avg_relevance': round(row['avg_relevance'] or 0, 3),
                    'avg_accuracy': round(((row['avg_faithfulness'] or 0) + (row['avg_relevance'] or 0)) / 2, 3),
                    'error_rate': round((row['error_count'] or 0) / total * 100, 2) if total > 0 else 0
                }

            return stats

    def get_llm_stats(self) -> Dict[str, Dict]:
        """Get aggregated statistics for each LLM model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    llm_model,
                    COUNT(*) as total_queries,
                    AVG(latency) as avg_latency,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(relevance) as avg_relevance,
                    SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) as error_count
                FROM metrics
                WHERE llm_model IS NOT NULL
                GROUP BY llm_model
            """)

            stats = {}
            for row in cursor.fetchall():
                model = row['llm_model']
                total = row['total_queries']
                stats[model] = {
                    'total_queries': total,
                    'avg_latency': round(row['avg_latency'] or 0, 3),
                    'avg_faithfulness': round(row['avg_faithfulness'] or 0, 3),
                    'avg_relevance': round(row['avg_relevance'] or 0, 3),
                    'avg_accuracy': round(((row['avg_faithfulness'] or 0) + (row['avg_relevance'] or 0)) / 2, 3),
                    'error_rate': round((row['error_count'] or 0) / total * 100, 2) if total > 0 else 0
                }

            return stats

    def get_reranker_stats(self) -> Dict[str, Dict]:
        """Get aggregated statistics for each reranker model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    reranker_model,
                    COUNT(*) as total_queries,
                    AVG(latency) as avg_latency,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(relevance) as avg_relevance,
                    SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) as error_count
                FROM metrics
                WHERE reranker_model IS NOT NULL AND reranker_model != 'none'
                GROUP BY reranker_model
            """)

            stats = {}
            for row in cursor.fetchall():
                model = row['reranker_model']
                total = row['total_queries']
                stats[model] = {
                    'total_queries': total,
                    'avg_latency': round(row['avg_latency'] or 0, 3),
                    'avg_faithfulness': round(row['avg_faithfulness'] or 0, 3),
                    'avg_relevance': round(row['avg_relevance'] or 0, 3),
                    'avg_accuracy': round(((row['avg_faithfulness'] or 0) + (row['avg_relevance'] or 0)) / 2, 3),
                    'error_rate': round((row['error_count'] or 0) / total * 100, 2) if total > 0 else 0
                }

            return stats

    def get_query_enhancement_comparison(self) -> Dict[str, Dict]:
        """Get comparison stats for query enhancement on vs off."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    CASE WHEN query_enhanced = 1 THEN 'With QE' ELSE 'Without QE' END as qe_status,
                    COUNT(*) as total_queries,
                    AVG(latency) as avg_latency,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(relevance) as avg_relevance,
                    SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) as error_count
                FROM metrics
                GROUP BY query_enhanced
            """)

            stats = {}
            for row in cursor.fetchall():
                qe_status = row['qe_status']
                total = row['total_queries']
                stats[qe_status] = {
                    'total_queries': total,
                    'avg_latency': round(row['avg_latency'] or 0, 3),
                    'avg_faithfulness': round(row['avg_faithfulness'] or 0, 3),
                    'avg_relevance': round(row['avg_relevance'] or 0, 3),
                    'avg_accuracy': round(((row['avg_faithfulness'] or 0) + (row['avg_relevance'] or 0)) / 2, 3),
                    'error_rate': round((row['error_count'] or 0) / total * 100, 2) if total > 0 else 0
                }

            return stats
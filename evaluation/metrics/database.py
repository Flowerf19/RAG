"""
Metrics Database Manager
Handles SQLite database operations for storing and retrieving evaluation metrics.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


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
                    recall REAL,
                    error BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            # Ground truth Q&A table for benchmarking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ground_truth_qa (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            # Migration: add missing columns if table existed with older schema
            cursor = conn.execute("PRAGMA table_info('ground_truth_qa')")
            existing_cols = [r[1] for r in cursor.fetchall()]
            # Ensure 'answer', 'source', 'created_at' exist
            try:
                if 'answer' not in existing_cols:
                    conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN answer TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                if 'source' not in existing_cols:
                    conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN source TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                if 'created_at' not in existing_cols:
                    conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN created_at TEXT DEFAULT (datetime('now'))")
            except sqlite3.OperationalError:
                pass
            # Add columns to store RAG evaluation results
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN predicted_answer TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN retrieved_context TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN retrieved_sources TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN retrieval_chunks INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN evaluated_at TEXT")
            except sqlite3.OperationalError:
                pass
            # Additional evaluation metric columns (recall / token-overlap / scores)
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN retrieval_recall_at_k INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN answer_token_recall REAL")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN answer_token_precision REAL")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN answer_f1 REAL")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN rouge_l REAL")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN faithfulness REAL")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE ground_truth_qa ADD COLUMN relevance REAL")
            except sqlite3.OperationalError:
                pass
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
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN embedding_tokens INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN reranking_tokens INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN llm_tokens INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN total_tokens INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN retrieval_chunks INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE metrics ADD COLUMN recall REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()

    # --- Ground truth methods ---
    def insert_ground_truth(self, question: str, answer: str, source: Optional[str] = None) -> int:
        """Insert a ground truth QA pair."""
        with sqlite3.connect(self.db_path) as conn:
            # Discover actual column names to remain backward-compatible
            cur = conn.execute("PRAGMA table_info('ground_truth_qa')")
            cols = [r[1] for r in cur.fetchall()]

            q_col = 'question' if 'question' in cols else ('ground_truth_question' if 'ground_truth_question' in cols else None)
            a_col = 'answer' if 'answer' in cols else ('ground_truth_answer' if 'ground_truth_answer' in cols else None)
            s_col = 'source' if 'source' in cols else None

            if not q_col or not a_col:
                raise RuntimeError('ground_truth_qa table is missing expected question/answer columns')

            # Build insert statement dynamically
            insert_cols = [q_col, a_col]
            placeholders = ['?', '?']
            params = [question, answer]
            if s_col:
                insert_cols.append(s_col)
                placeholders.append('?')
                params.append(source)

            sql = f"INSERT INTO ground_truth_qa ({', '.join(insert_cols)}) VALUES ({', '.join(placeholders)})"
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.lastrowid

    def get_ground_truth(self, limit: int = 1000) -> List[Dict]:
        """Retrieve ground truth QA pairs."""
        query = "SELECT * FROM ground_truth_qa ORDER BY id ASC LIMIT ?"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (limit,))
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # Ground truth helpers (bulk operations)
    def insert_ground_truth_rows(self, rows: List[Dict]) -> int:
        """Insert multiple ground-truth QA rows. Returns number inserted."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            count = 0
            for r in rows:
                cur.execute("""
                    INSERT INTO ground_truth_qa (question, answer, source)
                    VALUES (?, ?, ?)
                """, (r.get('question'), r.get('answer'), r.get('source')))
                count += 1
            conn.commit()
            return count

    def get_ground_truth_list(self, limit: int = 1000) -> List[Dict]:
        """Return ground truth QA entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM ground_truth_qa ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
        return [dict(r) for r in rows]

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

    def insert_metric(self,
                      query: str = None,
                      model: str = None,
                      embedder_model: Optional[str] = None,
                      llm_model: Optional[str] = None,
                      reranker_model: Optional[str] = None,
                      query_enhanced: bool = False,
                      latency: Optional[float] = None,
                      faithfulness: Optional[float] = None,
                      relevance: Optional[float] = None,
                      recall: Optional[float] = None,
                      error: bool = False,
                      error_message: Optional[str] = None,
                      metadata: Optional[str] = None,
                      total_tokens: int = 0,
                      retrieval_chunks: int = 0,
                      timestamp: str = None,
                      embedding_tokens: int = 0,
                      reranking_tokens: int = 0,
                      llm_tokens: int = 0,
                      metric_dict: Optional[Dict[str, Any]] = None) -> int:
        """Insert a new metric record. Can accept individual parameters or a metric dictionary."""

        # If metric_dict is provided, use it (for backward compatibility with EvaluationLogger)
        if metric_dict is not None:
            data = metric_dict.copy()
            # Convert metadata dict to JSON string if needed
            if isinstance(data.get('metadata'), dict):
                data['metadata'] = json.dumps(data['metadata'])
        else:
            # Use individual parameters
            data = {
                'timestamp': timestamp or datetime.utcnow().isoformat(),
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
                'metadata': metadata,
                'total_tokens': total_tokens,
                'retrieval_chunks': retrieval_chunks,
                'embedding_tokens': embedding_tokens,
                'reranking_tokens': reranking_tokens,
                'llm_tokens': llm_tokens
            }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO metrics (
                    timestamp, query, model, embedder_model, llm_model, reranker_model,
                    query_enhanced, latency, faithfulness, relevance, recall, error,
                    error_message, metadata, total_tokens, retrieval_chunks, embedding_tokens,
                    reranking_tokens, llm_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['timestamp'],
                data['query'],
                data['model'],
                data['embedder_model'],
                data['llm_model'],
                data['reranker_model'],
                data['query_enhanced'],
                data['latency'],
                data['faithfulness'],
                data['relevance'],
                data['recall'],
                data['error'],
                data['error_message'],
                data['metadata'],
                data['total_tokens'],
                data['retrieval_chunks'],
                data['embedding_tokens'],
                data['reranking_tokens'],
                data['llm_tokens']
            ))
            conn.commit()
            return cursor.lastrowid

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
                    AVG(recall) as avg_recall,
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
                    'avg_recall': round(row['avg_recall'] or 0, 3),
                    'avg_accuracy': round(((row['avg_faithfulness'] or 0) + (row['avg_relevance'] or 0) + (row['avg_recall'] or 0)) / 3, 3),
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
                    AVG(recall) as avg_recall,
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
                    'avg_recall': round(row['avg_recall'] or 0, 3),
                    'avg_accuracy': round(((row['avg_faithfulness'] or 0) + (row['avg_relevance'] or 0) + (row['avg_recall'] or 0)) / 3, 3),
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
                    AVG(recall) as avg_recall,
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
                    'avg_recall': round(row['avg_recall'] or 0, 3),
                    'avg_accuracy': round(((row['avg_faithfulness'] or 0) + (row['avg_relevance'] or 0) + (row['avg_recall'] or 0)) / 3, 3),
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

    def get_token_usage_stats(self) -> Dict[str, Dict]:
        """Get aggregated token usage statistics by component."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    SUM(embedding_tokens) as total_embedding_tokens,
                    SUM(reranking_tokens) as total_reranking_tokens,
                    SUM(llm_tokens) as total_llm_tokens,
                    SUM(total_tokens) as total_tokens,
                    AVG(embedding_tokens) as avg_embedding_tokens,
                    AVG(reranking_tokens) as avg_reranking_tokens,
                    AVG(llm_tokens) as avg_llm_tokens,
                    AVG(total_tokens) as avg_total_tokens
                FROM metrics
                WHERE total_tokens > 0
            """)

            row = cursor.fetchone()
            if row:
                total_queries = row['total_queries'] or 0
                return {
                    'total_queries': total_queries,
                    'total_embedding_tokens': row['total_embedding_tokens'] or 0,
                    'total_reranking_tokens': row['total_reranking_tokens'] or 0,
                    'total_llm_tokens': row['total_llm_tokens'] or 0,
                    'total_tokens': row['total_tokens'] or 0,
                    'avg_embedding_tokens': round(row['avg_embedding_tokens'] or 0, 2),
                    'avg_reranking_tokens': round(row['avg_reranking_tokens'] or 0, 2),
                    'avg_llm_tokens': round(row['avg_llm_tokens'] or 0, 2),
                    'avg_total_tokens': round(row['avg_total_tokens'] or 0, 2)
                }
            return {
                'total_queries': 0,
                'total_embedding_tokens': 0,
                'total_reranking_tokens': 0,
                'total_llm_tokens': 0,
                'total_tokens': 0,
                'avg_embedding_tokens': 0.0,
                'avg_reranking_tokens': 0.0,
                'avg_llm_tokens': 0.0,
                'avg_total_tokens': 0.0
            }

    def get_token_usage_by_model(self, model_type: str) -> Dict[str, Dict]:
        """Get token usage statistics by specific model type (embedder, llm, reranker)."""
        column_map = {
            'embedder': 'embedder_model',
            'llm': 'llm_model',
            'reranker': 'reranker_model'
        }

        if model_type not in column_map:
            return {}

        model_column = column_map[model_type]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT
                    {model_column} as model_name,
                    COUNT(*) as total_queries,
                    SUM(embedding_tokens) as total_embedding_tokens,
                    SUM(reranking_tokens) as total_reranking_tokens,
                    SUM(llm_tokens) as total_llm_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(retrieval_chunks) as total_retrieval_chunks,
                    AVG(embedding_tokens) as avg_embedding_tokens,
                    AVG(reranking_tokens) as avg_reranking_tokens,
                    AVG(llm_tokens) as avg_llm_tokens,
                    AVG(total_tokens) as avg_total_tokens,
                    AVG(retrieval_chunks) as avg_retrieval_chunks
                FROM metrics
                WHERE {model_column} IS NOT NULL AND total_tokens > 0
                GROUP BY {model_column}
            """)

            stats = {}
            for row in cursor.fetchall():
                model_name = row['model_name']
                total_queries = row['total_queries'] or 0
                stats[model_name] = {
                    'total_queries': total_queries,
                    'total_embedding_tokens': row['total_embedding_tokens'] or 0,
                    'total_reranking_tokens': row['total_reranking_tokens'] or 0,
                    'total_llm_tokens': row['total_llm_tokens'] or 0,
                    'total_tokens': row['total_tokens'] or 0,
                    'total_retrieval_chunks': row['total_retrieval_chunks'] or 0,
                    'avg_embedding_tokens': round(row['avg_embedding_tokens'] or 0, 2),
                    'avg_reranking_tokens': round(row['avg_reranking_tokens'] or 0, 2),
                    'avg_llm_tokens': round(row['avg_llm_tokens'] or 0, 2),
                    'avg_total_tokens': round(row['avg_total_tokens'] or 0, 2),
                    'avg_retrieval_chunks': round(row['avg_retrieval_chunks'] or 0, 2)
                }

            return stats

    def get_token_usage_over_time(self, hours: int = 24) -> List[Dict]:
        """Get token usage data over time for time series visualization."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    SUM(embedding_tokens) as embedding_tokens,
                    SUM(reranking_tokens) as reranking_tokens,
                    SUM(llm_tokens) as llm_tokens,
                    SUM(total_tokens) as total_tokens,
                    COUNT(*) as query_count
                FROM metrics
                WHERE timestamp >= datetime('now', '-{} hours')
                GROUP BY hour
                ORDER BY hour
            """.format(hours))

            return [
                {
                    'timestamp': row['hour'],
                    'embedding_tokens': row['embedding_tokens'] or 0,
                    'reranking_tokens': row['reranking_tokens'] or 0,
                    'llm_tokens': row['llm_tokens'] or 0,
                    'total_tokens': row['total_tokens'] or 0,
                    'query_count': row['query_count'] or 0
                }
                for row in cursor.fetchall()
            ]

    def _get_retrieval_chunks_stats(self) -> Dict[str, Any]:
        """Get retrieval chunks statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    SUM(retrieval_chunks) as total_retrieval_chunks,
                    AVG(retrieval_chunks) as avg_retrieval_chunks
                FROM metrics
                WHERE retrieval_chunks > 0
            """)

            row = cursor.fetchone()
            if row:
                return {
                    'total_retrieval_chunks': row['total_retrieval_chunks'] or 0,
                    'avg_retrieval_chunks': round(row['avg_retrieval_chunks'] or 0, 2)
                }
            return {
                'total_retrieval_chunks': 0,
                'avg_retrieval_chunks': 0.0
            }

    def update_ground_truth_result(
        self,
        gt_id: int,
        predicted_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        retrieved_sources: Optional[str] = None,
        retrieval_chunks: int = 0,
        evaluated_at: Optional[str] = None,
        retrieval_recall_at_k: int | None = None,
        answer_token_recall: float | None = None,
        answer_token_precision: float | None = None,
        answer_f1: float | None = None,
        rouge_l: float | None = None,
        faithfulness: float | None = None,
        relevance: float | None = None,
    ) -> None:
        """Update a ground_truth_qa row with RAG evaluation results."""
        if evaluated_at is None:
            evaluated_at = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Build dynamic update to be tolerant to missing new columns
            update_cols = [
                ('retrieval_chunks', retrieval_chunks),
                ('evaluated_at', evaluated_at),
            ]

            # Optional fields
            if predicted_answer is not None:
                update_cols.append(('predicted_answer', predicted_answer))
            if retrieved_context is not None:
                update_cols.append(('retrieved_context', retrieved_context))
            if retrieved_sources is not None:
                update_cols.append(('retrieved_sources', retrieved_sources))

            # Optional metric fields
            if retrieval_recall_at_k is not None:
                update_cols.append(('retrieval_recall_at_k', int(retrieval_recall_at_k)))
            if answer_token_recall is not None:
                update_cols.append(('answer_token_recall', float(answer_token_recall)))
            if answer_token_precision is not None:
                update_cols.append(('answer_token_precision', float(answer_token_precision)))
            if answer_f1 is not None:
                update_cols.append(('answer_f1', float(answer_f1)))
            if rouge_l is not None:
                update_cols.append(('rouge_l', float(rouge_l)))
            if faithfulness is not None:
                update_cols.append(('faithfulness', float(faithfulness)))
            if relevance is not None:
                update_cols.append(('relevance', float(relevance)))

            set_clause = ", ".join([f"{col} = ?" for col, _ in update_cols])
            params = [val for _, val in update_cols]
            params.append(gt_id)

            sql = f"UPDATE ground_truth_qa SET {set_clause} WHERE id = ?"
            conn.execute(sql, params)
            conn.commit()
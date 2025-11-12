"""
RAG Evaluation Dashboard
Streamlit app for visualizing and comparing RAG model performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.express as px

from evaluation.backend_dashboard.api import BackendDashboard


class RAGEvaluationDashboard:
    """Streamlit dashboard for RAG model evaluation and comparison."""

    def __init__(self):
        """Initialize dashboard with backend API."""
        self.backend = BackendDashboard()
        st.set_page_config(
            page_title="RAG Evaluation Dashboard",
            page_icon="📊",
            layout="wide"
        )

    def run(self):
        """Run the dashboard application."""
        st.title("🚀 RAG Evaluation Dashboard")
        st.markdown("Compare performance and accuracy across different RAG models")

        # Add refresh button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("🔄 Refresh Data"):
                self.backend.refresh_data()
                st.rerun()

        with col2:
            # model_filter = st.selectbox(
            #     "Filter by Model",
            #     ["All Models", "GPT-4", "GPT-3.5", "Claude", "Mistral", "DeepSeek", "Local Models"],
            #     key="model_filter"
            # )
            st.empty()  # Placeholder for future model filter

        # Overview Statistics
        self._display_overview_stats()

        # Model Comparison Tables
        self._display_model_comparison()

        # Performance Charts
        self._display_performance_charts()

        # Recent Activity
        self._display_recent_activity()

    def _display_overview_stats(self):
        """Display overview statistics in header."""
        st.header("📈 Overview Statistics")

        stats = self.backend.get_overview_stats()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Queries", f"{stats['total_queries']:,}")

        with col2:
            st.metric("Avg Accuracy", f"{stats['avg_accuracy']:.3f}")

        with col3:
            st.metric("Avg Latency", f"{stats['avg_latency']:.3f}s")

        with col4:
            st.metric("Error Rate", f"{stats['error_rate']:.1f}%")

        with col5:
            st.metric("Models Tested", stats['model_count'])

    def _display_model_comparison(self):
        """Display comparison tables for different model types."""
        st.header("🔍 Model Comparison")

        comparison_data = self.backend.get_model_comparison_data()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🤖 LLM Models", "🔗 Embedding Models", "📊 Reranking Models", "✨ Query Enhancement", "💰 Token Usage", "📥 Ground Truth"])

        with tab1:
            if comparison_data['llm']:
                df = pd.DataFrame(comparison_data['llm'])
                st.dataframe(df, width='stretch')

                # Add accuracy chart for LLM models
                llm_accuracy = self.backend.get_llm_accuracy()
                if llm_accuracy:
                    st.subheader("🎯 LLM Model Accuracy")
                    df_acc = pd.DataFrame(llm_accuracy)
                    fig = px.bar(df_acc, x='model', y='accuracy',
                               title='LLM Model Accuracy Comparison',
                               labels={'accuracy': 'Accuracy Score', 'model': 'LLM Model'},
                               color='accuracy',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No LLM model data available")

        with tab2:
            if comparison_data['embedding']:
                df = pd.DataFrame(comparison_data['embedding'])
                st.dataframe(df, width='stretch')

                # Add accuracy chart for embedding models
                embedder_accuracy = self.backend.get_embedder_accuracy()
                if embedder_accuracy:
                    st.subheader("🎯 Embedding Model Accuracy")
                    df_acc = pd.DataFrame(embedder_accuracy)
                    fig = px.bar(df_acc, x='model', y='accuracy',
                               title='Embedding Model Accuracy Comparison',
                               labels={'accuracy': 'Accuracy Score', 'model': 'Embedding Model'},
                               color='accuracy',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No embedding model data available")

        with tab3:
            if comparison_data['reranking']:
                df = pd.DataFrame(comparison_data['reranking'])
                st.dataframe(df, width='stretch')

                # Add accuracy chart for reranking models
                reranker_accuracy = self.backend.get_reranker_accuracy()
                if reranker_accuracy:
                    st.subheader("🎯 Reranking Model Accuracy")
                    df_acc = pd.DataFrame(reranker_accuracy)
                    fig = px.bar(df_acc, x='model', y='accuracy',
                               title='Reranking Model Accuracy Comparison',
                               labels={'accuracy': 'Accuracy Score', 'model': 'Reranking Model'},
                               color='accuracy',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No reranking model data available")

        with tab4:
            if comparison_data.get('query_enhancement'):
                df = pd.DataFrame(comparison_data['query_enhancement'])
                st.dataframe(df, width='stretch')

                # Add accuracy chart for query enhancement comparison
                qe_accuracy = self.backend.get_query_enhancement_accuracy()
                if qe_accuracy:
                    st.subheader("🎯 Query Enhancement Accuracy")
                    df_acc = pd.DataFrame(qe_accuracy)
                    fig = px.bar(df_acc, x='model', y='accuracy',
                               title='Query Enhancement Accuracy Comparison',
                               labels={'accuracy': 'Accuracy Score', 'model': 'Query Enhancement'},
                               color='accuracy',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No query enhancement data available")

        with tab5:
            self._display_token_usage()

        with tab6:
            st.header("📥 Import Ground-truth Q&A")
            st.markdown("Upload an Excel (.xlsx/.xls) or CSV file with columns: `STT`, `Câu hỏi`, `Câu trả lời`, `Nguồn`.")

            # Controls: select embedder, reranker, QEM toggle, sample size
            embedder_choice = st.selectbox(
                "Embedder Model",
                ["ollama", "huggingface_local", "huggingface_api", "e5_large_instruct", "e5_base", "gte_multilingual_base", "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"],
                index=0,
                help="Choose which embedder to use for evaluation (also used for retrieval if supported)."
            )

            reranker_choice = st.selectbox(
                "Reranker",
                ["none", "bge_m3_ollama", "bge_m3_hf_api", "bge_m3_hf_local", "jina_v2_multilingual", "gte_multilingual", "bge_base"],
                index=0,
                help="Choose reranker used during retrieval (if any)."
            )

            use_qem = st.checkbox("Use Query Enhancement (QEM)", value=True)

            sample_size = st.number_input("Max rows to evaluate (0 = all)", value=10, min_value=0, step=1)

            uploaded = st.file_uploader("Choose ground-truth file", type=["xlsx", "xls", "csv"], key="gt_upload")

            def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
                # Defensive parsing: drop empty rows and handle files where header is missing
                df = df.copy()
                # Replace literal 'nan' strings coming from Excel with actual NaN
                df = df.replace({'nan': pd.NA, 'NaN': pd.NA, 'NONE': pd.NA, 'None': pd.NA})
                df = df.dropna(how='all')

                # Robust mapping: drop empty cols/rows, promote first row to header if needed,
                # then heuristically pick question/answer/source columns with sensible fallbacks.
                # 1) Drop fully empty columns
                df = df.dropna(axis=1, how='all')

                # 2) If columns look like 'Unnamed' or empty, and the first row contains header-like values,
                #    promote first row to header (common for poorly exported Excel files)
                cols = [str(c) for c in df.columns]
                if any(c.lower().startswith('unnamed') or c.strip() == '' for c in cols):
                    first_row = df.iloc[0].astype(str).tolist()
                    first_row_lc = [str(x).lower() for x in first_row]
                    header_keywords = ['câu', 'question', 'trả', 'answer', 'nguồn', 'source', 'stt']
                    if any(any(k in cell for k in header_keywords) for cell in first_row_lc):
                        # Promote
                        df = df[1:].copy()
                        df.columns = [str(x).strip() for x in first_row]

                # 3) Drop rows that are entirely empty
                df = df.dropna(axis=0, how='all')

                # Helper to pick a column by candidate substrings, with fallback index
                def pick_col(df_local: pd.DataFrame, candidates: list, fallback_idx: int | None = None):
                    for c in df_local.columns:
                        lc = str(c).lower()
                        for cand in candidates:
                            if cand in lc:
                                return c
                    if fallback_idx is not None and 0 <= fallback_idx < len(df_local.columns):
                        return df_local.columns[fallback_idx]
                    return None

                # 4) Identify columns
                q_col = pick_col(df, ['câu', 'cau', 'question', 'câu hỏi', 'question)'], fallback_idx=1 if df.shape[1] > 1 else 0)
                a_col = pick_col(df, ['trả', 'tra', 'answer', 'câu trả', 'answer)'], fallback_idx=2 if df.shape[1] > 2 else (-1 if df.shape[1] > 0 else None))
                s_col = pick_col(df, ['nguồn', 'nguon', 'source'], fallback_idx=3 if df.shape[1] > 3 else None)

                # If a_col fallback returned -1 (use last column)
                if a_col is None and df.shape[1] > 0:
                    # attempt assign second non-STT column
                    non_stt = [c for c in df.columns if 'stt' not in str(c).lower()]
                    if len(non_stt) >= 2:
                        a_col = non_stt[1]
                    elif len(non_stt) == 1:
                        a_col = non_stt[0]

                # If q_col is still None, pick the first non-STT column
                if q_col is None:
                    non_stt = [c for c in df.columns if 'stt' not in str(c).lower()]
                    q_col = non_stt[0] if non_stt else df.columns[0]

                # If s_col is None, set empty source later

                # 5) Build normalized DataFrame safely
                normalized = pd.DataFrame()
                normalized['question'] = df[q_col].astype(str).fillna('').str.strip()

                if a_col is not None:
                    normalized['answer'] = df[a_col].astype(str).fillna('').str.strip()
                else:
                    # fallback: try third column or last column
                    if df.shape[1] > 2:
                        normalized['answer'] = df.iloc[:, 2].astype(str).fillna('').str.strip()
                    else:
                        normalized['answer'] = df.iloc[:, -1].astype(str).fillna('').str.strip()

                if s_col is not None:
                    normalized['source'] = df[s_col].astype(str).fillna('').str.strip()
                else:
                    normalized['source'] = ''

                # 6) Clean up some common artifacts
                # Remove rows where question is empty or equals header labels like 'stt'
                normalized['question_norm'] = normalized['question'].str.lower().str.strip()
                normalized = normalized[~normalized['question_norm'].isin(['', 'stt', 'số thứ tự'])]
                normalized = normalized.drop(columns=['question_norm'])

                # Reset index
                normalized = normalized.reset_index(drop=True)
                return normalized
                # If the DataFrame has 'Unnamed' headers or numeric headers, and the first row looks like header labels,
                # promote the first row to header. This handles files where the real header was written in the first row.
                cols_are_generic = all((str(c).lower().startswith('unnamed') or str(c).strip().isdigit()) for c in df.columns)
                first_row_vals = []
                try:
                    first_row_vals = [str(x).lower() for x in df.iloc[0].fillna('')]
                except Exception:
                    first_row_vals = []

                header_like = any(k in ' '.join(first_row_vals) for k in ['stt', 'câu', 'cau', 'question', 'answer', 'nguồn', 'nguon', 'source'])

                if cols_are_generic and header_like:
                    # promote
                    df.columns = df.iloc[0].astype(str)
                    df = df.iloc[1:].reset_index(drop=True)

                # Normalize column names list for detection
                lc = [str(c).lower() for c in df.columns]

                q_col = None
                a_col = None
                s_col = None

                for i, name in enumerate(lc):
                    if 'câu' in name or 'question' in name or 'cau' in name or 'question' == name.strip():
                        q_col = df.columns[i]
                    if 'trả' in name or 'tra' in name or 'answer' in name or 'ans' in name:
                        a_col = df.columns[i]
                    if 'nguồn' in name or 'nguon' in name or 'source' in name:
                        s_col = df.columns[i]

                # Fallback heuristics: pick the most text-heavy columns
                if q_col is None or a_col is None:
                    # compute text density (non-null count and average string length)
                    stats = []
                    for c in df.columns:
                        col_vals = df[c].dropna().astype(str)
                        cnt = len(col_vals)
                        avg_len = col_vals.map(len).mean() if cnt > 0 else 0
                        stats.append((c, cnt, avg_len))

                    # sort by cnt then avg_len
                    stats_sorted = sorted(stats, key=lambda x: (x[1], x[2]), reverse=True)
                    if q_col is None and stats_sorted:
                        q_col = stats_sorted[0][0]
                    if a_col is None and len(stats_sorted) > 1:
                        # prefer the second-best column for answer
                        a_col = stats_sorted[1][0]

                # Build normalized DataFrame
                normalized = pd.DataFrame()
                # If still None, defensively choose columns
                if q_col is not None:
                    normalized['question'] = df[q_col].astype(str).fillna('')
                else:
                    normalized['question'] = df.iloc[:, 0].astype(str).fillna('')

                if a_col is not None and a_col in df.columns:
                    normalized['answer'] = df[a_col].astype(str).fillna('')
                else:
                    if df.shape[1] > 1:
                        normalized['answer'] = df.iloc[:, 1].astype(str).fillna('')
                    else:
                        normalized['answer'] = ''

                if s_col is not None and s_col in df.columns:
                    normalized['source'] = df[s_col].astype(str).fillna('')
                else:
                    # If no source column, set to empty string
                    normalized['source'] = ''

                # Final cleanup: replace the string 'nan' with empty
                normalized = normalized.replace({'nan': '', 'None': '', 'NoneType': ''})

                return normalized

            if uploaded is not None:
                try:
                    if uploaded.name.lower().endswith('.csv'):
                        df = pd.read_csv(uploaded)
                    else:
                        df = pd.read_excel(uploaded)

                    st.write(f"Detected {df.shape[0]} rows in uploaded file")

                    normalized = _map_columns(df)

                    st.subheader("Preview parsed ground-truth (first 10 rows)")
                    st.dataframe(normalized.head(10))

                    if st.button("Import to DB", key="import_gt"):
                        # Prepare rows list
                        rows = []
                        for _, r in normalized.iterrows():
                            rows.append({
                                'question': r.get('question', ''),
                                'answer': r.get('answer', ''),
                                'source': r.get('source', '')
                            })

                        try:
                            inserted = self.backend.insert_ground_truth_rows(rows)
                            st.success(f"Inserted {inserted} ground-truth rows into DB")
                        except Exception as e:
                            st.error(f"Failed to insert ground-truth into DB: {e}")

                    # Button to run RAG on all ground-truth rows (or sample)
                    if st.button("Run RAG on ground-truth", key="run_rag_gt"):
                        with st.spinner("Running retrieval for ground-truth rows (may take a while)..."):
                            try:
                                max_rows = None if sample_size == 0 else int(sample_size)
                                res = self.backend.evaluate_ground_truth(
                                    embedder_type=embedder_choice,
                                    reranker_type=reranker_choice,
                                    use_query_enhancement=use_qem,
                                    top_k=5,
                                    limit=max_rows,
                                )
                                st.success(f"RAG run completed: {res['processed']} processed, {res['errors']} errors")
                                if res.get('error_details'):
                                    st.json(res['error_details'])
                            except Exception as e:
                                st.error(f"RAG evaluation failed: {e}")

                    # New button: run 5 ground-truth rows sequentially and stream logs to UI
                    if st.button("Run 5 GT (with logs)", key="run_5_gt"):
                        # Create a live log area
                        logs_box = st.empty()
                        logs = []

                        try:
                            # Fetch up to 5 ground-truth rows from DB
                            rows = self.backend.get_ground_truth_list(limit=5)

                            if not rows:
                                st.info("No ground-truth rows available to run.")
                            else:
                                from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval

                                for idx, row in enumerate(rows, start=1):
                                    gt_id = row.get('id')
                                    question = row.get('question')
                                    logs.append(f"=== Running {idx}/{len(rows)} | id={gt_id} ===")
                                    logs.append(f"Question: {question}")
                                    logs_box.code("\n".join(logs), language='text')

                                    try:
                                        # Call the same retrieval method used by backend but do it here so we can stream logs
                                        result = fetch_retrieval(
                                            query_text=question,
                                            top_k=5,
                                            embedder_type=embedder_choice,
                                            reranker_type=reranker_choice,
                                            use_query_enhancement=use_qem,
                                        )

                                        context = result.get('context', '')
                                        sources = result.get('sources', [])
                                        retrieval_info = result.get('retrieval_info', {}) or {}
                                        total_retrieved = retrieval_info.get('total_retrieved', retrieval_info.get('final_count', 'unknown'))

                                        logs.append(f"Retrieved chunks: {total_retrieved}")
                                        # Truncate context for the UI
                                        logs.append(f"Context (truncated 1000 chars): {context[:1000]}")
                                        logs.append(f"Sources: {sources}")
                                        logs.append("")
                                        logs_box.code("\n".join(logs), language='text')

                                    except Exception as e:
                                        logs.append(f"Error processing id={gt_id}: {e}")
                                        logs_box.code("\n".join(logs), language='text')

                                st.success(f"Finished running {len(rows)} ground-truth queries. See log above.")

                        except Exception as e:
                            st.error(f"Failed to run sample ground-truth queries: {e}")

                except Exception as e:
                    st.error(f"Failed to read uploaded file: {e}")

    def _display_performance_charts(self):
        """Display performance visualization charts."""
        st.header("📊 Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⏱️ Latency Over Time")
            latency_data = self.backend.get_latency_over_time()
            if latency_data:
                df = pd.DataFrame(latency_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                fig = px.line(df, x='timestamp', y='avg_latency',
                            title='Average Latency Trend',
                            labels={'avg_latency': 'Latency (s)', 'timestamp': 'Time'})
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No latency data available")

        with col2:
            st.info("Model-specific accuracy charts are now available in the Model Comparison tabs above.")

    def _display_recent_activity(self):
        """Display recent evaluation activity."""
        st.header("📋 Recent Activity")

        recent_metrics = self.backend.get_recent_metrics(limit=20)

        if recent_metrics:
            # Convert to DataFrame for display
            df = pd.DataFrame(recent_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Create detailed model description
            def format_model_info(row):
                llm = row.get('llm_model', 'Unknown')
                embedder = row.get('embedder_model', 'Unknown')
                reranker = row.get('reranker_model', 'none')
                qe = "QE" if row.get('query_enhanced', False) else "NoQE"
                return f"{llm} + {embedder} + {reranker} + {qe}"

            df['model_config'] = df.apply(format_model_info, axis=1)

            df['model_config'] = df.apply(format_model_info, axis=1)

            df['model_config'] = df.apply(format_model_info, axis=1)

            # Select and reorder columns
            display_cols = ['timestamp', 'model_config', 'query', 'latency', 'faithfulness', 'relevance', 'error']
            df_display = df[display_cols].head(10)

            st.dataframe(df_display, width='stretch')

            # Show summary stats
            total_recent = len(recent_metrics)
            error_recent = sum(1 for m in recent_metrics if m.get('error', False))
            avg_latency_recent = sum(m.get('latency', 0) for m in recent_metrics) / total_recent if total_recent > 0 else 0

            st.caption(f"Showing last 10 of {total_recent} recent evaluations | "
                      f"Avg Latency: {avg_latency_recent:.3f}s | Error Rate: {error_recent/total_recent*100:.1f}%")
        else:
            st.info("No recent evaluation data available")

    def _display_token_usage(self):
        """Display token usage statistics and visualizations."""
        st.header("💰 Token Usage Analytics")

        # Get token usage data
        token_overview = self.backend.get_token_usage_overview()
        token_by_embedder = self.backend.get_token_usage_by_embedder()
        token_by_llm = self.backend.get_token_usage_by_llm()
        token_by_reranker = self.backend.get_token_usage_by_reranker()
        token_over_time = self.backend.get_token_usage_over_time()
        token_costs = self.backend.get_token_costs()

        # Overview metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Total Queries", f"{token_overview['total_queries']:,}")

        with col2:
            st.metric("Total Tokens", f"{token_overview['total_tokens']:,}")

        with col3:
            st.metric("Embedding Tokens", f"{token_overview['total_embedding_tokens']:,}")

        with col4:
            st.metric("LLM Tokens", f"{token_overview['total_llm_tokens']:,}")

        with col5:
            st.metric("Retrieval Chunks", f"{token_overview['total_retrieval_chunks']:,}")

        # Cost estimates
        st.subheader("💵 Estimated Costs")
        cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)

        with cost_col1:
            st.metric("Embedding Cost", f"${token_costs['embedding_cost']:.4f}")

        with cost_col2:
            st.metric("Reranking Cost", f"${token_costs['reranking_cost']:.4f}")

        with cost_col3:
            st.metric("LLM Cost", f"${token_costs['llm_cost']:.4f}")

        with cost_col4:
            st.metric("Total Cost", f"${token_costs['total_cost']:.4f}")

        st.caption(f"Cost per query: ${token_costs['cost_per_query']:.6f}")

        # Token usage by component tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔗 By Embedder", "🤖 By LLM", "📊 By Reranker", "📈 Over Time"])

        with tab1:
            if token_by_embedder:
                df = pd.DataFrame(token_by_embedder)
                st.dataframe(df, width='stretch')

                # Token usage chart for embedders
                st.subheader("🔗 Token Usage by Embedder Model")
                fig = px.bar(df, x='model', y='total_tokens',
                           title='Total Tokens Used by Embedder',
                           labels={'total_tokens': 'Total Tokens', 'model': 'Embedder Model'},
                           color='total_tokens',
                           color_continuous_scale='blues')
                st.plotly_chart(fig, use_container_width=True)

                # Average tokens per query
                st.subheader("📊 Average Tokens per Query by Embedder")
                fig = px.bar(df, x='model', y='avg_total_tokens',
                           title='Average Tokens per Query by Embedder',
                           labels={'avg_total_tokens': 'Avg Tokens/Query', 'model': 'Embedder Model'},
                           color='avg_total_tokens',
                           color_continuous_scale='greens')
                st.plotly_chart(fig, use_container_width=True)

                # Retrieval chunks by embedder
                st.subheader("📦 Retrieval Chunks by Embedder")
                fig = px.bar(df, x='model', y='total_retrieval_chunks',
                           title='Total Retrieval Chunks by Embedder',
                           labels={'total_retrieval_chunks': 'Total Chunks', 'model': 'Embedder Model'},
                           color='total_retrieval_chunks',
                           color_continuous_scale='purples')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No embedder token usage data available")

        with tab2:
            if token_by_llm:
                df = pd.DataFrame(token_by_llm)
                st.dataframe(df, width='stretch')

                # Token usage chart for LLMs
                st.subheader("🤖 Token Usage by LLM Model")
                fig = px.bar(df, x='model', y='total_llm_tokens',
                           title='LLM Tokens Used by Model',
                           labels={'total_llm_tokens': 'LLM Tokens', 'model': 'LLM Model'},
                           color='total_llm_tokens',
                           color_continuous_scale='reds')
                st.plotly_chart(fig, use_container_width=True)

                # Average tokens per query
                st.subheader("📊 Average LLM Tokens per Query")
                fig = px.bar(df, x='model', y='avg_llm_tokens',
                           title='Average LLM Tokens per Query by Model',
                           labels={'avg_llm_tokens': 'Avg LLM Tokens/Query', 'model': 'LLM Model'},
                           color='avg_llm_tokens',
                           color_continuous_scale='oranges')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No LLM token usage data available")

        with tab3:
            if token_by_reranker:
                df = pd.DataFrame(token_by_reranker)
                st.dataframe(df, width='stretch')

                # Token usage chart for rerankers
                st.subheader("📊 Token Usage by Reranker Model")
                fig = px.bar(df, x='model', y='total_reranking_tokens',
                           title='Reranking Tokens Used by Model',
                           labels={'total_reranking_tokens': 'Reranking Tokens', 'model': 'Reranker Model'},
                           color='total_reranking_tokens',
                           color_continuous_scale='purples')
                st.plotly_chart(fig, use_container_width=True)

                # Average tokens per query
                st.subheader("📊 Average Reranking Tokens per Query")
                fig = px.bar(df, x='model', y='avg_reranking_tokens',
                           title='Average Reranking Tokens per Query by Model',
                           labels={'avg_reranking_tokens': 'Avg Reranking Tokens/Query', 'model': 'Reranker Model'},
                           color='avg_reranking_tokens',
                           color_continuous_scale='teals')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No reranker token usage data available")

        with tab4:
            if token_over_time:
                df = pd.DataFrame(token_over_time)
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Token usage over time
                st.subheader("📈 Token Usage Over Time")
                fig = px.line(df, x='timestamp', y=['embedding_tokens', 'reranking_tokens', 'llm_tokens', 'total_tokens'],
                            title='Token Usage Trends Over Time',
                            labels={'value': 'Tokens', 'timestamp': 'Time', 'variable': 'Token Type'},
                            color_discrete_map={
                                'embedding_tokens': 'blue',
                                'reranking_tokens': 'purple',
                                'llm_tokens': 'red',
                                'total_tokens': 'black'
                            })
                st.plotly_chart(fig, use_container_width=True)

                # Query count over time
                st.subheader("📊 Query Volume Over Time")
                fig = px.bar(df, x='timestamp', y='query_count',
                           title='Number of Queries Over Time',
                           labels={'query_count': 'Query Count', 'timestamp': 'Time'},
                           color='query_count',
                           color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time-series token usage data available")

    def _display_token_usage(self):
        """Display token usage statistics and charts."""
        st.subheader("💰 Token Usage Overview")

        # Get token usage overview
        token_overview = self.backend.get_token_usage_overview()

        if token_overview['total_queries'] == 0:
            st.info("No token usage data available yet")
            return

        # Display overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", f"{token_overview['total_queries']:,}")

        with col2:
            st.metric("Total Tokens", f"{token_overview['total_tokens']:,}")

        with col3:
            st.metric("Avg Tokens/Query", f"{token_overview['avg_total_tokens']:.1f}")

        with col4:
            # Calculate cost estimate (rough estimate)
            costs = self.backend.get_token_costs()
            st.metric("Est. Total Cost", f"${costs['total_cost']:.4f}")

        # Token breakdown by component
        st.subheader("🔍 Token Usage by Component")

        component_data = {
            'Component': ['Embedding', 'Reranking', 'LLM'],
            'Total Tokens': [
                token_overview['total_embedding_tokens'],
                token_overview['total_reranking_tokens'],
                token_overview['total_llm_tokens']
            ],
            'Avg Tokens/Query': [
                token_overview['avg_embedding_tokens'],
                token_overview['avg_reranking_tokens'],
                token_overview['avg_llm_tokens']
            ]
        }

        df_components = pd.DataFrame(component_data)
        fig = px.bar(df_components, x='Component', y='Total Tokens',
                    title='Token Usage by Component',
                    color='Component',
                    color_discrete_map={
                        'Embedding': '#1f77b4',
                        'Reranking': '#ff7f0e',
                        'LLM': '#2ca02c'
                    })
        st.plotly_chart(fig, use_container_width=True)

        # Token usage over time
        st.subheader("📈 Token Usage Over Time")
        time_data = self.backend.get_token_usage_over_time(hours=24)
        if time_data:
            df_time = pd.DataFrame(time_data)
            df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])

            # Create stacked area chart
            fig = px.area(df_time, x='timestamp',
                         y=['embedding_tokens', 'reranking_tokens', 'llm_tokens'],
                         title='Token Usage Over Time (Last 24 Hours)',
                         labels={'value': 'Tokens', 'timestamp': 'Time', 'variable': 'Component'},
                         color_discrete_map={
                             'embedding_tokens': '#1f77b4',
                             'reranking_tokens': '#ff7f0e',
                             'llm_tokens': '#2ca02c'
                         })
            fig.update_layout(yaxis_title='Tokens')
            st.plotly_chart(fig, use_container_width=True)

        # Model-specific token usage
        st.subheader("🤖 Model-Specific Token Usage")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Embedding Models**")
            embedder_tokens = self.backend.get_token_usage_by_embedder()
            if embedder_tokens:
                df_embedder = pd.DataFrame(embedder_tokens)
                df_embedder = df_embedder.sort_values('total_tokens', ascending=False)
                fig = px.bar(df_embedder.head(5), x='model', y='total_tokens',
                           title='Top Embedding Models by Token Usage',
                           labels={'total_tokens': 'Total Tokens', 'model': 'Model'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No embedder token data")

        with col2:
            st.markdown("**LLM Models**")
            llm_tokens = self.backend.get_token_usage_by_llm()
            if llm_tokens:
                df_llm = pd.DataFrame(llm_tokens)
                df_llm = df_llm.sort_values('total_tokens', ascending=False)
                fig = px.bar(df_llm.head(5), x='model', y='total_tokens',
                           title='Top LLM Models by Token Usage',
                           labels={'total_tokens': 'Total Tokens', 'model': 'Model'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No LLM token data")

        with col3:
            st.markdown("**Reranker Models**")
            reranker_tokens = self.backend.get_token_usage_by_reranker()
            if reranker_tokens:
                df_reranker = pd.DataFrame(reranker_tokens)
                df_reranker = df_reranker.sort_values('total_tokens', ascending=False)
                fig = px.bar(df_reranker.head(5), x='model', y='total_tokens',
                           title='Top Reranker Models by Token Usage',
                           labels={'total_tokens': 'Total Tokens', 'model': 'Model'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No reranker token data")

        # Cost analysis
        st.subheader("💵 Cost Analysis")
        costs = self.backend.get_token_costs()

        cost_data = {
            'Component': ['Embedding', 'Reranking', 'LLM', 'Total'],
            'Estimated Cost ($)': [
                costs['embedding_cost'],
                costs['reranking_cost'],
                costs['llm_cost'],
                costs['total_cost']
            ]
        }

        df_costs = pd.DataFrame(cost_data)
        fig = px.bar(df_costs, x='Component', y='Estimated Cost ($)',
                    title='Estimated Costs by Component',
                    color='Component',
                    color_discrete_map={
                        'Embedding': '#1f77b4',
                        'Reranking': '#ff7f0e',
                        'LLM': '#2ca02c',
                        'Total': '#d62728'
                    })
        st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 Cost estimates are based on default rates: $0.0001/1K embedding tokens, $0.001/1K reranking tokens, $0.002/1K LLM tokens. Adjust rates as needed.")


def main():
    """Main entry point for the dashboard."""
    dashboard = RAGEvaluationDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
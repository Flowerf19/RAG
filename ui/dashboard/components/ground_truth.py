"""
Ground Truth Component
Handles ground truth Q&A import and evaluation.
"""

import streamlit as st
import pandas as pd
from evaluation.backend_dashboard.api import BackendDashboard


class GroundTruthComponent:
    """Component for ground truth import and evaluation."""

    def __init__(self, backend: BackendDashboard):
        """Initialize with backend API."""
        self.backend = backend

    def display(self):
        """Display ground truth import and evaluation interface."""
        st.header("📥 Import Ground-truth Q&A")
        st.markdown("Upload an Excel (.xlsx/.xls) or CSV file with columns: `STT`, `Câu hỏi`, `Câu trả lời`, `Nguồn`.")

        # Controls: select embedder, reranker, LLM, QEM toggle, sample size
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

        llm_choice = st.selectbox(
            "LLM for Evaluation",
            ["none", "gemini", "ollama", "lmstudio", "openai"],
            index=0,
            help="Choose LLM to use for answer quality evaluation (faithfulness, relevance). 'none' skips evaluation."
        )

        use_qem = st.checkbox("Use Query Enhancement (QEM)", value=True)

        sample_size = st.number_input("Max rows to evaluate (0 = all)", value=10, min_value=0, step=1)

        save_to_db = st.checkbox("Save results to database", value=True, help="Store evaluation results for historical tracking and dashboard analytics")

        uploaded = st.file_uploader("Choose ground-truth file", type=["xlsx", "xls", "csv"], key="gt_upload")

        if uploaded is not None:
            self._handle_file_upload(uploaded, embedder_choice, reranker_choice, llm_choice, use_qem, sample_size)

    def _handle_file_upload(self, uploaded, embedder_choice, reranker_choice, llm_choice, use_qem, sample_size):
        """Handle uploaded ground truth file."""
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.write(f"Detected {df.shape[0]} rows in uploaded file")

            normalized = self._map_columns(df)

            st.subheader("Preview parsed ground-truth (first 10 rows)")
            st.dataframe(normalized.head(10))

            if st.button("Import to DB", key="import_gt"):
                self._import_to_db(normalized)

            # Button to run RAG on all ground-truth rows (or sample)
            if st.button("Run RAG on ground-truth", key="run_rag_gt"):
                self._run_rag_evaluation(embedder_choice, reranker_choice, llm_choice, use_qem, sample_size)

            # Button to run semantic similarity evaluation
            if st.button("🔍 Run Semantic Similarity Evaluation", key="run_semantic_eval"):
                self._run_semantic_similarity_evaluation(embedder_choice, reranker_choice, use_qem, sample_size, save_to_db)

            # Button to run recall evaluation
            if st.button("📊 Evaluate Recall", key="run_recall_eval"):
                self._run_recall_evaluation(embedder_choice, reranker_choice, use_qem, sample_size, save_to_db)

            # Button to run relevance evaluation
            if st.button("🎯 Evaluate Relevance", key="run_relevance_eval"):
                self._run_relevance_evaluation(embedder_choice, reranker_choice, use_qem, sample_size, save_to_db)

            # Button to run full evaluation suite (all 3 metrics)
            if st.button("🚀 Full Evaluation Suite (Ground-truth + Recall + Relevance)", key="run_full_eval"):
                self._run_full_evaluation_suite(embedder_choice, reranker_choice, use_qem, sample_size, save_to_db)

            # Button to run 5 ground-truth rows sequentially and stream logs to UI
            if st.button("Run 5 GT (with logs)", key="run_5_gt"):
                self._run_5_gt_with_logs(embedder_choice, reranker_choice, llm_choice, use_qem)

        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map and normalize columns from uploaded file."""
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

    def _import_to_db(self, normalized_df):
        """Import normalized data to database."""
        # Prepare rows list
        rows = []
        for _, r in normalized_df.iterrows():
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

    def _run_rag_evaluation(self, embedder_choice, reranker_choice, llm_choice, use_qem, sample_size):
        """Run RAG evaluation on ground truth."""
        with st.spinner("Running retrieval for ground-truth rows (may take a while)..."):
            try:
                max_rows = None if sample_size == 0 else int(sample_size)
                res = self.backend.evaluate_ground_truth(
                    embedder_type=embedder_choice,
                    reranker_type=reranker_choice,
                    llm_model=llm_choice if llm_choice != "none" else None,
                    use_query_enhancement=use_qem,
                    top_k=5,
                    limit=max_rows,
                )
                st.success(f"RAG run completed: {res['processed']} processed, {res['errors']} errors")
                if res.get('error_details'):
                    st.json(res['error_details'])
            except Exception as e:
                st.error(f"RAG evaluation failed: {e}")

    def _run_semantic_similarity_evaluation(self, embedder_choice, reranker_choice, use_qem, sample_size, save_to_db):
        """Run semantic similarity evaluation metric."""
        with st.spinner("Running semantic similarity evaluation (may take a while)..."):
            try:
                max_rows = None if sample_size == 0 else int(sample_size)
                res = self.backend.evaluate_ground_truth_with_semantic_similarity(
                    embedder_type=embedder_choice,
                    reranker_type=reranker_choice,
                    use_query_enhancement=use_qem,
                    top_k=10,
                    limit=max_rows,
                    save_to_db=save_to_db,
                )

                # Display summary
                summary = res.get('summary', {})
                st.subheader("📊 Semantic Similarity Evaluation Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Questions", summary.get('total_questions', 0))
                with col2:
                    st.metric("Processed", summary.get('processed', 0))
                with col3:
                    st.metric("Errors", summary.get('errors', 0))
                with col4:
                    st.metric("Avg Similarity", f"{summary.get('avg_semantic_similarity', 0):.4f}")

                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Avg Best Match", f"{summary.get('avg_best_match_score', 0):.4f}")
                with col6:
                    st.metric("Chunks > Threshold", summary.get('total_chunks_above_threshold', 0))
                with col7:
                    st.metric("Embedder", summary.get('embedder_used', 'unknown'))

                # Display detailed results
                results = res.get('results', [])
                if results:
                    st.subheader("📋 Detailed Results")

                    # Create dataframe for display
                    df_data = []
                    for res_item in results:
                        df_data.append({
                            'ID': res_item.get('ground_truth_id', 'unknown'),
                            'Semantic Similarity': f"{res_item.get('semantic_similarity', 0):.4f}",
                            'Best Match Score': f"{res_item.get('best_match_score', 0):.4f}",
                            'Retrieved Chunks': res_item.get('retrieved_chunks', 0),
                            'Chunks > Threshold': res_item.get('chunks_above_threshold', 0),
                            'Status': '❌ Error' if res_item.get('error') else '✅ OK'
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)

                    # Show expandable details for each result
                    for i, res_item in enumerate(results):
                        with st.expander(f"Question ID {res_item.get('ground_truth_id', 'unknown')} - Similarity: {res_item.get('semantic_similarity', 0):.4f}"):
                            if res_item.get('error'):
                                st.error(f"Error: {res_item['error']}")
                            else:
                                matched_chunks = res_item.get('matched_chunks', [])
                                if matched_chunks:
                                    st.write("**Top Matched Chunks:**")
                                    for j, chunk in enumerate(matched_chunks[:5], 1):  # Show top 5
                                        st.write(f"{j}. **Score:** {chunk['similarity_score']:.4f} - **File:** {chunk['file_name']} (page {chunk['page_number']})")
                                        if 'chunk_text' in chunk:
                                            st.write(f"   *Text:* {chunk['chunk_text'][:200]}...")
                                else:
                                    st.warning("No chunks matched above threshold")

                # Display errors if any
                errors_list = res.get('errors_list', [])
                if errors_list:
                    st.subheader("❌ Errors")
                    for error in errors_list[:10]:  # Show first 10 errors
                        st.error(error)
                    if len(errors_list) > 10:
                        st.info(f"... and {len(errors_list) - 10} more errors")

                st.success(f"Semantic similarity evaluation completed: {summary.get('processed', 0)} processed, {summary.get('errors', 0)} errors")

            except Exception as e:
                st.error(f"Semantic similarity evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    def _run_recall_evaluation(self, embedder_choice, reranker_choice, use_qem, sample_size, save_to_db):
        """Run recall evaluation metric."""
        with st.spinner("Running recall evaluation (may take a while)..."):
            try:
                max_rows = None if sample_size == 0 else int(sample_size)
                res = self.backend.evaluate_recall(
                    embedder_type=embedder_choice,
                    reranker_type=reranker_choice,
                    use_query_enhancement=use_qem,
                    top_k=10,
                    similarity_threshold=0.5,
                    limit=max_rows,
                    save_to_db=save_to_db,
                )

                # Display summary
                summary = res.get('summary', {})
                st.subheader("📊 Recall Evaluation Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Recall", f"{summary.get('overall_recall', 0):.4f}")
                with col2:
                    st.metric("Overall Precision", f"{summary.get('overall_precision', 0):.4f}")
                with col3:
                    st.metric("Overall F1 Score", f"{summary.get('overall_f1_score', 0):.4f}")
                with col4:
                    st.metric("Avg Recall", f"{summary.get('avg_recall', 0):.4f}")

                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("Avg Precision", f"{summary.get('avg_precision', 0):.4f}")
                with col6:
                    st.metric("True Positives", summary.get('total_true_positives', 0))
                with col7:
                    st.metric("False Positives", summary.get('total_false_positives', 0))
                with col8:
                    st.metric("False Negatives", summary.get('total_false_negatives', 0))

                # Display detailed results
                results = res.get('results', [])
                if results:
                    st.subheader("📋 Detailed Recall Results")

                    # Create dataframe for display
                    df_data = []
                    for res_item in results:
                        df_data.append({
                            'ID': res_item.get('ground_truth_id', 'unknown'),
                            'Recall': f"{res_item.get('recall', 0):.4f}",
                            'Precision': f"{res_item.get('precision', 0):.4f}",
                            'F1 Score': f"{res_item.get('f1_score', 0):.4f}",
                            'TP': res_item.get('true_positives', 0),
                            'FP': res_item.get('false_positives', 0),
                            'FN': res_item.get('false_negatives', 0),
                            'Status': '❌ Error' if res_item.get('error') else '✅ OK'
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)

                # Display errors if any
                errors_list = res.get('errors_list', [])
                if errors_list:
                    st.subheader("❌ Errors")
                    for error in errors_list[:10]:
                        st.error(error)

                st.success(f"Recall evaluation completed: {summary.get('processed', 0)} processed, {summary.get('errors', 0)} errors")

            except Exception as e:
                st.error(f"Recall evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    def _run_relevance_evaluation(self, embedder_choice, reranker_choice, use_qem, sample_size, save_to_db):
        """Run relevance evaluation metric."""
        with st.spinner("Running relevance evaluation (may take a while)..."):
            try:
                max_rows = None if sample_size == 0 else int(sample_size)
                res = self.backend.evaluate_relevance(
                    embedder_type=embedder_choice,
                    reranker_type=reranker_choice,
                    use_query_enhancement=use_qem,
                    top_k=10,
                    limit=max_rows,
                    save_to_db=save_to_db,
                )

                # Display summary
                summary = res.get('summary', {})
                st.subheader("🎯 Relevance Evaluation Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Overall Relevance", f"{summary.get('avg_overall_relevance', 0):.4f}")
                with col2:
                    st.metric("Avg Chunk Relevance", f"{summary.get('avg_chunk_relevance', 0):.4f}")
                with col3:
                    st.metric("Global Avg Relevance", f"{summary.get('global_avg_relevance', 0):.4f}")
                with col4:
                    st.metric("High Relevance Ratio", f"{summary.get('global_high_relevance_ratio', 0):.4f}")

                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Relevant Ratio (>0.5)", f"{summary.get('global_relevant_ratio', 0):.4f}")
                with col6:
                    st.metric("Total Chunks", summary.get('total_chunks_evaluated', 0))
                with col7:
                    st.metric("Processed Questions", summary.get('processed', 0))

                # Display relevance distribution
                dist = summary.get('relevance_distribution', {})
                if dist:
                    st.subheader("📈 Relevance Score Distribution")
                    dist_df = pd.DataFrame(list(dist.items()), columns=['Score Range', 'Count'])
                    st.bar_chart(dist_df.set_index('Score Range'))

                # Display detailed results
                results = res.get('results', [])
                if results:
                    st.subheader("📋 Detailed Relevance Results")

                    # Create dataframe for display
                    df_data = []
                    for res_item in results:
                        df_data.append({
                            'ID': res_item.get('ground_truth_id', 'unknown'),
                            'Overall Relevance': f"{res_item.get('overall_relevance', 0):.4f}",
                            'Avg Chunk Relevance': f"{res_item.get('avg_chunk_relevance', 0):.4f}",
                            'Semantic Similarity': f"{res_item.get('semantic_similarity', 0):.4f}",
                            'Relevant Ratio': f"{res_item.get('relevant_chunks_ratio', 0):.4f}",
                            'High Relevance Chunks': res_item.get('high_relevance_chunks', 0),
                            'Status': '❌ Error' if res_item.get('error') else '✅ OK'
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)

                # Display errors if any
                errors_list = res.get('errors_list', [])
                if errors_list:
                    st.subheader("❌ Errors")
                    for error in errors_list[:10]:
                        st.error(error)

                st.success(f"Relevance evaluation completed: {summary.get('processed', 0)} processed, {summary.get('errors', 0)} errors")

            except Exception as e:
                st.error(f"Relevance evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    def _run_full_evaluation_suite(self, embedder_choice, reranker_choice, use_qem, sample_size, save_to_db):
        """Run complete evaluation suite with all 3 metrics."""
        with st.spinner("Running full evaluation suite (may take several minutes)..."):
            try:
                max_rows = None if sample_size == 0 else int(sample_size)

                # Run all three evaluations
                semantic_result = self.backend.evaluate_ground_truth_with_semantic_similarity(
                    embedder_type=embedder_choice,
                    reranker_type=reranker_choice,
                    use_query_enhancement=use_qem,
                    top_k=10,
                    limit=max_rows,
                    save_to_db=save_to_db,
                )

                recall_result = self.backend.evaluate_recall(
                    embedder_type=embedder_choice,
                    reranker_type=reranker_choice,
                    use_query_enhancement=use_qem,
                    top_k=10,
                    similarity_threshold=0.5,
                    limit=max_rows,
                    save_to_db=save_to_db,
                )

                relevance_result = self.backend.evaluate_relevance(
                    embedder_type=embedder_choice,
                    reranker_type=reranker_choice,
                    use_query_enhancement=use_qem,
                    top_k=10,
                    limit=max_rows,
                    save_to_db=save_to_db,
                )

                # Display comprehensive summary
                st.subheader("🚀 Full Evaluation Suite Results")

                # Create comparison table
                comparison_data = {
                    'Metric': ['Ground-truth Coverage', 'Semantic Similarity', 'Recall', 'Precision', 'F1 Score', 'Overall Relevance'],
                    'Score': [
                        f"{semantic_result.get('summary', {}).get('processed', 0)}/{semantic_result.get('summary', {}).get('total_questions', 0)}",
                        f"{semantic_result.get('summary', {}).get('avg_semantic_similarity', 0):.4f}",
                        f"{recall_result.get('summary', {}).get('overall_recall', 0):.4f}",
                        f"{recall_result.get('summary', {}).get('overall_precision', 0):.4f}",
                        f"{recall_result.get('summary', {}).get('overall_f1_score', 0):.4f}",
                        f"{relevance_result.get('summary', {}).get('avg_overall_relevance', 0):.4f}"
                    ]
                }

                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)

                # Display individual summaries in tabs
                tab1, tab2, tab3 = st.tabs(["📊 Semantic Similarity", "📈 Recall Metrics", "🎯 Relevance Scores"])

                with tab1:
                    sem_sum = semantic_result.get('summary', {})
                    st.metric("Semantic Similarity", f"{sem_sum.get('avg_semantic_similarity', 0):.4f}")
                    st.metric("Best Match Score", f"{sem_sum.get('avg_best_match_score', 0):.4f}")
                    st.metric("Chunks Above Threshold", sem_sum.get('total_chunks_above_threshold', 0))

                with tab2:
                    rec_sum = recall_result.get('summary', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Recall", f"{rec_sum.get('overall_recall', 0):.4f}")
                    with col2:
                        st.metric("Precision", f"{rec_sum.get('overall_precision', 0):.4f}")
                    with col3:
                        st.metric("F1 Score", f"{rec_sum.get('overall_f1_score', 0):.4f}")

                with tab3:
                    rel_sum = relevance_result.get('summary', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Relevance", f"{rel_sum.get('avg_overall_relevance', 0):.4f}")
                    with col2:
                        st.metric("High Relevance Ratio", f"{rel_sum.get('global_high_relevance_ratio', 0):.4f}")
                    with col3:
                        st.metric("Relevant Ratio", f"{rel_sum.get('global_relevant_ratio', 0):.4f}")

                    # Show relevance distribution
                    dist = rel_sum.get('relevance_distribution', {})
                    if dist:
                        st.subheader("Relevance Distribution")
                        dist_df = pd.DataFrame(list(dist.items()), columns=['Range', 'Count'])
                        st.bar_chart(dist_df.set_index('Range'))

                # Configuration summary
                st.subheader("⚙️ Evaluation Configuration")
                config_data = {
                    'Setting': ['Embedder', 'Reranker', 'Query Enhancement', 'Top K', 'Sample Size'],
                    'Value': [embedder_choice, reranker_choice, str(use_qem), '10', str(sample_size) if sample_size > 0 else 'All']
                }
                config_df = pd.DataFrame(config_data)
                st.table(config_df)

                st.success("✅ Full evaluation suite completed successfully!")

            except Exception as e:
                st.error(f"Full evaluation suite failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    def _run_5_gt_with_logs(self, embedder_choice, reranker_choice, llm_choice, use_qem):
        """Run 5 ground truth queries with streaming logs."""
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
                        # Check if vector store exists before attempting retrieval
                        import os
                        vector_files = [f for f in os.listdir('data/vectors') if f.endswith('.faiss') or f.endswith('.pkl')]
                        if not vector_files:
                            logs.append("⚠️ No vector store found - skipping retrieval")
                            logs.append("Retrieved chunks: 0")
                            logs.append("Context (truncated 1000 chars): ")
                            logs.append("Sources: []")
                            logs.append("💡 To enable retrieval, first load PDF documents and run: python -c \"from pipeline.rag_pipeline import RAGPipeline; RAGPipeline().process_directory('data/pdf')\"")
                            logs.append("")
                            logs_box.code("\n".join(logs), language='text')
                            continue

                        # Call the same retrieval method used by RAG pipeline
                        from pipeline.rag_pipeline import RAGPipeline
                        from pipeline.retrieval.retrieval_service import RAGRetrievalService
                        from query_enhancement.query_processor import create_query_processor

                        # Parse embedder type (same logic as retrieval_orchestrator)
                        def _parse_embedder_type(embedder_type: str):
                            from embedders.embedder_type import EmbedderType
                            if embedder_type.lower() in ["e5_large_instruct", "e5_base", "gte_multilingual_base",
                                                       "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"]:
                                return EmbedderType.HUGGINGFACE, False
                            elif embedder_type == "huggingface_api":
                                return EmbedderType.HUGGINGFACE, True
                            elif embedder_type == "huggingface_local":
                                return EmbedderType.HUGGINGFACE, False
                            else:  # ollama and others
                                return EmbedderType.OLLAMA, False

                        embedder_enum, use_api = _parse_embedder_type(embedder_choice)

                        # Initialize pipeline (reuse cached if available)
                        pipeline = RAGPipeline(embedder_type=embedder_enum, hf_use_api=use_api)

                        # Override embedder for specific multilingual models
                        if embedder_choice.lower() in ["e5_large_instruct", "e5_base", "gte_multilingual_base",
                                                     "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"]:
                            from embedders.embedder_factory import EmbedderFactory
                            factory = EmbedderFactory()

                            if embedder_choice.lower() == "e5_large_instruct":
                                pipeline.embedder = factory.create_e5_large_instruct(device="cpu")
                            elif embedder_choice.lower() == "e5_base":
                                pipeline.embedder = factory.create_e5_base(device="cpu")
                            elif embedder_choice.lower() == "gte_multilingual_base":
                                pipeline.embedder = factory.create_gte_multilingual_base(device="cpu")
                            elif embedder_choice.lower() == "paraphrase_mpnet_base_v2":
                                pipeline.embedder = factory.create_paraphrase_mpnet_base_v2(device="cpu")
                            elif embedder_choice.lower() == "paraphrase_minilm_l12_v2":
                                pipeline.embedder = factory.create_paraphrase_minilm_l12_v2(device="cpu")

                        # Create retriever service
                        retriever = RAGRetrievalService(pipeline)

                        # Query Enhancement (same as RAG pipeline)
                        query_processor = create_query_processor(use_qem, pipeline.embedder)
                        expanded_queries = query_processor.enhance_query(question, use_qem)

                        # Create fused embedding
                        fused_embedding = query_processor.fuse_query_embeddings(expanded_queries)
                        bm25_query = " ".join(expanded_queries).strip()

                        # Hybrid retrieval
                        results = retriever.retrieve_hybrid(
                            query_text=question,
                            top_k=10,
                            query_embedding=fused_embedding,
                            bm25_query=bm25_query,
                        )

                        context = retriever.build_context(results, max_chars=2000)
                        sources = retriever.to_ui_items(results)

                        total_retrieved = len(results)

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
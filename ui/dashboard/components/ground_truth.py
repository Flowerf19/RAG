"""
Ground Truth Component
Handles ground truth Q&A import and evaluation.
"""

import streamlit as st
import pandas as pd
import logging
import altair as alt
from evaluation.backend_dashboard.api import BackendDashboard


class GroundTruthComponent:
    """Component for ground truth import and evaluation."""

    def __init__(self, backend: BackendDashboard):
        """Initialize with backend API."""
        self.backend = backend
        try:
            self.logger = logging.getLogger(self.__class__.__module__)
        except Exception:
            self.logger = logging.getLogger(__name__)
        # Use stable key prefix so widget state (e.g., uploaded files) persists across reruns
        self._key_prefix = "ground_truth_main"

    def display(self):
        """Display ground truth import and evaluation interface."""
        st.header("Import Ground-truth Q&A")
        st.markdown("Upload an Excel (.xlsx/.xls) or CSV file with columns: `STT`, `Câu hỏi`, `Câu trả lời`, `Nguồn`.")

        # Hướng dẫn chọn model để tránh lỗi
        with st.expander("Hướng dẫn chọn Model (quan trọng!)", expanded=False):
            st.markdown("""
            **⚠️ NGUYÊN TẮC: Embedding phải cùng chiều với vector store (1024d)**

            **✅ Embedder tương thích (1024d):**
            - `huggingface_local` ✅ (BGE-M3, mặc định)
            - `e5_large_instruct` ✅ (1024d)

            **❌ Embedder KHÔNG tương thích (768d/384d):**
            - `ollama` ❌ (Gemma, 768d)
            - `e5_base` ❌ (768d)
            - `gte_multilingual_base` ❌ (768d)
            - `paraphrase_mpnet_base_v2` ❌ (768d)
            - `paraphrase_minilm_l12_v2` ❌ (384d)

            **📊 Metrics cần gì:**
            - **Faithfulness:** Cần LLM (gemini/ollama/lmstudio/openai)
            - **Relevance:** Chỉ cần embedder
            - **Recall:** Chỉ cần embedder

            **🔄 Đổi embedder khác:** Xóa vector store cũ và rebuild
            """)

        # Controls: select embedder, reranker, LLM, QEM toggle, sample size
        embedder_choice = st.selectbox(
            "Embedder Model",
            ["ollama", "huggingface_local", "huggingface_api", "e5_large_instruct", "e5_base", "gte_multilingual_base", "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"],
            index=0,  # Default to huggingface_local (BGE-M3, 1024d) - compatible with vector store
            help="Choose which embedder to use for evaluation (also used for retrieval if supported).",
            key=f"{self._key_prefix}_embedder_select",
        )

        reranker_choice = st.selectbox(
            "Reranker",
            ["none", "bge_m3_ollama", "bge_m3_hf_api", "bge_m3_hf_local", "jina_v2_multilingual", "gte_multilingual", "bge_base"],
            index=0,
            help="Choose reranker used during retrieval (if any).",
            key=f"{self._key_prefix}_reranker_select",
        )

        llm_choice = st.selectbox(
            "LLM for Evaluation",
            ["none", "gemini", "ollama", "lmstudio", "openai"],
            index=0,
            help="Choose LLM to use for answer quality evaluation (faithfulness, relevance). 'none' skips evaluation.",
            key=f"{self._key_prefix}_llm_select",
        )

        use_qem = st.checkbox(
            "Use Query Enhancement (QEM)",
            value=True,
            key=f"{self._key_prefix}_use_qem",
        )

        sample_size = st.number_input(
            "Max ground-truth questions to evaluate (0 = all questions)",
            value=10,
            min_value=0,
            step=1,
            help="Number of ground-truth questions to test. Each question will retrieve top 10 chunks for evaluation.",
            key=f"{self._key_prefix}_sample_size",
        )

        save_to_db = st.checkbox(
            "Save results to database",
            value=True,
            help="Store evaluation results for historical tracking and dashboard analytics",
            key=f"{self._key_prefix}_save_to_db",
        )

        uploaded = st.file_uploader(
            "Choose ground-truth file",
            type=["xlsx", "xls", "csv"],
            key=f"{self._key_prefix}_upload",
        )

        if uploaded is not None:
            self._handle_file_upload(
                uploaded,
                embedder_choice,
                reranker_choice,
                llm_choice,
                use_qem,
                sample_size,
                save_to_db,
            )

    def _handle_file_upload(
        self,
        uploaded,
        embedder_choice,
        reranker_choice,
        llm_choice,
        use_qem,
        sample_size,
        save_to_db,
    ):
        """Handle uploaded ground truth file."""
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.write(f"Detected {df.shape[0]} rows in uploaded file")

            normalized = self._map_columns(df)

            st.subheader("Preview parsed ground-truth")
            st.write(f"Detected {normalized.shape[0]} parsed rows. Use the checkbox below to show all rows.")
            show_all = st.checkbox("Show full parsed preview (all rows)", value=False, key=f"{self._key_prefix}_show_all_preview")
            if show_all:
                st.dataframe(normalized, width='stretch')
            else:
                st.dataframe(normalized.head(10), width='stretch')

            if st.button("Import to DB", key=f"{self._key_prefix}_import_gt"):
                # Prevent duplicate imports
                import_key = f"import_done_{hash(str(normalized.values.tobytes())[:10])}"
                if not st.session_state.get(import_key, False):
                    self._import_to_db(normalized)
                    st.session_state[import_key] = True
                    st.success("✅ Import completed!")
                else:
                    st.info("ℹ️ Data already imported (skipping duplicate operation)")

            # Button to run full evaluation suite (all 3 metrics)
            eval_key = f"eval_{embedder_choice}_{reranker_choice}_{llm_choice}_{use_qem}_{sample_size}_{save_to_db}"
            if st.button("Full Evaluation Suite (Ground-truth + Recall + Relevance + Faithfulness)", key=f"{self._key_prefix}_run_full_eval"):
                # Prevent duplicate evaluations
                if not st.session_state.get(f"eval_done_{eval_key}", False):
                    self._run_full_evaluation_suite(embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db)
                    st.session_state[f"eval_done_{eval_key}"] = True
                else:
                    st.info("ℹ️ Evaluation already completed for these settings (skipping duplicate operation)")

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
        try:
            self.logger.info("Importing normalized ground-truth to DB: %d rows", len(normalized_df))
        except Exception:
            pass
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
            # Clear evaluation done flags to allow re-running after new import
            for key in list(st.session_state.keys()):
                if key.startswith("eval_done_"):
                    del st.session_state[key]
            try:
                self.logger.info("Inserted %d ground-truth rows into DB", inserted)
            except Exception:
                pass
        except Exception as e:
            st.error(f"Failed to insert ground-truth into DB: {e}")
            try:
                self.logger.exception("Failed to insert ground-truth into DB: %s", e)
            except Exception:
                pass

    def _run_rag_evaluation(self, embedder_choice, reranker_choice, llm_choice, use_qem, sample_size):
        """Run RAG evaluation on ground truth."""
        try:
            self.logger.info("Starting RAG evaluation: embedder=%s reranker=%s llm=%s use_qem=%s sample_size=%s",
                             embedder_choice, reranker_choice, llm_choice, use_qem, sample_size)
        except Exception:
            pass
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
                try:
                    self.logger.info("RAG evaluation completed: processed=%s errors=%s", res.get('processed'), res.get('errors'))
                except Exception:
                    pass
            except Exception as e:
                st.error(f"RAG evaluation failed: {e}")
                try:
                    self.logger.exception("RAG evaluation failed: %s", e)
                except Exception:
                    pass

    def _run_full_evaluation_suite(self, embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db):
        """Run complete evaluation suite with all 3 metrics."""
        try:
            self.logger.info("Starting full evaluation suite: embedder=%s reranker=%s llm=%s use_qem=%s sample_size=%s save_to_db=%s",
                             embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db)
        except Exception:
            pass
        with st.spinner("Running full evaluation suite (may take several minutes)..."):
            try:
                # Run all evaluations in batch to avoid redundant retrieval
                res = self.backend.evaluate_all_metrics_batch(
                    embedder_type=embedder_choice,
                    reranker_type=reranker_choice,
                    llm_choice=llm_choice,
                    use_query_enhancement=use_qem,
                    top_k=10,
                    limit=None,  # Always run on all ground truth entries
                    save_to_db=save_to_db,
                )

                # Extract results from batch
                semantic_result = res.get('semantic_similarity', {})
                recall_result = res.get('recall', {})
                relevance_result = res.get('relevance', {})
                faithfulness_result = res.get('faithfulness', {})

                # Display comprehensive summary
                st.subheader("Full Evaluation Suite Results")

                # Create comparison table
                comparison_data = {
                    'Metric': ['Ground-truth Coverage', 'Semantic Similarity', 'Recall', 'Precision', 'F1 Score', 'Overall Relevance', 'Avg Faithfulness'],
                    'Score': [
                        f"{semantic_result.get('summary', {}).get('processed', 0)}/{semantic_result.get('summary', {}).get('total_questions', 0)}",
                        f"{semantic_result.get('summary', {}).get('avg_semantic_similarity', 0):.4f}",
                        f"{recall_result.get('summary', {}).get('overall_recall', 0):.4f}",
                        f"{recall_result.get('summary', {}).get('overall_precision', 0):.4f}",
                        f"{recall_result.get('summary', {}).get('overall_f1_score', 0):.4f}",
                        f"{relevance_result.get('summary', {}).get('avg_overall_relevance', 0):.4f}",
                        f"{faithfulness_result.get('summary', {}).get('avg_faithfulness', 0):.4f}"
                    ]
                }

                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)

                # Dedicated Faithfulness and Recall table
                st.subheader("Faithfulness & Recall Summary")
                faithfulness_recall_data = {
                    'Metric': ['Average Faithfulness Score', 'High Faithfulness Ratio (>0.8)', 'Faithful Ratio (>0.5)', 'Overall Recall', 'Recall Precision', 'Recall F1 Score'],
                    'Score': [
                        f"{faithfulness_result.get('summary', {}).get('avg_faithfulness', 0):.4f}",
                        f"{faithfulness_result.get('summary', {}).get('global_high_faithfulness_ratio', 0):.4f}",
                        f"{faithfulness_result.get('summary', {}).get('global_faithful_ratio', 0):.4f}",
                        f"{recall_result.get('summary', {}).get('overall_recall', 0):.4f}",
                        f"{recall_result.get('summary', {}).get('overall_precision', 0):.4f}",
                        f"{recall_result.get('summary', {}).get('overall_f1_score', 0):.4f}"
                    ]
                }
                faithfulness_recall_df = pd.DataFrame(faithfulness_recall_data)
                st.table(faithfulness_recall_df)
                tab1, tab2, tab3, tab4 = st.tabs(["Semantic Similarity", "Recall Metrics", "Relevance Scores", "Faithfulness Scores"])

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

                    dist = rel_sum.get('relevance_distribution', {})
                    self._render_score_distribution("Relevance Distribution", relevance_result.get('results', []), dist, score_keys=['overall_relevance', 'avg_chunk_relevance', 'semantic_similarity'])

                with tab4:
                    faith_sum = faithfulness_result.get('summary', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Faithfulness", f"{faith_sum.get('avg_faithfulness', 0):.4f}")
                    with col2:
                        st.metric("High Faithfulness Ratio", f"{faith_sum.get('global_high_faithfulness_ratio', 0):.4f}")
                    with col3:
                        st.metric("Faithful Ratio (>0.5)", f"{faith_sum.get('global_faithful_ratio', 0):.4f}")

                    dist = faith_sum.get('faithfulness_distribution', {})
                    self._render_score_distribution("Faithfulness Distribution", faithfulness_result.get('results', []), dist, score_keys=['faithfulness_score'])

                # Configuration summary
                st.subheader("Evaluation Configuration")
                config_data = {
                    'Setting': ['Embedder', 'Reranker', 'Query Enhancement', 'Top K (chunks per question)', 'Max Questions to Evaluate'],
                    'Value': [embedder_choice, reranker_choice, str(use_qem), '10', str(sample_size) if sample_size > 0 else 'All']
                }
                config_df = pd.DataFrame(config_data)
                st.table(config_df)

                st.success("✅ Full evaluation suite completed successfully!")
                try:
                    self.logger.info("Full evaluation suite completed successfully")
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Full evaluation suite failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                try:
                    self.logger.exception("Full evaluation suite failed: %s", e)
                except Exception:
                    pass

    def _render_score_distribution(self, title: str, results: list, dist_dict: dict, score_keys: list):
        """Render a distribution: prefer raw per-result scores (boxplot + histogram),
        fallback to bucketed counts when raw scores are not available.
        """
        try:
            # Extract numeric scores from results using candidate keys
            scores = []
            for r in (results or []):
                for key in score_keys:
                    if not isinstance(r, dict):
                        continue
                    v = r.get(key)
                    if v is None:
                        continue
                    try:
                        scores.append(float(v))
                        break
                    except Exception:
                        continue

            # If we have enough raw scores, show boxplot + histogram
            if len(scores) >= 3:
                df_scores = pd.DataFrame({'score': scores})
                st.subheader(title)
                box = alt.Chart(df_scores).mark_boxplot().encode(x='score:Q').properties(height=170)
                hist = alt.Chart(df_scores).mark_bar().encode(
                    x=alt.X('score:Q', bin=alt.Bin(maxbins=30), title='Score'),
                    y=alt.Y('count()', title='Count')
                ).properties(height=170)
                combined_chart = alt.vconcat(box, hist)
                st.altair_chart(combined_chart, use_container_width=True)
                return

            # Fallback to bucket dict if provided
            if dist_dict:
                dist_df = pd.DataFrame(list(dist_dict.items()), columns=['range', 'count'])
                st.subheader(title)
                chart = alt.Chart(dist_df).mark_bar().encode(
                    x=alt.X('count:Q', title='Count'),
                    y=alt.Y('range:N', sort=alt.SortField('count', order='descending'), title='Range'),
                    tooltip=['range', 'count']
                ).properties(height=340)
                st.altair_chart(chart, use_container_width=True)
                return

            st.info(f"No distribution data available for {title}")

        except Exception as e:
            st.error(f"Failed to render distribution: {e}")

    def _run_5_gt_with_logs(self, embedder_choice, reranker_choice, llm_choice, use_qem):
        """Run 5 ground truth queries with streaming logs."""
        try:
            self.logger.info("Running 5 GT with logs: embedder=%s reranker=%s llm=%s use_qem=%s",
                             embedder_choice, reranker_choice, llm_choice, use_qem)
        except Exception:
            pass
        # Create a live log area
        logs_box = st.empty()
        logs = []

        try:
            # Fetch up to 5 ground-truth rows from DB
            rows = self.backend.get_ground_truth_list(limit=5)

            if not rows:
                st.info("No ground-truth rows available to run.")
            else:
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

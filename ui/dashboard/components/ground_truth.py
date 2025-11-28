"""
Ground Truth Component
Handles ground truth Q&A import and evaluation.
"""

import streamlit as st
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation.backend_dashboard.api import BackendDashboard

# Import extracted services
from .ground_truth.file_handler import GroundTruthFileHandler
from .ground_truth.evaluation_service import GroundTruthEvaluationService
from .ground_truth.embedder_helper import EmbedderHelper


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
        
        # Initialize extracted services
        self.file_handler = GroundTruthFileHandler(backend, self.logger)
        self.evaluation_service = GroundTruthEvaluationService(backend, self.logger)
        self.embedder_helper = EmbedderHelper()

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
            self.file_handler.handle_file_upload(
                uploaded,
                embedder_choice,
                reranker_choice,
                llm_choice,
                use_qem,
                sample_size,
                save_to_db,
                key_prefix=self._key_prefix,
            )



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
                try:
                    fig, (ax_box, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), gridspec_kw={'height_ratios': [1, 1]})
                    sns.boxplot(x='score', data=df_scores, ax=ax_box, color='lightgray')
                    ax_box.set_xlabel('')
                    ax_box.set_title(f'{title} - Boxplot')

                    sns.histplot(df_scores['score'], bins=30, kde=False, ax=ax_hist, color='skyblue')
                    ax_hist.set_xlabel('Score')
                    ax_hist.set_ylabel('Count')
                    ax_hist.set_title(f'{title} - Histogram')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Could not render score plots: {e}")
                return

            # Fallback to bucket dict if provided
            if dist_dict:
                dist_df = pd.DataFrame(list(dist_dict.items()), columns=['range', 'count'])
                st.subheader(title)
                try:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    dist_df_sorted = dist_df.sort_values('count', ascending=True)
                    ax.barh(dist_df_sorted['range'], dist_df_sorted['count'], color='teal')
                    ax.set_xlabel('Count')
                    ax.set_ylabel('Range')
                    ax.set_title(f'{title} Distribution')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Could not render distribution chart: {e}")
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

                        # Parse embedder type and configure pipeline
                        embedder_enum, use_api = self.embedder_helper.parse_embedder_type(embedder_choice)
                        
                        # Initialize pipeline (reuse cached if available)
                        pipeline = RAGPipeline(embedder_type=embedder_enum, hf_use_api=use_api)

                        # Override embedder for specific multilingual models
                        pipeline.embedder = self.embedder_helper.configure_embedder_for_choice(pipeline.embedder, embedder_choice)

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

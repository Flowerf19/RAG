"""
Ground Truth File Handler
Handles file upload, parsing, and database import for ground truth data.
"""

import streamlit as st
import pandas as pd
import logging


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and data from uploaded files."""
    # Expected column mappings (case-insensitive)
    column_mappings = {
        'question': ['question', 'câu hỏi', 'cau hoi', 'q', 'query'],
        'answer': ['answer', 'câu trả lời', 'cau tra loi', 'a', 'response'],
        'source': ['source', 'nguồn', 'nguon', 's', 'reference']
    }

    # Create normalized dataframe
    normalized = pd.DataFrame()

    # Find and map columns
    df_columns = [col.lower().strip() for col in df.columns]

    for target_col, possible_names in column_mappings.items():
        for possible_name in possible_names:
            if possible_name in df_columns:
                col_idx = df_columns.index(possible_name)
                normalized[target_col] = df.iloc[:, col_idx].fillna('')
                break

    # Ensure all required columns exist
    for col in ['question', 'answer', 'source']:
        if col not in normalized.columns:
            normalized[col] = ''

    return normalized


class GroundTruthFileHandler:
    """Handles ground truth file operations."""

    def __init__(self, backend, logger=None):
        self.backend = backend
        self.logger = logger or logging.getLogger(__name__)

    def handle_file_upload(self, uploaded, embedder_choice, reranker_choice, llm_choice,
                          use_qem, sample_size, save_to_db, key_prefix="ground_truth_main"):
        """Handle uploaded ground truth file."""
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.write(f"Detected {df.shape[0]} rows in uploaded file")

            # Normalize columns
            normalized = normalize_columns(df)

            st.subheader("Preview parsed ground-truth")
            st.write(f"Detected {normalized.shape[0]} parsed rows. Use the checkbox below to show all rows.")
            show_all = st.checkbox("Show full parsed preview (all rows)", value=False, key=f"{key_prefix}_show_all_preview")
            if show_all:
                st.dataframe(normalized, width='stretch')
            else:
                st.dataframe(normalized.head(10), width='stretch')

            # Auto-run options
            auto_import = st.checkbox(
                "Auto-import to DB after file load",
                value=True,
                help="Automatically import ground truth data to database when file is loaded",
                key=f"{key_prefix}_auto_import"
            )
            
            auto_run_eval = st.checkbox(
                "Auto-run evaluation after import",
                value=True,
                help="Automatically run full evaluation suite after importing data",
                key=f"{key_prefix}_auto_run_eval"
            )

            # Manual import button (always available)
            if st.button("Import to DB", key=f"{key_prefix}_import_gt"):
                self._import_to_db(normalized)
                st.success("✅ Import completed!")
                
                # Auto-run evaluation if enabled
                if auto_run_eval:
                    st.info("🔄 Auto-running evaluation...")
                    self._run_evaluation(embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db, key_prefix)

            # Auto-import logic
            if auto_import and not st.session_state.get(f"auto_import_done_{key_prefix}", False):
                st.info("🔄 Auto-importing ground truth data...")
                self._import_to_db(normalized)
                st.session_state[f"auto_import_done_{key_prefix}"] = True
                st.success("✅ Auto-import completed!")
                
                # Auto-run evaluation if enabled
                if auto_run_eval:
                    st.info("🔄 Auto-running evaluation...")
                    self._run_evaluation(embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db, key_prefix)

            # Manual evaluation button (always available)
            if st.button("Full Evaluation Suite (Ground-truth + Recall + Relevance + Faithfulness)", key=f"{key_prefix}_run_full_eval"):
                self._run_evaluation(embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db, key_prefix)

        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    def _run_evaluation(self, embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db, key_prefix):
        """Run evaluation suite with duplicate prevention."""
        eval_key = f"eval_{embedder_choice}_{reranker_choice}_{llm_choice}_{use_qem}_{sample_size}_{save_to_db}"
        
        # Prevent duplicate evaluations
        if not st.session_state.get(f"eval_done_{eval_key}", False):
            # Import evaluation service here to avoid circular imports
            from .evaluation_service import GroundTruthEvaluationService
            eval_service = GroundTruthEvaluationService(self.backend, self.logger)
            eval_service.run_full_evaluation_suite(embedder_choice, reranker_choice, llm_choice, use_qem, sample_size, save_to_db)
            st.session_state[f"eval_done_{eval_key}"] = True
        else:
            st.info("ℹ️ Evaluation already completed for these settings (skipping duplicate operation)")

    def _import_to_db(self, normalized_df):
        """Import normalized data to database."""
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
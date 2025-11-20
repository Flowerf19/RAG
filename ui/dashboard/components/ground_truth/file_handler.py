"""
Ground Truth File Handler
Handles file upload, parsing, and database import for ground truth data.
"""

import streamlit as st
import pandas as pd
import logging
import unicodedata


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and data from uploaded files."""
    # Expected column mappings (case-insensitive, flexible)
    column_mappings = {
        'question': ['question', 'câu hỏi', 'cau hoi', 'q', 'query', 'câu hoi', 'cau hỏi', 'câu hỏi (question)'],
        'answer': ['answer', 'câu trả lời', 'cau tra loi', 'a', 'response', 'cau tra loi', 'cau trả lời', 'câu trả lời (answer)'],
        'source': ['source', 'nguồn', 'nguon', 's', 'reference', 'nguon', 'nguồn', 'nguồn (source)']
    }

    # Debug: print actual columns
    print(f"DEBUG: Original columns: {list(df.columns)}")
    print(f"DEBUG: Lowercased columns: {[col.lower().strip() for col in df.columns]}")

    # Create normalized dataframe with the same number of rows
    normalized = pd.DataFrame(index=df.index)

    # Find and map columns
    df_columns = [col.lower().strip() for col in df.columns]
    df_columns_normalized = [unicodedata.normalize('NFC', col) for col in df_columns]

    for target_col, possible_names in column_mappings.items():
        possible_names_normalized = [unicodedata.normalize('NFC', name) for name in possible_names]
        
        mapped = False
        for i, possible_name in enumerate(possible_names_normalized):
            if possible_name in df_columns_normalized:
                col_idx = df_columns_normalized.index(possible_name)
                normalized[target_col] = df.iloc[:, col_idx].fillna('')
                print(f"DEBUG: Mapped '{df.columns[col_idx]}' to '{target_col}'")
                mapped = True
                break
        
        # If no exact match, try partial matching
        if not mapped:
            for i, col in enumerate(df_columns_normalized):
                for possible_name in possible_names_normalized:
                    if possible_name in col or col in possible_name:
                        normalized[target_col] = df.iloc[:, i].fillna('')
                        print(f"DEBUG: Partial mapped '{df.columns[i]}' to '{target_col}'")
                        mapped = True
                        break
                if mapped:
                    break
        
        # If still no match, try position-based mapping
        if not mapped:
            # Assume order: STT (index), Question, Answer, Source
            if target_col == 'question' and len(df.columns) >= 2:
                normalized[target_col] = df.iloc[:, 1].astype(str).fillna('')
                print(f"DEBUG: Position mapped column 1 '{df.columns[1]}' to '{target_col}'")
                mapped = True
            elif target_col == 'answer' and len(df.columns) >= 3:
                normalized[target_col] = df.iloc[:, 2].astype(str).fillna('')
                print(f"DEBUG: Position mapped column 2 '{df.columns[2]}' to '{target_col}'")
                mapped = True
            elif target_col == 'source' and len(df.columns) >= 4:
                normalized[target_col] = df.iloc[:, 3].astype(str).fillna('')
                print(f"DEBUG: Position mapped column 3 '{df.columns[3]}' to '{target_col}'")
                mapped = True
        
        # If still no mapping found after all attempts, create empty column
        if not mapped:
            normalized[target_col] = ''

    print(f"DEBUG: Normalized columns: {list(normalized.columns)}")
    print(f"DEBUG: Normalized shape: {normalized.shape}")
    
    # Filter out empty rows (rows where question or answer are empty/NaN)
    # But keep rows that have at least some data
    def has_content(val):
        val_str = str(val).strip().lower()
        return val_str not in ['', 'nan', 'none', 'null', 'nat'] and not pd.isna(val)
    
    # Keep rows where at least question or answer has content
    try:
        mask = []
        for idx, row in normalized.iterrows():
            q_content = has_content(row['question'])
            a_content = has_content(row['answer'])
            has_any = q_content or a_content
            mask.append(has_any)
            print(f"DEBUG: Row {idx}: q='{str(row['question'])[:30]}...', a='{str(row['answer'])[:30]}...', has_content={has_any}")
        
        print(f"DEBUG: Final mask: {mask}")
        normalized = normalized[mask].reset_index(drop=True)
    except Exception as e:
        print(f"DEBUG: Error in filtering: {e}")
        # If filtering fails, keep all rows
        pass
    
    print(f"DEBUG: After filtering empty rows: {normalized.shape}")
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
                # Try to read Excel with more robust options
                try:
                    # Try reading the first sheet
                    df = pd.read_excel(uploaded, engine='openpyxl', sheet_name=0)
                    print(f"DEBUG: Read sheet 0, shape: {df.shape}")
                    
                    # If the first few rows are empty, try to find the header
                    if df.empty or df.iloc[0].isna().all():
                        # Try reading with header detection
                        df = pd.read_excel(uploaded, engine='openpyxl', header=0)
                        print(f"DEBUG: Re-read with header=0, shape: {df.shape}")
                        
                except ImportError:
                    df = pd.read_excel(uploaded)

            print(f"DEBUG: File read successfully. Shape: {df.shape}")
            print(f"DEBUG: Columns: {list(df.columns)}")
            print(f"DEBUG: dtypes:\n{df.dtypes}")
            print(f"DEBUG: First few rows:\n{df.head()}")

            # Skip empty rows at the beginning if any
            df = df.dropna(how='all').reset_index(drop=True)
            
            # Also skip rows where the main columns are all empty
            main_cols = [col for col in df.columns if col.lower().strip() not in ['stt', 'index', 'no', 'number']]
            if main_cols:
                df = df.dropna(subset=main_cols, how='all').reset_index(drop=True)

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
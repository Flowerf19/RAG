import streamlit as st
import pandas as pd
from evaluation.backend_dashboard.api import BackendDashboard

class GroundTruthComponent:
    def __init__(self, backend: BackendDashboard):
        self.backend = backend

    def display(self):
        st.header("Ground Truth QA")
        # File uploader for importing test ground-truth (used for quick dashboard tests)
        st.markdown("**Import ground-truth file (Excel/CSV) for quick testing**")
        uploaded = st.file_uploader("Chọn file (.xlsx/.xls/.csv)", type=['xlsx', 'xls', 'csv'], key='gt_upload')

        # If uploaded, parse and show preview with simple column mapping
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith('.csv'):
                    raw_df = pd.read_csv(uploaded)
                else:
                    raw_df = pd.read_excel(uploaded)

                st.write(f"Detected {raw_df.shape[0]:,} rows in uploaded file")

                def pick_col(df_local, candidates, fallback_idx=None):
                    for c in df_local.columns:
                        lc = str(c).lower()
                        for cand in candidates:
                            if cand in lc:
                                return c
                    if fallback_idx is not None and 0 <= fallback_idx < len(df_local.columns):
                        return df_local.columns[fallback_idx]
                    return None

                q_col = pick_col(raw_df, ['câu hỏi', 'cau hoi', 'question', 'question'])
                src_col = pick_col(raw_df, ['nguồn', 'nguon', 'source'])
                st.subheader("Preview parsed (first 10 rows)")

                parsed = pd.DataFrame()
                if q_col:
                    parsed['question'] = raw_df[q_col].astype(str).fillna('').str.strip()
                else:
                    # fallback to first non-empty column
                    non_empty = [c for c in raw_df.columns if raw_df[c].notna().any()]
                    parsed['question'] = raw_df[non_empty[0]].astype(str).fillna('').str.strip() if non_empty else ''

                if src_col:
                    parsed['source'] = raw_df[src_col].astype(str).fillna('').str.strip()
                else:
                    parsed['source'] = ''

                # Show preview
                st.dataframe(parsed.head(10), use_container_width=True)

                # Buttons and options
                run_eval_after = st.checkbox("Chạy evaluation ngay sau khi import (có thể tốn thời gian)", value=False)
                eval_choice = st.selectbox("Chọn evaluation để chạy", ["none", "semantic_similarity", "recall", "relevance", "full_suite"], index=0)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Import & Save to DB", key='import_save'):
                        rows = []
                        for _, r in parsed.iterrows():
                            rows.append({'question': r.get('question', ''), 'answer': '', 'source': r.get('source', '')})
                        try:
                            inserted = self.backend.insert_ground_truth_rows(rows)
                            st.success(f"Inserted {inserted} rows into ground_truth_qa")
                        except Exception as e:
                            st.error(f"Failed to insert rows: {e}")

                        # Optionally run evaluation
                        if run_eval_after and eval_choice != 'none':
                            self._run_selected_evaluation(eval_choice)

                with col2:
                    if st.button("Import & Save + Run now", key='import_and_run'):
                        rows = []
                        for _, r in parsed.iterrows():
                            rows.append({'question': r.get('question', ''), 'answer': '', 'source': r.get('source', '')})
                        try:
                            inserted = self.backend.insert_ground_truth_rows(rows)
                            st.success(f"Inserted {inserted} rows into ground_truth_qa")
                        except Exception as e:
                            st.error(f"Failed to insert rows: {e}")
                        # Run chosen evaluation (default: semantic_similarity)
                        choice = eval_choice if eval_choice != 'none' else 'semantic_similarity'
                        self._run_selected_evaluation(choice)

            except Exception as e:
                st.error(f"Failed to parse uploaded file: {e}")
            # Done upload handling — return early to keep preview/import UX focused
            return

        # If no upload, fall back to listing existing ground-truth rows
        rows = self.backend.get_ground_truth_list(limit=1000)
        if not rows:
            st.info("Chưa có dữ liệu ground-truth.")
            return

        df = pd.DataFrame(rows)
        st.write(f"Tổng số câu hỏi: {len(df):,}")

        # Safely compute averages if columns exist
        if 'faithfulness' in df.columns:
            try:
                faith_mean = pd.to_numeric(df['faithfulness'], errors='coerce').mean()
                st.write(f"Faithfulness trung bình: {faith_mean:.2f}")
            except Exception:
                st.write("Faithfulness trung bình: N/A")
        else:
            st.write("Faithfulness trung bình: N/A")

        if 'relevance' in df.columns:
            try:
                rel_mean = pd.to_numeric(df['relevance'], errors='coerce').mean()
                st.write(f"Relevance trung bình: {rel_mean:.2f}")
            except Exception:
                st.write("Relevance trung bình: N/A")
        else:
            st.write("Relevance trung bình: N/A")

        # Select columns to display if present
        cols = [c for c in ['id', 'question', 'faithfulness', 'relevance', 'predicted_answer', 'retrieved_context'] if c in df.columns]
        st.dataframe(df[cols].sort_values('id', ascending=False), use_container_width=True)
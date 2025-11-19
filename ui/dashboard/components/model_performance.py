import streamlit as st
import pandas as pd
import altair as alt
from evaluation.backend_dashboard.api import BackendDashboard


class ModelPerformanceComponent:
    def __init__(self, backend: BackendDashboard):
        self.backend = backend

    def display(self):
        st.header("Hiệu năng theo model")

        # Giải thích về tên model
        with st.expander("ℹ️ Hiểu về tên Model", expanded=False):
            st.markdown("""
            **Tên model trong bảng được tạo theo quy tắc thống nhất:**

            **🔹 Định dạng thống nhất: `{embedder}_{reranker}_{llm}`**
            - Ví dụ: `huggingface_local_none_gemini` = embedder huggingface_local + reranker none + llm gemini
            - Ý nghĩa: Tất cả các loại đánh giá (semantic similarity, recall, relevance, faithfulness) từ cùng một cấu hình sẽ được gộp lại

            **🔹 Các thành phần:**
            - **Embedder**: `huggingface_local` (BGE-M3, 1024d), `ollama` (768d), etc.
            - **Reranker**: `none` (không rerank), `bge_m3_hf_local`, `jina_v2_multilingual`, etc.
            - **LLM**: `gemini`, `ollama`, `lmstudio`, `openai` (được dùng cho faithfulness evaluation)

            **💡 Lưu ý:** Bảng này hiện gộp tất cả metrics từ cùng một cấu hình model để dễ so sánh!
            """)

        stats = self.backend.get_model_comparison_data()
        # Combine all models from the unified stats (now all in 'llm' list)
        df = pd.DataFrame(stats['llm'])

        if df.empty:
            st.info("Chưa có dữ liệu model.")
            return

        # Ensure numeric columns exist and fillna
        for col in ['accuracy', 'recall', 'latency', 'error_rate', 'faithfulness', 'relevance']:
            if col not in df.columns:
                df[col] = pd.NA

        # Show table preview
        st.dataframe(df[['model', 'accuracy', 'faithfulness', 'relevance', 'recall', 'latency', 'error_rate']])

        # Scatter: Accuracy vs Recall — size by latency
        scatter_df = df[['model', 'accuracy', 'recall', 'latency']].fillna(0)
        scatter = (
            alt.Chart(scatter_df)
            .mark_square(size=100)
            .encode(
                x=alt.X('accuracy:Q', title='Accuracy'),
                y=alt.Y('recall:Q', title='Recall'),
                size=alt.Size('latency:Q', title='Latency (s)', scale=alt.Scale(range=[100, 1000])),
                color=alt.Color('latency:Q', title='Latency (s)', scale=alt.Scale(scheme='viridis')),
                tooltip=['model', 'accuracy', 'recall', 'latency']
            )
            .interactive()
            .properties(height=420)
        )

        st.altair_chart(scatter, use_container_width=True)

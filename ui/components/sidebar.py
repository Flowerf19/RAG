"""
Sidebar Component - Handles all sidebar UI controls
Includes model selection, file upload, embedding controls, etc.
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import streamlit as st


class Sidebar:
    """
    Sidebar component for app controls and settings
    Manages: file upload, model selection, retrieval settings, embedding controls
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize sidebar component
        
        Args:
            data_dir: Path to data directory for file uploads
        """
        self.data_dir = data_dir
    
    def render(
        self,
        on_embedding_clicked: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Render sidebar and return user selections
        
        Args:
            on_embedding_clicked: Callback when embedding button is clicked
        
        Returns:
            Dict with user selections:
            {
                "backend_mode": str,
                "embedder_type": str,
                "reranker_type": str,
                "top_k_embed": int,
                "top_k_rerank": int,
                "use_query_enhancement": bool
            }
        """
        with st.sidebar:
            self._render_menu_buttons()
            st.markdown("---")
            
            self._render_file_upload()
            st.markdown("---")
            
            # Get user selections
            backend_mode = self._render_backend_selection()
            embedder_type = self._render_embedder_selection()
            reranker_type = self._render_reranker_selection()
            
            # Retrieval settings
            top_k_embed, top_k_rerank = self._render_retrieval_settings()
            use_qem = self._render_query_enhancement_toggle()
            
            # Show API token status for rerankers
            self._show_reranker_token_status(reranker_type)
            
            # Embedding controls
            self._render_embedding_controls(on_embedding_clicked)
            
            st.markdown("---")
            st.markdown("<div class='sidebar-footer'>Welcome back</div>", unsafe_allow_html=True)
        
        return {
            "backend_mode": backend_mode,
            "embedder_type": embedder_type,
            "reranker_type": reranker_type,
            "top_k_embed": top_k_embed,
            "top_k_rerank": top_k_rerank,
            "use_query_enhancement": use_qem,
        }
    
    def _render_menu_buttons(self) -> None:
        """Render New Chat and Clear Cache buttons"""
        st.markdown("### Menu")
        
        if st.button("New Chat"):
            st.session_state["messages"] = []
            st.session_state["is_generating"] = False
            st.session_state["pending_prompt"] = None
            st.session_state["last_sources"] = []
            st.rerun()
        
        if st.button("Clear Cache"):
            st.session_state.clear()
            st.rerun()
        
        if st.button("🗑️ Clear Data", type="secondary", help="Xóa tất cả dữ liệu đã xử lý (chunks, embeddings, vectors, BM25, logs)"):
            self._clear_data_directories()
    
    def _clear_data_directories(self) -> None:
        """Clear all processed data directories and files"""
        import shutil
        
        data_dirs_to_clear = [
            "bm25_index",
            "cache", 
            "chunks",
            "embeddings",
            "logs",
            "metadata",
            "vectors"
        ]
        
        files_to_remove = [
            "batch_summary_*.json"
        ]
        
        base_data_dir = Path("data")
        
        # Clear directories
        for dir_name in data_dirs_to_clear:
            dir_path = base_data_dir / dir_name
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    st.success(f"✅ Đã xóa thư mục: {dir_name}/")
                except Exception as e:
                    st.error(f"❌ Lỗi xóa {dir_name}: {e}")
            else:
                st.info(f"ℹ️ Thư mục {dir_name}/ không tồn tại")
        
        # Clear specific files
        for file_pattern in files_to_remove:
            try:
                for file_path in base_data_dir.glob(file_pattern):
                    file_path.unlink()
                    st.success(f"✅ Đã xóa file: {file_path.name}")
            except Exception as e:
                st.error(f"❌ Lỗi xóa {file_pattern}: {e}")
        
        st.success("🎉 Đã hoàn thành việc xóa dữ liệu!")
        st.rerun()
    
    def _render_file_upload(self) -> None:
        """Render file upload control"""
        st.markdown("### Upload file")
        uploaded_file = st.file_uploader(
            "Chọn file để tải lên",
            type=["pdf", "docx", "txt"]
        )
        
        if uploaded_file is not None:
            os.makedirs(self.data_dir, exist_ok=True)
            save_path = self.data_dir / uploaded_file.name
            
            with open(save_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)
            
            st.success(f"Đã lưu file: {uploaded_file.name}")
    
    def _render_backend_selection(self) -> str:
        """Render LLM backend selection"""
        backend_options = ["gemini", "lmstudio"]
        
        if "backend_mode" not in st.session_state:
            try:
                from llm.config_loader import ui_default_backend
                default = ui_default_backend()
                st.session_state["backend_mode"] = (
                    default if default in backend_options else backend_options[0]
                )
            except Exception:
                st.session_state["backend_mode"] = backend_options[0]
        
        st.radio(
            "Response source",
            backend_options,
            key="backend_mode",
            help="Chọn nguồn trả lời cho chatbot",
            format_func=lambda x: "Gemini API" if x == "gemini" else "LM Studio Local"
        )
        
        return st.session_state["backend_mode"]
    
    def _render_embedder_selection(self) -> str:
        """Render embedding model selection"""
        st.markdown("---")
        embedding_options = ["ollama", "huggingface_local", "huggingface_api"]
        
        if "embedder_type" not in st.session_state:
            st.session_state["embedder_type"] = "huggingface_local"
        
        st.radio(
            "Embedding Model",
            embedding_options,
            key="embedder_type",
            help="Chọn loại embedder cho retrieval",
            format_func=lambda x: {
                "ollama": "Ollama (Gemma/BGE-M3)",
                "huggingface_local": "HF Local (BGE-M3 1024-dim)",
                "huggingface_api": "HF API (E5-Large 1024-dim)"
            }.get(x, x)
        )
        
        return st.session_state["embedder_type"]
    
    def _render_reranker_selection(self) -> str:
        """Render reranker model selection"""
        st.markdown("---")
        reranker_options = ["none", "bge_m3_hf_local", "bge_m3_ollama", "bge_m3_hf_api"]
        
        if "reranker_type" not in st.session_state:
            st.session_state["reranker_type"] = "bge_m3_hf_local"
        
        st.radio(
            "Reranker Model",
            reranker_options,
            key="reranker_type",
            help="Chọn loại reranker để sắp xếp lại kết quả",
            format_func=lambda x: {
                "none": "No Reranking",
                "bge_m3_hf_local": "BGE v2-m3 Local (HF)",
                "bge_m3_ollama": "BGE-M3 Ollama",
                "bge_m3_hf_api": "Sentence-Transformers HF API",
                "cohere": "Cohere API",
                "jina": "Jina API"
            }.get(x, x)
        )
        
        return st.session_state["reranker_type"]
    
    def _render_retrieval_settings(self) -> tuple:
        """Render retrieval settings (top_k sliders)"""
        st.markdown("---")
        st.markdown("### Retrieval Settings")
        
        if "top_k_embed" not in st.session_state:
            st.session_state["top_k_embed"] = 10
        
        st.slider(
            "Top K Embedding Retrieval",
            min_value=5,
            max_value=50,
            value=st.session_state.get("top_k_embed", 10),
            step=5,
            key="top_k_embed",
            help="Số lượng kết quả từ embedding search (trước reranking)"
        )
        
        if "top_k_rerank" not in st.session_state:
            st.session_state["top_k_rerank"] = 5
        
        st.slider(
            "Top K Reranking",
            min_value=1,
            max_value=20,
            value=st.session_state.get("top_k_rerank", 5),
            step=1,
            key="top_k_rerank",
            help="Số lượng kết quả cuối cùng sau reranking"
        )
        
        return st.session_state["top_k_embed"], st.session_state["top_k_rerank"]
    
    def _render_query_enhancement_toggle(self) -> bool:
        """Render query enhancement toggle"""
        st.markdown("---")
        
        if "use_query_enhancement" not in st.session_state:
            st.session_state["use_query_enhancement"] = True
        
        st.checkbox(
            "🔍 Query Enhancement (QEM)",
            value=st.session_state.get("use_query_enhancement", True),
            key="use_query_enhancement",
            help="Tự động mở rộng query để cải thiện kết quả tìm kiếm"
        )
        
        return st.session_state["use_query_enhancement"]
    
    def _show_reranker_token_status(self, reranker_type: str) -> None:
        """Show API token status for API-based rerankers"""
        if reranker_type in ["bge_m3_hf_api", "cohere", "jina"]:
            try:
                if reranker_type == "bge_m3_hf_api":
                    from embedders.providers.huggingface.token_manager import get_hf_token
                    token = get_hf_token()
                    service_name = "HuggingFace"
                elif reranker_type == "cohere":
                    token = os.getenv("COHERE_API_KEY") or os.getenv("COHERE_TOKEN")
                    service_name = "Cohere"
                elif reranker_type == "jina":
                    token = os.getenv("JINA_API_KEY") or os.getenv("JINA_TOKEN")
                    service_name = "Jina"
                
                if token:
                    st.success(f"✅ {service_name} API token: OK")
                else:
                    st.warning(f"⚠️ {service_name} token chưa thiết lập")
            except Exception as e:
                st.error(f"⚠️ Lỗi token: {e}")
    
    def _render_embedding_controls(
        self,
        on_embedding_clicked: Optional[Callable] = None
    ) -> None:
        """Render embedding controls and button"""
        st.markdown("### Embedding Controls")
        
        # Show PDF count
        pdf_dir = Path("data/pdf")
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            pdf_count = len(pdf_files)
            if pdf_count > 0:
                st.info(f"📁 {pdf_count} file PDF sẵn sàng")
            else:
                st.warning("⚠️ Không có PDF nào trong data/pdf/")
        else:
            st.error("❌ Thư mục data/pdf/ không tồn tại")
            st.info("Tạo thư mục: `mkdir data/pdf` và đặt PDF vào đó")
        
        # Run Embedding button
        if st.button("🚀 Run Embedding", type="primary", help="Chạy embedding cho tất cả PDF"):
            if on_embedding_clicked:
                on_embedding_clicked()

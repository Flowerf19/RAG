import sys
import os
from pathlib import Path
import streamlit as st
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.dirname(__file__))  # Add current directory


# Import chat_handler và LLM clients
# Handle both direct execution and module import
try:
    # When run as module
    from .chat_handler import build_messages
    from .LLM_API import call_gemini
    from .LLM_LOCAL import call_lmstudio
    from .config_loader import ui_default_backend, paths_data_dir
except ImportError:
    # When run directly as script
    from chat_handler import build_messages
    from LLM_API import call_gemini
    from LLM_LOCAL import call_lmstudio
    from config_loader import ui_default_backend, paths_data_dir

# Import pipeline_qa
try:
    from pipeline.backend_connector import fetch_retrieval
except ImportError:
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from pipeline.backend_connector import fetch_retrieval
    except ImportError:
        raise


# === PAGE CONFIG ===
st.set_page_config(page_title="AI Chatbot", page_icon=":speech_balloon:", layout="wide")

# === GLOBAL STYLES ===
css_path = Path(__file__).with_name("chat_styles.css")
st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.markdown("### Menu")
    
    # New Chat button
    if st.button("New Chat"):
        st.session_state["messages"] = []
        st.session_state["is_generating"] = False
        st.session_state["pending_prompt"] = None
        st.session_state["last_sources"] = []
        st.rerun()
    
    # Clear Cache button
    if st.button("Clear Cache"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    
    # === UPLOAD FILE ===
    st.markdown("### Upload file")
    uploaded_file = st.file_uploader("Chọn file để tải lên", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        save_dir_path = paths_data_dir()
        os.makedirs(save_dir_path, exist_ok=True)
        save_path = os.path.join(str(save_dir_path), uploaded_file.name)

        # Lưu file về thư mục data
        with open(save_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        st.success(f"Đã lưu file: {uploaded_file.name}")

    st.markdown("---")
    st.markdown("<div style='flex: 1;'></div>", unsafe_allow_html=True)  # Spacer

    # === BACKEND SELECTION ===
    backend_options = ["gemini", "lmstudio"]
    if "backend_mode" not in st.session_state:
        default_backend = ui_default_backend()
        st.session_state["backend_mode"] = (
            default_backend if default_backend in backend_options else backend_options[0]
        )

    st.markdown("<div class='sidebar-footer'>", unsafe_allow_html=True)
    st.radio(
        "Response source",
        backend_options,
        key="backend_mode",
        help="Chọn nguồn trả lời cho chatbot",
        format_func=lambda x: "Gemini API" if x == "gemini" else "LM Studio Local"
    )
    
    # Embedding Model Selection
    st.markdown("---")
    embedding_options = ["ollama", "huggingface_local", "huggingface_api"]
    if "embedder_type" not in st.session_state:
        st.session_state["embedder_type"] = "huggingface_local"  # Default to BGE-M3 local
    
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
    
    # Reranker Model Selection
    st.markdown("---")
    reranker_options = ["none", "bge_local", "bge_m3_ollama", "bge_m3_hf_local", "bge_m3_hf_api"]
    if "reranker_type" not in st.session_state:
        st.session_state["reranker_type"] = "bge_m3_hf_local"  # Default to BGE-M3 HF local
    
    st.radio(
        "Reranker Model",
        reranker_options,
        key="reranker_type",
        help="Chọn loại reranker để sắp xếp lại kết quả",
        format_func=lambda x: {
            "none": "No Reranking",
            "bge_local": "BGE v2-m3 Local",
            "bge_m3_ollama": "BGE-M3 Ollama",
            "bge_m3_hf_local": "BGE-M3 HF Local",
            "bge_m3_hf_api": "Sentence-Transformers HF API",
            "cohere": "Cohere API",
            "jina": "Jina API"
        }.get(x, x)
    )
    
    # Top K Settings
    st.markdown("---")
    st.markdown("### Retrieval Settings")
    
    # Top K for Embedding Retrieval
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
    
    # Top K for Reranking
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

    # Query Enhancement Toggle
    st.markdown("---")
    if "use_query_enhancement" not in st.session_state:
        st.session_state["use_query_enhancement"] = True  # Default to enabled

    st.checkbox(
        "🔍 Query Enhancement (QEM)",
        value=st.session_state.get("use_query_enhancement", True),
        key="use_query_enhancement",
        help="Tự động mở rộng query để cải thiện kết quả tìm kiếm (ví dụ: 'quản lý rủi ro' → 'quản trị rủi ro', 'kiểm soát rủi ro', ...)"
    )

    
    # API token status for API-based rerankers
    reranker_type = st.session_state.get("reranker_type", "bge_m3_hf_local")
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
    
    # === EMBEDDING CONTROLS ===
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
        try:
            # Initialize pipeline based on selected embedder
            embedder_type = st.session_state.get("embedder_type", "huggingface_local")
            
            with st.spinner(f"Đang chạy embedding với {embedder_type}..."):
                from pipeline.rag_pipeline import RAGPipeline
                from embedders.embedder_type import EmbedderType
                
                # Map UI selection to pipeline parameters
                if embedder_type == "huggingface_local":
                    pipeline = RAGPipeline(
                        output_dir="data",
                        pdf_dir="data/pdf",
                        embedder_type=EmbedderType.HUGGINGFACE,
                        hf_use_api=False
                    )
                elif embedder_type == "huggingface_api":
                    pipeline = RAGPipeline(
                        output_dir="data",
                        pdf_dir="data/pdf",
                        embedder_type=EmbedderType.HUGGINGFACE,
                        hf_use_api=True
                    )
                else:  # ollama
                    pipeline = RAGPipeline(
                        output_dir="data",
                        pdf_dir="data/pdf",
                        embedder_type=EmbedderType.OLLAMA
                    )
                
                # Process all PDFs in directory
                pdf_dir = Path("data/pdf")
                results = pipeline.process_directory(pdf_dir)
                
                if results:
                    st.success(f"✅ Đã xử lý {len(results)} file PDF!")
                    st.balloons()
                    
                    # Show results summary
                    with st.expander("📊 Kết quả xử lý"):
                        for result in results:
                            # Use correct keys from pipeline.process_pdf return dict
                            file_name = result.get('file_name', 'Unknown')
                            chunks = result.get('chunks', 0)
                            embeddings = result.get('embeddings', 0)
                            
                            st.write(f"📄 **{file_name}**")
                            st.write(f"   - Chunks: {chunks}")
                            st.write(f"   - Embeddings: {embeddings}")
                else:
                    st.warning("⚠️ Không có PDF nào được xử lý")
                    
        except Exception as e:
            st.error(f"❌ Lỗi embedding: {str(e)}")
            with st.expander("Chi tiết lỗi"):
                import traceback
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    st.markdown("<div class='sidebar-footer'>", unsafe_allow_html=True)
    st.markdown("Welcome back", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Dùng biến backend thống nhất
backend = st.session_state["backend_mode"]

# === SESSION STATE INIT ===
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # OpenAI format: [{"role": "user"/"assistant", "content": "..."}]
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None
if "last_sources" not in st.session_state:
    st.session_state["last_sources"] = []
if "last_retrieval_info" not in st.session_state:
    st.session_state["last_retrieval_info"] = {}
if "last_queries" not in st.session_state:
    st.session_state["last_queries"] = []

# === CHAT HEADER ===
st.markdown("<div class='chat-header'>Chat Window</div>", unsafe_allow_html=True)

# === CHAT LOG RENDER ===
chat_html_parts = ["<div class='chat-log'>"]
for msg in st.session_state["messages"]:
    # Normalize role for display
    role = msg.get("role", "user")
    if role == "assistant":
        role = "bot"  # UI dùng "bot" để styling
    
    bubble = (
        f"<div class='chat-row {role}'><div class='chat-bubble {role}'>"
        f"{msg.get('content', '')}"
        "</div></div>"
    )
    chat_html_parts.append(bubble)

if st.session_state.get("is_generating") and st.session_state.get("pending_prompt"):
    chat_html_parts.append(
        "<div class='chat-row bot'><div class='chat-bubble bot'><span class='typing'>"
        "<span></span><span></span><span></span></span></div></div>"
    )

chat_html_parts.append("</div>")
st.markdown("".join(chat_html_parts), unsafe_allow_html=True)

# === RETRIEVAL SOURCES (UI) ===
sources = st.session_state.get("last_sources", [])
retrieval_info = st.session_state.get("last_retrieval_info", {})

if sources or retrieval_info:
    st.markdown("### Nguồn tham khảo")
    
    # Display retrieval info if available
    if retrieval_info:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Retrieved", retrieval_info.get("total_retrieved", 0))
        with col2:
            st.metric("Final Count", retrieval_info.get("final_count", 0))
        with col3:
            reranked_status = "✅ Yes" if retrieval_info.get("reranked", False) else "❌ No"
            st.metric("Reranked", reranked_status)
        with col4:
            reranker = retrieval_info.get("reranker", "none")
            st.metric("Reranker", reranker[:10] + "..." if len(reranker) > 10 else reranker)
        with col5:
            qem_status = "✅ Yes" if retrieval_info.get("query_enhanced", False) else "❌ No"
            st.metric("QEM", qem_status)

    # Display expanded queries if QEM was used
    if retrieval_info.get("query_enhanced", False):
        queries = st.session_state.get("last_queries", [])
        if len(queries) > 1:  # Only show if there are multiple queries (meaning expansion happened)
            with st.expander("🔍 Expanded Queries (QEM)"):
                st.write("Query gốc đã được mở rộng thành:")
                for i, query in enumerate(queries, 1):
                    st.write(f"{i}. {query}")
    
    # Display sources
    for i, src in enumerate(sources, 1):
        file_name = src.get("file_name", "?")
        page = src.get("page_number", "?")
        
        # Get different score types
        hybrid_score = src.get("similarity_score", 0.0)
        vector_sim = src.get("vector_similarity")
        rerank_score = src.get("rerank_score")
        
        try:
            hybrid_score = float(hybrid_score)
        except (ValueError, TypeError):
            hybrid_score = 0.0
        
        # Build score display text
        score_parts = []
        
        # Show vector similarity (cosine) if available
        if vector_sim is not None:
            try:
                vector_sim = float(vector_sim)
                score_parts.append(f"Vec: {vector_sim:.4f}")
            except (ValueError, TypeError):
                pass
        
        # Show hybrid score (z-score weighted)
        score_parts.append(f"Hybrid: {hybrid_score:.4f}")
        
        # Show rerank score if available
        if rerank_score is not None:
            try:
                rerank_score = float(rerank_score)
                score_parts.append(f"Rerank: {rerank_score:.4f}")
            except (ValueError, TypeError):
                pass
        
        score_text = " | ".join(score_parts)
        
        text = src.get("snippet", "") or ""  # Sử dụng 'snippet' thay vì 'text'
        snippet = text if len(text) <= 500 else text[:500] + "..."
        st.markdown(f"- [{i}] {file_name} - trang {page} ({score_text})")
        with st.expander(f"Xem trích đoạn {i}"):
            if snippet.strip():
                st.markdown(snippet)
            else:
                st.write("Không có nội dung trích đoạn")
else:
    st.info("Chưa có nguồn tham khảo nào được tìm thấy. Hãy đặt câu hỏi để hệ thống tìm kiếm tài liệu liên quan.")

# === BACKEND CALL ===
def ask_backend(prompt_text: str) -> str:
    """
    Xử lý request tới LLM backend
    
    Args:
        prompt_text: User query
    
    Returns:
        Response từ LLM
    """
    try:
        # TODO: Khi có retrieval system, lấy context ở đây
        context = ""  # Tạm thời để trống
        
        # Build messages bằng chat_handler
        # Lấy context từ Retrieval (nếu có) và lưu nguồn để hiển thị.
        try:
            embedder_type = st.session_state.get("embedder_type", "huggingface_local")
            reranker_type = st.session_state.get("reranker_type", "bge_m3_hf_local")
            top_k_rerank = st.session_state.get("top_k_rerank", 5)
            use_query_enhancement = st.session_state.get("use_query_enhancement", True)
            
            # Collect API tokens for rerankers
            api_tokens = {}
            if reranker_type == "bge_m3_hf_api":
                try:
                    from embedders.providers.huggingface.token_manager import get_hf_token
                    token = get_hf_token()
                    api_tokens["hf"] = token
                    if token:
                        st.info(f"✅ HF token loaded: {'***' + token[-4:]}")
                    else:
                        st.error("❌ HF token not found!")
                except Exception as e:
                    st.error(f"❌ Failed to get HF token: {e}")
            elif reranker_type == "cohere":
                token = os.getenv("COHERE_API_KEY") or os.getenv("COHERE_TOKEN")
                api_tokens["cohere"] = token
                if not token:
                    st.warning("⚠️ Cohere token not found in environment")
            elif reranker_type == "jina":
                token = os.getenv("JINA_API_KEY") or os.getenv("JINA_TOKEN")
                api_tokens["jina"] = token
                if not token:
                    st.warning("⚠️ Jina token not found in environment")
            
            ret = fetch_retrieval(
                prompt_text, 
                top_k=top_k_rerank,  # Use final top_k for simplified API
                max_chars=8000, 
                embedder_type=embedder_type, 
                reranker_type=reranker_type,
                use_query_enhancement=use_query_enhancement,
                api_tokens=api_tokens
            )
            context = ret.get("context", "") or ""
            st.session_state["last_sources"] = ret.get("sources", [])
            st.session_state["last_retrieval_info"] = ret.get("retrieval_info", {})
            st.session_state["last_queries"] = ret.get("queries", [])
        except Exception as e:
            st.error(f"Lỗi retrieval: {e}")
            context = ""
            st.session_state["last_sources"] = []
            st.session_state["last_retrieval_info"] = {}

        messages = build_messages(
            query=prompt_text,
            context=context,
            history=st.session_state["messages"]
        )
        
        # Gọi LLM tương ứng
        if backend == "gemini":
            reply = call_gemini(messages)
        else:  # lmstudio
            reply = call_lmstudio(messages)
        
        return reply
    
    except Exception as e:
        return f"[Error] {e}"

# === CHAT INPUT ===
prompt = st.chat_input("Type a new message here", disabled=st.session_state.get("is_generating", False))

if prompt and not st.session_state.get("is_generating", False):
    # Thêm user message vào history (OpenAI format)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["pending_prompt"] = prompt
    st.session_state["is_generating"] = True
    st.rerun()

# === GENERATE RESPONSE ===
if st.session_state.get("is_generating") and st.session_state.get("pending_prompt"):
    with st.spinner("Assistant is typing..."):
        reply = ask_backend(st.session_state["pending_prompt"])

    # Thêm assistant response vào history (OpenAI format)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.session_state["pending_prompt"] = None
    st.session_state["is_generating"] = False
    st.rerun()
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.dirname(__file__))  # Add current directory

# Debug: Print paths
print("Current working directory:", os.getcwd())
print("Script directory:", os.path.dirname(__file__))
print("Python path:", sys.path)

from pathlib import Path
import streamlit as st
import os
import shutil

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
    print("Successfully imported fetch_retrieval")
except ImportError as e:
    print(f"Failed to import fetch_retrieval: {e}")
    print("Trying alternative import...")
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from pipeline.backend_connector import fetch_retrieval
        print("Successfully imported fetch_retrieval with alternative path")
    except ImportError as e2:
        print(f"Still failed: {e2}")
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
    
    st.button("Recent Chats")
    st.button("Rephrase text...")
    st.button("Fix this code...")
    st.button("Sample Copy for...")

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
    
    # HuggingFace API token status
    if st.session_state.get("embedder_type") == "huggingface_api":
        try:
            from embedders.providers.huggingface.token_manager import get_hf_token
            token = get_hf_token()
            
            if token:
                st.success("✅ HuggingFace API token: OK")
            else:
                st.warning("⚠️ HuggingFace token chưa thiết lập")
        except Exception as e:
            st.error(f"⚠️ Lỗi token: {e}")
    
    st.markdown("---")
    
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
if sources:
    st.markdown("### Nguồn tham khảo")
    for i, src in enumerate(sources, 1):
        file_name = src.get("file_name", "?")
        page = src.get("page_number", "?")
        try:
            score = float(src.get("similarity_score", 0.0))
        except Exception:
            score = 0.0
        text = src.get("snippet", "") or ""  # Sử dụng 'snippet' thay vì 'text'
        snippet = text if len(text) <= 500 else text[:500] + "..."
        st.markdown(f"- [{i}] {file_name} - trang {page} (điểm {score:.3f})")
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
            ret = fetch_retrieval(prompt_text, top_k=10, max_chars=8000, embedder_type=embedder_type)
            context = ret.get("context", "") or ""
            st.session_state["last_sources"] = ret.get("sources", [])
        except Exception as e:
            st.error(f"Lỗi retrieval: {e}")
            context = ""
            st.session_state["last_sources"] = []

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
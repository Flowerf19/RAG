import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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

from RAG_system.pipeline import fetch_retrieval


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
if st.session_state.get("last_sources"):
    st.markdown("### Nguồn tham khảo")
    for i, src in enumerate(st.session_state["last_sources"], 1):
        file_name = src.get("file_name", "?")
        page = src.get("page_number", "?")
        try:
            score = float(src.get("similarity_score", 0.0))
        except Exception:
            score = 0.0
        text = src.get("text", "") or ""
        snippet = text if len(text) <= 300 else text[:300] + "..."
        st.markdown(f"- [{i}] {file_name} - trang {page} (điểm {score:.3f})")
        with st.expander(f"Xem trích đoạn {i}"):
            st.markdown(snippet)

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
            ret = fetch_retrieval(prompt_text, top_k=5, max_chars=4000)
            context = ret.get("context", "") or ""
            st.session_state["last_sources"] = ret.get("sources", [])
        except Exception:
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

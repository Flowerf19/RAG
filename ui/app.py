"""
RAG Chatbot UI - Main Streamlit Application
============================================
Modular UI following OOP principles with separation of concerns

Architecture:
- UI Layer (this file): Streamlit app orchestration
- Components: Reusable UI elements (ChatDisplay, Sidebar, SourceDisplay)
- LLM Layer: LLM client abstraction (llm/)
- Backend Layer: Retrieval service (pipeline/backend_connector)
"""
import sys
import os
from pathlib import Path
import warnings

import streamlit as st

# Add project root to path FIRST before any imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.client_factory import LLMClientFactory
from llm.chat_handler import build_messages
from pipeline.backend_connector import fetch_retrieval

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='paddle')
warnings.filterwarnings('ignore', category=UserWarning, module='google')

# Import components
try:
    from ui.components import ChatDisplay, SourceDisplay, Sidebar
except ImportError:
    # Fallback for direct execution
    from components import ChatDisplay, SourceDisplay, Sidebar



# Import config
try:
    from llm.config_loader import paths_data_dir
except ImportError:
    # Fallback
    def paths_data_dir():
        return Path("data/pdf")


class RAGChatApp:
    """
    Main application class for RAG Chatbot
    Orchestrates UI components, LLM clients, and backend services
    """
    
    def __init__(self):
        """Initialize RAG Chat Application"""
        self._init_page_config()
        self._init_session_state()
        self._load_styles()
        
        # Initialize components
        self.chat_display = ChatDisplay()
        self.source_display = SourceDisplay()
        self.sidebar = Sidebar(data_dir=paths_data_dir())
    
    def _init_page_config(self) -> None:
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="AI Chatbot",
            page_icon=":speech_balloon:",
            layout="wide"
        )
    
    def _init_session_state(self) -> None:
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        
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
    
    def _load_styles(self) -> None:
        """Load CSS styles"""
        css_path = Path(__file__).parent / "styles" / "chat_styles.css"
        if css_path.exists():
            st.markdown(
                f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True
            )
    
    def run(self) -> None:
        """Main application loop"""
        # Render sidebar and get settings
        settings = self.sidebar.render(
            on_embedding_clicked=self._handle_embedding_click
        )
        
        # Render chat header
        ChatDisplay.render_header("CAI ICON NAY ONG CHIEN LAM MAT 1 ngay nen khong duoc sua linh tinh")
        
        # Render chat messages
        self.chat_display.render(
            messages=st.session_state["messages"],
            is_generating=st.session_state.get("is_generating", False),
            pending_prompt=st.session_state.get("pending_prompt")
        )
        
        # Render sources
        self.source_display.render(
            sources=st.session_state.get("last_sources", []),
            retrieval_info=st.session_state.get("last_retrieval_info", {}),
            expanded_queries=st.session_state.get("last_queries", [])
        )
        
        # Handle chat input
        self._handle_chat_input(settings)
        
        # Generate response if pending
        if st.session_state.get("is_generating") and st.session_state.get("pending_prompt"):
            self._generate_response(settings)
    
    def _handle_chat_input(self, settings: dict) -> None:
        """
        Handle user chat input
        
        Args:
            settings: Current app settings from sidebar
        """
        prompt = st.chat_input(
            "Type a new message here",
            disabled=st.session_state.get("is_generating", False)
        )
        
        if prompt and not st.session_state.get("is_generating", False):
            # Add user message to history
            st.session_state["messages"].append({
                "role": "user",
                "content": prompt
            })
            st.session_state["pending_prompt"] = prompt
            st.session_state["is_generating"] = True
            st.rerun()
    
    def _generate_response(self, settings: dict) -> None:
        """
        Generate LLM response for pending prompt
        
        Args:
            settings: Current app settings from sidebar
        """
        status_container = st.empty()
        
        with status_container.status("🔄 Processing your question...", expanded=True) as status:
            try:
                st.write("📝 Step 1: Retrieving relevant documents...")
                
                # Get context from retrieval
                context = self._fetch_context(
                    st.session_state["pending_prompt"],
                    settings
                )
                
                st.write(f"🤖 **Step 2:** Generating answer with **{settings['backend_mode'].upper()}**...")
                
                # Debug: Show context preview
                with st.expander("📝 Context Preview", expanded=False):
                    preview = context[:500] + "..." if len(context) > 500 else context
                    st.text(preview)
                    st.caption(f"Total context length: {len(context)} characters")
                
                # Build messages
                messages = build_messages(
                    query=st.session_state["pending_prompt"],
                    context=context,
                    history=st.session_state["messages"]
                )
                
                # Debug: Show message count
                st.write(f"💬 Built {len(messages)} messages for LLM (query + context + history)")
                
                # Get LLM response
                reply = self._call_llm(messages, settings["backend_mode"])
                
                st.write(f"✅ **Complete!** Generated {len(reply)} chars")
                status.update(label="✅ Done!", state="complete")
            
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                reply = f"[Error] {e}"
                status.update(label="❌ Failed", state="error")
        
        # Add assistant response to history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": reply
        })
        st.session_state["pending_prompt"] = None
        st.session_state["is_generating"] = False
        st.rerun()
    
    def _fetch_context(self, query: str, settings: dict) -> str:
        """
        Fetch context from retrieval system
        
        Args:
            query: User query
            settings: App settings
        
        Returns:
            Context string from retrieval
        """
        try:
            embedder_type = settings["embedder_type"]
            reranker_type = settings["reranker_type"]
            top_k = settings["top_k_rerank"]
            use_qem = settings["use_query_enhancement"]
            
            st.write(f"⚙️ **Config:** Embedder={embedder_type} | Reranker={reranker_type} | Top-K={top_k} | QEM={'ON' if use_qem else 'OFF'}")
            
            # Collect API tokens for rerankers
            api_tokens = self._collect_api_tokens(reranker_type)
            
            st.write("🔍 **Initializing retrieval system** (first time may take 30-60s to load models)...")
            
            # Debug: Show query
            with st.expander("🔎 Query Details", expanded=False):
                st.code(query, language="text")
                if use_qem:
                    st.caption("✨ Query Enhancement Module (QEM) enabled - will generate expanded queries")
            
            # Call retrieval
            ret = fetch_retrieval(
                query,
                top_k=top_k,
                max_chars=8000,
                embedder_type=embedder_type,
                reranker_type=reranker_type,
                use_query_enhancement=use_qem,
                api_tokens=api_tokens
            )
            
            # Update session state with results
            context = ret.get("context", "") or ""
            sources = ret.get("sources", [])
            retrieval_info = ret.get("retrieval_info", {})
            queries = ret.get("queries", [])
            
            st.session_state["last_sources"] = sources
            st.session_state["last_retrieval_info"] = retrieval_info
            st.session_state["last_queries"] = queries
            
            # Show retrieval stats
            st.write(f"✅ **Retrieved:** {len(sources)} documents | Context: {len(context)} chars")
            
            # Debug: Show retrieval details
            with st.expander("📊 Retrieval Details", expanded=False):
                if queries and len(queries) > 1:
                    st.write(f"**Expanded Queries:** {len(queries)} queries generated")
                    for i, q in enumerate(queries[:3], 1):
                        st.caption(f"{i}. {q}")
                
                if retrieval_info:
                    st.write("**Retrieval Info:**")
                    for key, value in retrieval_info.items():
                        if isinstance(value, (int, float, str)):
                            st.caption(f"• {key}: {value}")
                
                if sources:
                    st.write("**Top Sources:**")
                    for i, src in enumerate(sources[:3], 1):
                        doc_name = src.get('metadata', {}).get('source', 'Unknown')
                        score = src.get('score', 0)
                        st.caption(f"{i}. {doc_name} (score: {score:.3f})")
            
            return context
        
        except Exception as e:
            st.error(f"Lỗi retrieval: {e}")
            import traceback
            st.code(traceback.format_exc())
            
            st.session_state["last_sources"] = []
            st.session_state["last_retrieval_info"] = {}
            
            return ""
    
    def _collect_api_tokens(self, reranker_type: str) -> dict:
        """
        Collect API tokens for rerankers
        
        Args:
            reranker_type: Type of reranker
        
        Returns:
            Dict with API tokens
        """
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
        
        return api_tokens
    
    def _call_llm(self, messages: list, backend_mode: str) -> str:
        """
        Call LLM to generate response
        
        Args:
            messages: OpenAI format messages
            backend_mode: "gemini" or "lmstudio"
        
        Returns:
            Generated response text
        """
        try:
            # Debug: Show LLM call info
            with st.expander("🤖 LLM Call Details", expanded=False):
                st.write(f"**Backend:** {backend_mode}")
                st.write(f"**Messages:** {len(messages)} total")
                
                # Show message breakdown
                msg_types = {}
                total_chars = 0
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    msg_types[role] = msg_types.get(role, 0) + 1
                    total_chars += len(content)
                
                st.write("**Message breakdown:**")
                for role, count in msg_types.items():
                    st.caption(f"• {role}: {count} message(s)")
                st.caption(f"• Total input: ~{total_chars:,} chars")
                
                # Show last user message
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    st.write("**Last user message:**")
                    last_msg = user_msgs[-1].get("content", "")
                    preview = last_msg[:200] + "..." if len(last_msg) > 200 else last_msg
                    st.text(preview)
            
            st.write(f"⏳ Calling {backend_mode.upper()} API...")
            
            # Create client using factory
            client = LLMClientFactory.create_from_string(backend_mode)
            
            # Generate response
            response = client.generate(messages)
            
            st.write(f"✅ LLM response received: {len(response)} chars")
            
            return response
        
        except Exception as e:
            st.error(f"❌ LLM Error: {e}")
            import traceback
            with st.expander("🐛 Error Traceback", expanded=True):
                st.code(traceback.format_exc())
            raise RuntimeError(f"LLM error: {e}")
    
    def _handle_embedding_click(self) -> None:
        """Handle embedding button click"""
        try:
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
                
                # Process all PDFs
                pdf_dir = Path("data/pdf")
                results = pipeline.process_directory(pdf_dir)
                
                if results:
                    st.success(f"✅ Đã xử lý {len(results)} file PDF!")
                    st.balloons()
                    
                    # Show results summary
                    with st.expander("📊 Kết quả xử lý"):
                        for result in results:
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


def main():
    """Main entry point"""
    app = RAGChatApp()
    app.run()


if __name__ == "__main__":
    main()

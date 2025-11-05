"""
Source Display Component - Displays retrieval sources and metadata
Pure UI component for showing document sources
"""
from typing import List, Dict, Any
import streamlit as st


class SourceDisplay:
    """
    Component for displaying retrieval sources and metadata
    Shows document sources, scores, and chunk content
    """
    
    def render(
        self,
        sources: List[Dict[str, Any]],
        retrieval_info: Dict[str, Any] = None,
        expanded_queries: List[str] = None
    ) -> None:
        """
        Render source information and retrieval metadata
        
        Args:
            sources: List of source documents with metadata
            retrieval_info: Retrieval statistics (total, reranked, etc.)
            expanded_queries: Expanded queries from QEM (if used)
        """
        if not sources and not retrieval_info:
            st.info(
                "Chưa có nguồn tham khảo nào được tìm thấy. "
                "Hãy đặt câu hỏi để hệ thống tìm kiếm tài liệu liên quan."
            )
            return
        
        st.markdown("### Source Information")
        
        # Render retrieval statistics
        if retrieval_info:
            self._render_retrieval_stats(retrieval_info)
        
        # Render expanded queries if QEM was used
        if retrieval_info and retrieval_info.get("query_enhanced") and expanded_queries:
            if len(expanded_queries) > 1:
                self._render_expanded_queries(expanded_queries)
        
        # Render source documents
        if sources:
            self._render_sources(sources)
    
    def _render_retrieval_stats(self, retrieval_info: Dict[str, Any]) -> None:
        """
        Render retrieval statistics as metrics
        
        Args:
            retrieval_info: Dict with retrieval metadata
        """
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
            display_reranker = reranker[:10] + "..." if len(reranker) > 10 else reranker
            st.metric("Reranker", display_reranker)
        
        with col5:
            qem_status = "✅ Yes" if retrieval_info.get("query_enhanced", False) else "❌ No"
            st.metric("QEM", qem_status)
    
    def _render_expanded_queries(self, queries: List[str]) -> None:
        """
        Render expanded queries from QEM
        
        Args:
            queries: List of expanded query strings
        """
        with st.expander("🔍 Expanded Queries (QEM)"):
            st.write("Query gốc đã được mở rộng thành:")
            for i, query in enumerate(queries, 1):
                st.write(f"{i}. {query}")
    
    def _render_sources(self, sources: List[Dict[str, Any]]) -> None:
        """
        Render source documents with scores and content
        
        Args:
            sources: List of source dicts
        """
        for i, src in enumerate(sources, 1):
            file_name = src.get("file_name", "?")
            page = src.get("page_number", "?")
            
            # Build score display
            score_text = self._build_score_text(src)
            
            # Get content
            snippet = src.get("snippet", "") or ""
            full_text = src.get("full_text", snippet) or snippet
            
            # Display source header
            st.markdown(f"- [{i}] {file_name} - trang {page} ({score_text})")
            
            # Display content in expander
            with st.expander(f"Xem trích đoạn {i}"):
                if full_text.strip():
                    st.markdown("**Full Chunk Content:**")
                    
                    # Calculate dynamic height
                    estimated_lines = max(10, len(full_text) // 80 + 1)
                    display_height = min(estimated_lines * 25, 1000)
                    
                    st.text_area(
                        f"Full content for source {i}",
                        full_text,
                        height=display_height,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                else:
                    st.write("Không có nội dung trích đoạn")
    
    def _build_score_text(self, src: Dict[str, Any]) -> str:
        """
        Build score display text from source metadata
        
        Args:
            src: Source dict with score information
        
        Returns:
            Formatted score text (e.g., "Vec: 0.85 | Hybrid: 0.92 | Rerank: 0.95")
        """
        score_parts = []
        
        # Vector similarity (cosine)
        vector_sim = src.get("vector_similarity")
        if vector_sim is not None:
            try:
                score_parts.append(f"Vec: {float(vector_sim):.4f}")
            except (ValueError, TypeError):
                pass
        
        # Hybrid score (z-score weighted)
        hybrid_score = src.get("similarity_score", 0.0)
        try:
            score_parts.append(f"Hybrid: {float(hybrid_score):.4f}")
        except (ValueError, TypeError):
            score_parts.append("Hybrid: 0.0000")
        
        # Rerank score
        rerank_score = src.get("rerank_score")
        if rerank_score is not None:
            try:
                score_parts.append(f"Rerank: {float(rerank_score):.4f}")
            except (ValueError, TypeError):
                pass
        
        return " | ".join(score_parts)

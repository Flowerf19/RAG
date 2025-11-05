"""
Chat Display Component - Renders chat history and handles message display
Pure UI component - no business logic
"""
from typing import List, Dict
import streamlit as st


class ChatDisplay:
    """
    Component for displaying chat messages
    Handles rendering of user and assistant messages with proper styling
    """
    
    def __init__(self, css_class_prefix: str = "chat"):
        """
        Initialize chat display component
        
        Args:
            css_class_prefix: CSS class prefix for styling
        """
        self.css_prefix = css_class_prefix
    
    def render(
        self,
        messages: List[Dict[str, str]],
        is_generating: bool = False,
        pending_prompt: str = None
    ) -> None:
        """
        Render chat messages as HTML
        
        Args:
            messages: List of messages [{"role": "user"/"assistant", "content": "..."}]
            is_generating: Whether bot is currently generating
            pending_prompt: Pending prompt being processed
        """
        chat_html_parts = [f"<div class='{self.css_prefix}-log'>"]
        
        # Render existing messages
        for msg in messages:
            role = self._normalize_role_for_display(msg.get("role", "user"))
            content = msg.get("content", "")
            
            bubble = (
                f"<div class='{self.css_prefix}-row {role}'>"
                f"<div class='{self.css_prefix}-bubble {role}'>"
                f"{content}"
                f"</div></div>"
            )
            chat_html_parts.append(bubble)
        
        # Show typing indicator if generating
        if is_generating and pending_prompt:
            chat_html_parts.append(
                f"<div class='{self.css_prefix}-row bot'>"
                f"<div class='{self.css_prefix}-bubble bot'>"
                f"<span class='typing'>"
                f"<span></span><span></span><span></span>"
                f"</span></div></div>"
            )
        
        chat_html_parts.append("</div>")
        
        # Render to Streamlit
        st.markdown("".join(chat_html_parts), unsafe_allow_html=True)
    
    def _normalize_role_for_display(self, role: str) -> str:
        """
        Normalize role for CSS styling
        
        Args:
            role: "user" or "assistant"
        
        Returns:
            "user" or "bot" (for CSS classes)
        """
        if role == "assistant":
            return "bot"
        return "user"
    
    @staticmethod
    def render_header(title: str = "Chat Window") -> None:
        """
        Render chat header
        
        Args:
            title: Header title text
        """
        st.markdown(
            f"<div class='chat-header'>{title}</div>",
            unsafe_allow_html=True
        )

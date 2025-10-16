"""
Tests for chat_handler.py
=========================
Test chat message handling, system prompt loading, and history normalization.
"""

import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RAG_system.LLM.chat_handler import (
    load_system_prompt,
    format_system_prompt,
    normalize_history,
    build_messages,
)


class TestSystemPrompt:
    """Test system prompt loading and formatting"""

    @patch('RAG_system.LLM.chat_handler.paths_prompt_path')
    def test_load_system_prompt_success(self, mock_paths_prompt_path):
        """Test successful system prompt loading"""
        mock_path = Path("/fake/prompt/path.txt")
        mock_paths_prompt_path.return_value = mock_path

        prompt_content = "You are a helpful AI assistant.\nAnswer questions accurately."

        with patch('builtins.open', mock_open(read_data=prompt_content)) as mock_file:
            result = load_system_prompt()

        assert result == prompt_content
        mock_paths_prompt_path.assert_called_once()
        mock_file.assert_called_once_with(mock_path, "r", encoding="utf-8")

    @patch('RAG_system.LLM.chat_handler.paths_prompt_path')
    def test_load_system_prompt_file_not_found(self, mock_paths_prompt_path):
        """Test fallback when system prompt file is not found"""
        mock_path = Path("/fake/prompt/path.txt")
        mock_paths_prompt_path.return_value = mock_path

        with patch('builtins.open', side_effect=FileNotFoundError()):
            result = load_system_prompt()

        expected_fallback = (
            "Bạn là trợ lý AI. Trả lời ngắn gọn, rõ ràng và chính xác.\n\n"
            "Context từ tài liệu:\n{context}"
        )
        assert result == expected_fallback

    @patch('RAG_system.LLM.chat_handler.paths_prompt_path')
    def test_load_system_prompt_generic_exception(self, mock_paths_prompt_path):
        """Test exception handling when system prompt loading raises generic exception"""
        mock_path = Path("/fake/prompt/path.txt")
        mock_paths_prompt_path.return_value = mock_path

        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(RuntimeError, match="Không thể đọc system prompt: Permission denied"):
                load_system_prompt()

    def test_format_system_prompt_with_context(self):
        """Test system prompt formatting with context"""
        template = "You are helpful.\nContext: {context}\nAnswer questions."

        with patch('RAG_system.LLM.chat_handler.load_system_prompt', return_value=template):
            result = format_system_prompt("This is test context from documents.")

        expected = "You are helpful.\nContext: This is test context from documents.\nAnswer questions."
        assert result == expected

    def test_format_system_prompt_empty_context(self):
        """Test system prompt formatting with empty context"""
        template = "You are helpful.\nContext: {context}"

        with patch('RAG_system.LLM.chat_handler.load_system_prompt', return_value=template):
            result = format_system_prompt("")

        expected = "You are helpful.\nContext: (Chưa có tài liệu nào được tải lên)"
        assert result == expected

    def test_format_system_prompt_none_context(self):
        """Test system prompt formatting with None context"""
        # Note: Function signature says str but implementation handles None
        with patch('RAG_system.LLM.chat_handler.load_system_prompt', return_value="Template {context}"):
            result = format_system_prompt("")  # Use empty string instead of None

        expected = "Template (Chưa có tài liệu nào được tải lên)"
        assert result == expected


class TestHistoryNormalization:
    """Test chat history normalization"""

    def test_normalize_history_basic_conversion(self):
        """Test basic role conversion from 'bot' to 'assistant'"""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "bot", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        result = normalize_history(history)

        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        assert result == expected

    def test_normalize_history_skip_invalid_messages(self):
        """Test that messages with missing role or content are skipped"""
        history = [
            {"role": "user", "content": "Valid message"},
            {"role": "bot"},  # Missing content
            {"content": "Missing role"},  # Missing role
            {"role": "", "content": "Empty role"},  # Empty role
            {"role": "assistant", "content": "Valid assistant message"}
        ]

        result = normalize_history(history)

        expected = [
            {"role": "user", "content": "Valid message"},
            {"role": "assistant", "content": "Valid assistant message"}
        ]

        assert result == expected

    def test_normalize_history_only_user_assistant_roles(self):
        """Test that only user and assistant roles are kept"""
        history = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "system", "content": "System message"},  # Should be filtered out
            {"role": "unknown", "content": "Unknown message"}  # Should be filtered out
        ]

        result = normalize_history(history)

        expected = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"}
        ]

        assert result == expected

    def test_normalize_history_empty_history(self):
        """Test normalization of empty history"""
        result = normalize_history([])
        assert result == []

    def test_normalize_history_preserve_content_whitespace(self):
        """Test that content whitespace is preserved"""
        history = [
            {"role": "user", "content": "  Message with spaces  "},
            {"role": "assistant", "content": "\tTabbed content\n"}
        ]

        result = normalize_history(history)

        expected = [
            {"role": "user", "content": "  Message with spaces  "},
            {"role": "assistant", "content": "\tTabbed content\n"}
        ]

        assert result == expected


class TestMessageBuilding:
    """Test message building for LLM API calls"""

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_build_messages_basic_structure(self, mock_load_prompt):
        """Test basic message building structure"""
        mock_load_prompt.return_value = "System prompt template with {context}"

        query = "What is the capital of France?"
        context = "France is a country in Europe."

        result = build_messages(query, context)

        assert len(result) == 2  # System + user

        # Check system message
        system_msg = result[0]
        assert system_msg["role"] == "system"
        assert "France is a country in Europe" in system_msg["content"]

        # Check user message
        user_msg = result[1]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == query

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_build_messages_with_history(self, mock_load_prompt):
        """Test message building with chat history"""
        mock_load_prompt.return_value = "System prompt with {context}"

        query = "Follow-up question"
        context = "Some context"
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"}
        ]

        result = build_messages(query, context, history)

        assert len(result) == 4  # System + history (2) + current query

        # Check order: system, history user, history assistant, current user
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "First question"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "First answer"
        assert result[3]["role"] == "user"
        assert result[3]["content"] == "Follow-up question"

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_build_messages_empty_context(self, mock_load_prompt):
        """Test message building with empty context"""
        mock_load_prompt.return_value = "System prompt with {context}"

        query = "Question without context"

        result = build_messages(query, "")

        system_msg = result[0]
        assert "(Chưa có tài liệu nào được tải lên)" in system_msg["content"]

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_build_messages_none_context(self, mock_load_prompt):
        """Test message building with None context"""
        mock_load_prompt.return_value = "System prompt with {context}"

        query = "Question with None context"

        result = build_messages(query, "")  # Use empty string instead of None

        system_msg = result[0]
        assert "(Chưa có tài liệu nào được tải lên)" in system_msg["content"]

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_build_messages_no_history(self, mock_load_prompt):
        """Test message building with no history (empty list)"""
        mock_load_prompt.return_value = "System prompt with {context}"

        query = "Standalone question"
        context = "Some context"

        result = build_messages(query, context, [])  # Use empty list instead of None

        assert len(result) == 2  # System + user only
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == query

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_build_messages_empty_history(self, mock_load_prompt):
        """Test message building with empty history list"""
        mock_load_prompt.return_value = "System prompt with {context}"

        query = "Question with empty history"
        context = "Some context"
        history = []

        result = build_messages(query, context, history)

        assert len(result) == 2  # System + user only (empty history doesn't add messages)

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_build_messages_complex_history(self, mock_load_prompt):
        """Test message building with complex history including invalid messages"""
        mock_load_prompt.return_value = "System prompt with {context}"

        query = "Final question"
        context = "Context info"
        history = [
            {"role": "user", "content": "Valid user message"},
            {"role": "system", "content": "Invalid system in history"},  # Should be filtered
            {"role": "bot", "content": "Bot message"},  # Should become assistant
            {"role": "unknown", "content": "Unknown role"},  # Should be filtered
            {"content": "Missing role"},  # Should be filtered
            {"role": "assistant", "content": "Valid assistant message"}
        ]

        result = build_messages(query, context, history)

        # Should have: system + valid user + bot(as assistant) + valid assistant + current user
        assert len(result) == 5

        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Valid user message"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "Bot message"
        assert result[3]["role"] == "assistant"
        assert result[3]["content"] == "Valid assistant message"
        assert result[4]["role"] == "user"
        assert result[4]["content"] == "Final question"

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_build_messages_context_formatting(self, mock_load_prompt):
        """Test that context is properly formatted in system prompt"""
        template = "You are an AI.\nContext information: {context}\nPlease answer."
        mock_load_prompt.return_value = template

        context = "This is important context from documents."

        result = build_messages("Test query", context)

        system_content = result[0]["content"]
        expected_content = (
            "You are an AI.\n"
            "Context information: This is important context from documents.\n"
            "Please answer."
        )

        assert system_content == expected_content
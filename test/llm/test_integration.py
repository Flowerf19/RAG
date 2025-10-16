"""
Integration tests for LLM module
=================================
Test end-to-end query functionality with retrieval integration and data sources.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RAG_system.LLM.LLM_API import call_gemini
from RAG_system.LLM.LLM_LOCAL import call_lmstudio
from RAG_system.LLM.chat_handler import build_messages


class TestQueryWorkflow:
    """Test complete query workflow from user input to LLM response"""

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    @patch('RAG_system.LLM.LLM_API.genai')
    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_gemini_query_with_retrieval_context(self, mock_load_prompt, mock_genai, mock_resolve_settings):
        """Test Gemini query workflow with retrieval context"""
        # Mock system prompt
        mock_load_prompt.return_value = "You are a helpful assistant.\nContext: {context}\nAnswer questions."

        # Mock Gemini settings
        mock_resolve_settings.return_value = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash"
        }

        # Mock Gemini API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Based on the provided context, the answer is Paris."
        mock_model.generate_content.return_value = mock_response

        mock_genai.GenerativeModel.return_value = mock_model

        # Simulate retrieval context
        context = "France is a European country. Paris is the capital of France."

        # Build messages as the chat handler would
        messages = build_messages(
            query="What is the capital of France?",
            context=context,
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help you?"}
            ]
        )

        # Call Gemini API
        result = call_gemini(messages)

        # Verify the response
        assert "Paris" in result

        # Verify the API was called with properly formatted messages
        call_args = mock_model.generate_content.call_args
        prompt_text = call_args[0][0]

        # Check that context is included in the prompt
        assert "France is a European country" in prompt_text
        assert "Paris is the capital of France" in prompt_text

        # Check that conversation history is included
        assert "Hello" in prompt_text
        assert "Hi! How can I help you?" in prompt_text

        # Check that current query is included
        assert "What is the capital of France?" in prompt_text

    @patch('RAG_system.LLM.LLM_LOCAL.get_client')
    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_lmstudio_query_with_retrieval_context(self, mock_load_prompt, mock_resolve_settings, mock_get_client):
        """Test LM Studio query workflow with retrieval context"""
        # Mock system prompt
        mock_load_prompt.return_value = "You are a helpful assistant.\nContext: {context}\nAnswer questions."

        # Mock LM Studio settings
        mock_resolve_settings.return_value = {
            "model": "google/gemma-3-4b",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512
        }

        # Mock OpenAI client response
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "According to the documents, the capital is Paris."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_get_client.return_value = mock_client

        # Simulate retrieval context
        context = "France information: Paris is the capital city."

        # Build messages as the chat handler would
        messages = build_messages(
            query="What is France's capital?",
            context=context
        )

        # Call LM Studio API
        result = call_lmstudio(messages)

        # Verify the response
        assert "Paris" in result

        # Verify the API was called with the correct messages
        call_args = mock_client.chat.completions.create.call_args
        api_messages = call_args[1]["messages"]

        # Check message structure
        assert len(api_messages) == 2  # System + user
        assert api_messages[0]["role"] == "system"
        assert "Paris is the capital city" in api_messages[0]["content"]
        assert api_messages[1]["role"] == "user"
        assert api_messages[1]["content"] == "What is France's capital?"


class TestDataSources:
    """Test data source attribution and retrieval integration"""

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_context_includes_source_metadata(self, mock_load_prompt):
        """Test that context includes source information for attribution"""
        # Mock system prompt
        mock_load_prompt.return_value = "You are a helpful assistant.\nContext: {context}\nAnswer questions."

        # Simulate context from retrieval system with source metadata
        context_with_sources = """
        According to the Process Risk Management document (page 15):
        Risk management involves identifying, assessing, and controlling risks.

        From Service Configuration Management (page 8):
        Configuration management ensures consistent deployment.
        """

        messages = build_messages(
            query="What is risk management?",
            context=context_with_sources
        )

        system_message = messages[0]["content"]

        # Verify source attribution is preserved in context
        assert "Process Risk Management document (page 15)" in system_message
        assert "Service Configuration Management (page 8)" in system_message
        assert "identifying, assessing, and controlling risks" in system_message

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_empty_context_handling(self, mock_load_prompt):
        """Test query handling when no context is available"""
        # Mock system prompt
        mock_load_prompt.return_value = "You are a helpful assistant.\nContext: {context}\nAnswer questions."

        messages = build_messages(
            query="What is machine learning?",
            context=""
        )

        system_message = messages[0]["content"]

        # Verify fallback message for empty context
        assert "(Chưa có tài liệu nào được tải lên)" in system_message

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_context_formatting_preservation(self, mock_load_prompt):
        """Test that context formatting and structure is preserved"""
        # Mock system prompt
        mock_load_prompt.return_value = "You are a helpful assistant.\nContext: {context}\nAnswer questions."

        structured_context = """
        Key Points:
        1. First important fact
        2. Second important fact

        Technical Details:
        - Point A
        - Point B
        """

        messages = build_messages(
            query="Explain the technical details",
            context=structured_context
        )

        system_message = messages[0]["content"]

        # Verify formatting is preserved
        assert "Key Points:" in system_message
        assert "1. First important fact" in system_message
        assert "Technical Details:" in system_message
        assert "- Point A" in system_message


class TestErrorHandling:
    """Test error handling in LLM queries"""

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    @patch('RAG_system.LLM.LLM_API.genai')
    def test_gemini_api_error_handling(self, mock_genai, mock_resolve_settings):
        """Test Gemini API error handling"""
        mock_resolve_settings.return_value = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash"
        }

        # Mock API failure
        mock_genai.GenerativeModel.side_effect = Exception("API quota exceeded")

        messages = [{"role": "user", "content": "Test query"}]

        result = call_gemini(messages)

        assert result.startswith("[Gemini Error]")
        assert "API quota exceeded" in result

    @patch('RAG_system.LLM.LLM_LOCAL.get_client')
    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    def test_lmstudio_connection_error_handling(self, mock_resolve_settings, mock_get_client):
        """Test LM Studio connection error handling"""
        mock_resolve_settings.return_value = {
            "model": "test-model",
            "temperature": 0.7
        }

        # Mock connection failure
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")

        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Test query"}]

        result = call_lmstudio(messages)

        assert result.startswith("[LM Studio Error]")
        assert "Connection refused" in result

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    def test_gemini_missing_api_key_error(self, mock_resolve_settings):
        """Test Gemini error when API key is missing"""
        mock_resolve_settings.return_value = {
            "api_key": None,
            "model": "gemini-2.0-flash"
        }

        messages = [{"role": "user", "content": "Test query"}]

        result = call_gemini(messages)

        assert result == "[Gemini Error] Missing API key (configure secrets or env)."


class TestResponseDataStructure:
    """Test response data structure and metadata"""

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    @patch('RAG_system.LLM.LLM_API.genai')
    def test_gemini_response_cleaning(self, mock_genai, mock_resolve_settings):
        """Test that Gemini responses are properly cleaned"""
        mock_resolve_settings.return_value = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash"
        }

        mock_model = MagicMock()
        mock_response = MagicMock()
        # Simulate Gemini adding its own prefixes
        mock_response.text = "Assistant: The answer is 42. Assistant: Additional info."
        mock_model.generate_content.return_value = mock_response

        mock_genai.GenerativeModel.return_value = mock_model

        messages = [{"role": "user", "content": "What is the answer?"}]

        result = call_gemini(messages)

        # Should clean the first "Assistant:" prefix but keep the rest
        assert result == "The answer is 42. Assistant: Additional info."

    @patch('RAG_system.LLM.LLM_LOCAL.get_client')
    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    def test_lmstudio_response_structure(self, mock_resolve_settings, mock_get_client):
        """Test LM Studio response structure"""
        mock_resolve_settings.return_value = {
            "model": "test-model"
        }

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Structured response with multiple lines.\n\nAdditional information."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Give me a structured answer"}]

        result = call_lmstudio(messages)

        # Verify response structure is preserved
        assert "Structured response" in result
        assert "Additional information" in result
        assert "\n\n" in result  # Formatting preserved


class TestMultiTurnConversation:
    """Test multi-turn conversation handling"""

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_conversation_history_integration(self, mock_load_prompt):
        """Test that conversation history is properly integrated"""
        # Mock system prompt
        mock_load_prompt.return_value = "You are a helpful assistant.\nContext: {context}\nAnswer questions."

        context = "Document context about Python programming."

        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more about its features."}
        ]

        messages = build_messages(
            query="How does it compare to Java?",
            context=context,
            history=history
        )

        # Should have: system + history (3 messages) + current query = 5 total
        assert len(messages) == 5

        # Verify order
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is Python?"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Python is a programming language."
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Tell me more about its features."
        assert messages[4]["role"] == "user"
        assert messages[4]["content"] == "How does it compare to Java?"

        # Verify context is in system message
        assert "Python programming" in messages[0]["content"]

    @patch('RAG_system.LLM.chat_handler.load_system_prompt')
    def test_history_normalization_in_conversation(self, mock_load_prompt):
        """Test that history with 'bot' roles is properly normalized"""
        # Mock system prompt
        mock_load_prompt.return_value = "You are a helpful assistant.\nContext: {context}\nAnswer questions."

        context = "Context information."

        # History with 'bot' role (from UI)
        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "bot", "content": "Answer 1"},  # Should become 'assistant'
            {"role": "user", "content": "Question 2"}
        ]

        messages = build_messages(
            query="Final question",
            context=context,
            history=history
        )

        # Verify 'bot' role was converted to 'assistant'
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["content"] == "Answer 1"
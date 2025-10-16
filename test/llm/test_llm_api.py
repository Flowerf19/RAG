"""
Tests for LLM_API.py
====================
Test Gemini API integration, message conversion, and response cleaning.
"""

import pytest
from unittest.mock import patch, MagicMock

# Add project root to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RAG_system.LLM.LLM_API import (
    convert_to_gemini_format,
    clean_response,
    call_gemini,
)


class TestMessageConversion:
    """Test message format conversion for Gemini"""

    def test_convert_to_gemini_format_basic(self):
        """Test basic message conversion"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        result = convert_to_gemini_format(messages)

        expected = (
            "System: You are a helpful assistant.\n"
            "User: Hello!\n"
            "Assistant: Hi there!\n"
            "User: How are you?"
        )

        assert result == expected

    def test_convert_to_gemini_format_skip_empty_content(self):
        """Test that messages with empty content are skipped"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": ""},  # Empty content
            {"role": "user", "content": "Valid question"}
        ]

        result = convert_to_gemini_format(messages)

        expected = (
            "System: System prompt\n"
            "User: Valid question"
        )

        assert result == expected

    def test_convert_to_gemini_format_skip_missing_role(self):
        """Test that messages with missing role are skipped"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"content": "No role"},  # Missing role
            {"role": "user", "content": "Valid question"}
        ]

        result = convert_to_gemini_format(messages)

        expected = (
            "System: System prompt\n"
            "User: Valid question"
        )

        assert result == expected

    def test_convert_to_gemini_format_unknown_role(self):
        """Test handling of unknown roles"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "unknown", "content": "Unknown role"},  # Unknown role
            {"role": "user", "content": "Valid question"}
        ]

        result = convert_to_gemini_format(messages)

        # Unknown roles should be skipped (no prefix added)
        expected = (
            "System: System prompt\n"
            "User: Valid question"
        )

        assert result == expected


class TestResponseCleaning:
    """Test response cleaning functionality"""

    def test_clean_response_no_prefix(self):
        """Test cleaning response without Assistant prefix"""
        response = "This is a normal response."
        result = clean_response(response)
        assert result == "This is a normal response."

    def test_clean_response_with_assistant_prefix(self):
        """Test cleaning response with 'Assistant:' prefix"""
        response = "Assistant: This is the actual response."
        result = clean_response(response)
        assert result == "This is the actual response."

    def test_clean_response_with_assistant_colon_space(self):
        """Test cleaning response with 'Assistant :' prefix"""
        response = "Assistant : This is the actual response."
        result = clean_response(response)
        assert result == "This is the actual response."

    def test_clean_response_case_insensitive(self):
        """Test cleaning response with case variations"""
        test_cases = [
            ("assistant: lowercase prefix", "lowercase prefix"),
            ("ASSISTANT: uppercase prefix", "uppercase prefix"),
            ("Assistant: mixed case prefix", "mixed case prefix")
        ]

        for input_response, expected in test_cases:
            result = clean_response(input_response)
            assert result == expected

    def test_clean_response_whitespace_handling(self):
        """Test whitespace handling in cleaning"""
        response = "Assistant:   This has extra spaces   "
        result = clean_response(response)
        assert result == "This has extra spaces"

    def test_clean_response_no_content_after_prefix(self):
        """Test cleaning when there's no content after prefix"""
        response = "Assistant:"
        result = clean_response(response)
        assert result == ""

    def test_clean_response_multiple_prefixes(self):
        """Test that only the first matching prefix is removed"""
        response = "Assistant: Assistant: Double prefix"
        result = clean_response(response)
        assert result == "Assistant: Double prefix"


class TestGeminiAPI:
    """Test Gemini API calls"""

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    @patch('RAG_system.LLM.LLM_API.genai')
    def test_call_gemini_success(self, mock_genai, mock_resolve_settings):
        """Test successful Gemini API call"""
        # Mock settings
        mock_resolve_settings.return_value = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash",
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # Mock Gemini response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a test response."
        mock_model.generate_content.return_value = mock_response

        mock_genai.GenerativeModel.return_value = mock_model

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]

        result = call_gemini(messages)

        assert result == "This is a test response."
        mock_genai.configure.assert_called_once_with(api_key="test_key")
        mock_genai.GenerativeModel.assert_called_once_with("gemini-2.0-flash")
        mock_model.generate_content.assert_called_once()

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    @patch('RAG_system.LLM.LLM_API.genai')
    def test_call_gemini_with_generation_config(self, mock_genai, mock_resolve_settings):
        """Test Gemini API call with generation config parameters"""
        mock_resolve_settings.return_value = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash",
            "temperature": 0.8,
            "max_tokens": 500
        }

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Configured response."
        mock_model.generate_content.return_value = mock_response

        mock_genai.GenerativeModel.return_value = mock_model

        messages = [{"role": "user", "content": "Test"}]

        result = call_gemini(messages, temperature=0.8, max_tokens=500)

        assert result == "Configured response."

        # Check that generate_content was called with generation_config
        call_args = mock_model.generate_content.call_args
        assert call_args[0][0] == "User: Test"  # prompt_text
        assert "generation_config" in call_args[1]  # keyword argument
        generation_config = call_args[1]["generation_config"]
        assert generation_config["temperature"] == 0.8
        assert generation_config["max_output_tokens"] == 500

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    @patch('RAG_system.LLM.LLM_API.genai')
    def test_call_gemini_with_response_cleaning(self, mock_genai, mock_resolve_settings):
        """Test that Gemini response is cleaned"""
        mock_resolve_settings.return_value = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash"
        }

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Assistant: This response needs cleaning."
        mock_model.generate_content.return_value = mock_response

        mock_genai.GenerativeModel.return_value = mock_model

        messages = [{"role": "user", "content": "Test"}]

        result = call_gemini(messages)

        assert result == "This response needs cleaning."

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    def test_call_gemini_missing_api_key(self, mock_resolve_settings):
        """Test Gemini API call with missing API key"""
        mock_resolve_settings.return_value = {
            "api_key": None,
            "model": "gemini-2.0-flash"
        }

        messages = [{"role": "user", "content": "Test"}]

        result = call_gemini(messages)

        assert result == "[Gemini Error] Missing API key (configure secrets or env)."

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    @patch('RAG_system.LLM.LLM_API.genai')
    def test_call_gemini_api_exception(self, mock_genai, mock_resolve_settings):
        """Test Gemini API call with API exception"""
        mock_resolve_settings.return_value = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash"
        }

        mock_genai.GenerativeModel.side_effect = Exception("API Error")

        messages = [{"role": "user", "content": "Test"}]

        result = call_gemini(messages)

        assert result == "[Gemini Error] API Error"

    @patch('RAG_system.LLM.LLM_API.resolve_gemini_settings')
    @patch('RAG_system.LLM.LLM_API.genai')
    def test_call_gemini_with_parameter_overrides(self, mock_genai, mock_resolve_settings):
        """Test Gemini API call with parameter overrides"""
        mock_resolve_settings.return_value = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash",
            "temperature": 0.7,
            "max_tokens": 1000
        }

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Override response."
        mock_model.generate_content.return_value = mock_response

        mock_genai.GenerativeModel.return_value = mock_model

        messages = [{"role": "user", "content": "Test"}]

        result = call_gemini(
            messages,
            model_name="gemini-pro",
            temperature=0.9,
            max_tokens=2000
        )

        assert result == "Override response."

        # Verify that overrides were passed to resolve_settings
        mock_resolve_settings.assert_called_once_with(
            override_model="gemini-pro",
            override_temperature=0.9,
            override_max_tokens=2000
        )
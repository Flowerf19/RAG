"""
Tests for LLM_LOCAL.py
======================
Test LM Studio API integration and client creation.
"""

import pytest
from unittest.mock import patch, MagicMock

# Add project root to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RAG_system.LLM.LLM_LOCAL import (
    get_client,
    call_lmstudio,
)


class TestClientCreation:
    """Test OpenAI client creation for LM Studio"""

    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    def test_get_client_creation(self, mock_resolve_settings):
        """Test that get_client creates OpenAI client with correct parameters"""
        mock_resolve_settings.return_value = {
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key": "test_key"
        }

        with patch('RAG_system.LLM.LLM_LOCAL.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            result = get_client()

            mock_openai.assert_called_once_with(
                base_url="http://127.0.0.1:1234/v1",
                api_key="test_key"
            )
            assert result == mock_client


class TestLMStudioAPI:
    """Test LM Studio API calls"""

    @patch('RAG_system.LLM.LLM_LOCAL.get_client')
    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    def test_call_lmstudio_success(self, mock_resolve_settings, mock_get_client):
        """Test successful LM Studio API call"""
        # Mock settings
        mock_resolve_settings.return_value = {
            "model": "google/gemma-3-4b",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512
        }

        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a test response from LM Studio."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_get_client.return_value = mock_client

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]

        result = call_lmstudio(messages)

        assert result == "This is a test response from LM Studio."

        # Verify API call parameters
        mock_client.chat.completions.create.assert_called_once_with(
            model="google/gemma-3-4b",
            messages=messages,  # Messages passed through unchanged
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )

    @patch('RAG_system.LLM.LLM_LOCAL.get_client')
    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    def test_call_lmstudio_with_parameter_overrides(self, mock_resolve_settings, mock_get_client):
        """Test LM Studio API call with parameter overrides"""
        # Mock settings (should return the overridden values)
        mock_resolve_settings.return_value = {
            "model": "custom-model",  # Override applied
            "temperature": 0.9,      # Override applied
            "top_p": 0.8,            # Override applied
            "max_tokens": 2048       # Override applied
        }

        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Override response."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Test"}]

        result = call_lmstudio(
            messages,
            model="custom-model",
            temperature=0.9,
            top_p=0.8,
            max_tokens=2048
        )

        assert result == "Override response."

        # Verify that overrides were passed to resolve_settings
        mock_resolve_settings.assert_called_once_with(
            override_model="custom-model",
            override_temperature=0.9,
            override_top_p=0.8,
            override_max_tokens=2048
        )

        # Verify API call used resolved settings
        mock_client.chat.completions.create.assert_called_once_with(
            model="custom-model",
            messages=messages,
            temperature=0.9,  # Override applied
            top_p=0.8,       # Override applied
            max_tokens=2048  # Override applied
        )

    @patch('RAG_system.LLM.LLM_LOCAL.get_client')
    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    def test_call_lmstudio_empty_response(self, mock_resolve_settings, mock_get_client):
        """Test LM Studio API call with empty response content"""
        mock_resolve_settings.return_value = {
            "model": "test-model",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512
        }

        # Mock OpenAI client with empty content
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None  # Empty content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Test"}]

        result = call_lmstudio(messages)

        assert result == ""

    @patch('RAG_system.LLM.LLM_LOCAL.get_client')
    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    def test_call_lmstudio_api_exception(self, mock_resolve_settings, mock_get_client):
        """Test LM Studio API call with API exception"""
        mock_resolve_settings.return_value = {
            "model": "test-model",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512
        }

        # Mock OpenAI client to raise exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")

        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Test"}]

        result = call_lmstudio(messages)

        assert result == "[LM Studio Error] Connection failed"

    @patch('RAG_system.LLM.LLM_LOCAL.get_client')
    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    def test_call_lmstudio_default_parameters(self, mock_resolve_settings, mock_get_client):
        """Test LM Studio API call with default parameters"""
        mock_resolve_settings.return_value = {
            "model": "google/gemma-3-4b",
            "temperature": None,  # No temperature set
            "top_p": None,        # No top_p set
            "max_tokens": None    # No max_tokens set
        }

        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Default params response."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Test"}]

        result = call_lmstudio(messages)

        assert result == "Default params response."

        # Verify API call uses function defaults when resolved settings are None
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.7  # Function default
        assert call_args[1]["top_p"] == 0.95        # Function default
        assert call_args[1]["max_tokens"] is None   # No default in function

    @patch('RAG_system.LLM.LLM_LOCAL.get_client')
    @patch('RAG_system.LLM.LLM_LOCAL.resolve_lmstudio_settings')
    def test_call_lmstudio_direct_message_passing(self, mock_resolve_settings, mock_get_client):
        """Test that LM Studio receives messages in OpenAI format directly"""
        mock_resolve_settings.return_value = {
            "model": "test-model",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512
        }

        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Direct message response."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        mock_get_client.return_value = mock_client

        # Test with complex message structure
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up question"}
        ]

        result = call_lmstudio(messages)

        assert result == "Direct message response."

        # Verify messages were passed through unchanged
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["messages"] == messages
"""
LLM Module Tests
================
Comprehensive tests for the LLM (Large Language Model) integration module.

This module tests:
- Configuration loading and resolution (Gemini and LM Studio)
- API integration (Gemini and LM Studio)
- Chat message handling and formatting
- End-to-end query workflows
- Error handling and edge cases
- Data source attribution and retrieval integration

Test Structure:
- test_config_loader.py: Configuration management tests
- test_llm_api.py: Gemini API integration tests
- test_llm_local.py: LM Studio API integration tests
- test_chat_handler.py: Message handling and chat logic tests
- test_integration.py: End-to-end workflow tests
- conftest.py: Test fixtures and mock data
"""

# Import test modules for easier running
from . import test_config_loader
from . import test_llm_api
from . import test_llm_local
from . import test_chat_handler
from . import test_integration

__all__ = [
    'test_config_loader',
    'test_llm_api',
    'test_llm_local',
    'test_chat_handler',
    'test_integration'
]
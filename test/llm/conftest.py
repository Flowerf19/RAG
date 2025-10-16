"""
Test fixtures and configuration for LLM tests
==============================================
Provides mock data and fixtures for testing LLM functionality.
"""

import pytest
from pathlib import Path

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Mock API responses
MOCK_GEMINI_RESPONSE = {
    "text": "This is a mock Gemini response with useful information."
}

MOCK_LM_STUDIO_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "This is a mock LM Studio response with detailed analysis."
            }
        }
    ]
}

# Mock configuration data
MOCK_APP_CONFIG = {
    "ui": {
        "default_backend": "gemini"
    },
    "paths": {
        "data_dir": "data",
        "prompt_path": "prompt/system_prompt.txt"
    },
    "llm": {
        "gemini": {
            "model": "gemini-2.0-flash",
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key": {
                "secrets_key": "gemini_api_key",
                "env": "GEMINI_API_KEY"
            }
        },
        "lmstudio": {
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key": {
                "env": "LMSTUDIO_API_KEY",
                "default": "lm-studio"
            },
            "model": "google/gemma-3-4b",
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512
            }
        }
    }
}

# Mock system prompt content
MOCK_SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh, hữu ích.

Context từ tài liệu:
{context}

Hãy trả lời một cách chính xác, ngắn gọn và có căn cứ vào thông tin được cung cấp."""

# Mock retrieval context data
MOCK_RETRIEVAL_CONTEXT = """
Từ tài liệu "Process Risk Management" (trang 15):
Quản lý rủi ro bao gồm việc xác định, đánh giá và kiểm soát các rủi ro trong dự án.

Từ tài liệu "Service Configuration Management" (trang 8):
Quản lý cấu hình đảm bảo việc triển khai nhất quán và có thể theo dõi.
"""

# Mock conversation history
MOCK_CONVERSATION_HISTORY = [
    {"role": "user", "content": "Quản lý rủi ro là gì?"},
    {"role": "assistant", "content": "Quản lý rủi ro là quá trình xác định, đánh giá và kiểm soát rủi ro."},
    {"role": "user", "content": "Cho ví dụ cụ thể"}
]

# Mock OpenAI format messages
MOCK_OPENAI_MESSAGES = [
    {
        "role": "system",
        "content": "Bạn là trợ lý AI. Context: Quản lý rủi ro rất quan trọng trong dự án."
    },
    {
        "role": "user",
        "content": "Giải thích quản lý rủi ro"
    }
]

# Mock Gemini format conversion
MOCK_GEMINI_FORMAT = """System: Bạn là trợ lý AI. Context: Quản lý rủi ro rất quan trọng trong dự án.
User: Giải thích quản lý rủi ro"""


@pytest.fixture
def mock_config():
    """Fixture providing mock app configuration"""
    return MOCK_APP_CONFIG.copy()


@pytest.fixture
def mock_system_prompt():
    """Fixture providing mock system prompt content"""
    return MOCK_SYSTEM_PROMPT


@pytest.fixture
def mock_retrieval_context():
    """Fixture providing mock retrieval context"""
    return MOCK_RETRIEVAL_CONTEXT


@pytest.fixture
def mock_conversation_history():
    """Fixture providing mock conversation history"""
    return MOCK_CONVERSATION_HISTORY.copy()


@pytest.fixture
def mock_openai_messages():
    """Fixture providing mock OpenAI format messages"""
    return MOCK_OPENAI_MESSAGES.copy()


@pytest.fixture
def mock_gemini_format():
    """Fixture providing expected Gemini format output"""
    return MOCK_GEMINI_FORMAT


@pytest.fixture
def sample_pdf_content():
    """Fixture providing sample PDF content for testing"""
    return """
    CHƯƠNG 1: QUẢN LÝ RỦI RO

    1.1 Định nghĩa
    Quản lý rủi ro là quá trình có hệ thống nhằm xác định, phân tích, đánh giá
    và xử lý các rủi ro có thể ảnh hưởng đến mục tiêu của tổ chức.

    1.2 Các bước thực hiện
    - Xác định rủi ro
    - Đánh giá mức độ ảnh hưởng
    - Lập kế hoạch ứng phó
    - Giám sát và kiểm soát

    1.3 Ví dụ thực tế
    Trong dự án phần mềm, rủi ro có thể bao gồm:
    • Trễ deadline
    • Vượt ngân sách
    • Thiếu nhân lực
    • Thay đổi yêu cầu
    """


@pytest.fixture
def sample_query():
    """Fixture providing sample user query"""
    return "Quản lý rủi ro gồm những bước nào?"


@pytest.fixture
def sample_response():
    """Fixture providing sample LLM response"""
    return """Dựa trên tài liệu được cung cấp, quản lý rủi ro gồm các bước chính:

1. Xác định rủi ro - Tìm ra các rủi ro tiềm ẩn
2. Đánh giá mức độ ảnh hưởng - Xác định tác động và khả năng xảy ra
3. Lập kế hoạch ứng phó - Chuẩn bị biện pháp xử lý
4. Giám sát và kiểm soát - Theo dõi và điều chỉnh liên tục

Các ví dụ rủi ro trong dự án phần mềm bao gồm trễ deadline, vượt ngân sách, thiếu nhân lực và thay đổi yêu cầu."""


@pytest.fixture
def mock_gemini_response():
    """Fixture providing mock Gemini API response"""
    return MOCK_GEMINI_RESPONSE.copy()


@pytest.fixture
def mock_lm_studio_response():
    """Fixture providing mock LM Studio API response"""
    return MOCK_LM_STUDIO_RESPONSE.copy()


# Test data for configuration testing
TEST_ENV_VARS = {
    "GEMINI_API_KEY": "test_gemini_key_123",
    "LMSTUDIO_API_KEY": "test_lmstudio_key_456",
    "LMSTUDIO_BASE_URL": "http://localhost:5678/v1",
    "LMSTUDIO_MODEL": "custom-model-v1"
}

TEST_SECRETS = {
    "gemini_api_key": "secrets_gemini_key_789"
}
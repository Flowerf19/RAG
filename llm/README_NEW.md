# LLM Module v2.0 - OOP Refactored

## 🔄 Migration từ version cũ

Module LLM đã được refactor hoàn toàn theo nguyên tắc OOP và Single Responsibility:

### ❌ Deprecated Files (không dùng nữa)
- `LLM_API.py` → Thay bằng `gemini_client.py`
- `LLM_LOCAL.py` → Thay bằng `lmstudio_client.py`  
- `LLM_FE.py` → Thay bằng `ui/app.py`

### ✅ New Structure
```
llm/
├── base_client.py          # Abstract base class
├── gemini_client.py        # Gemini implementation
├── lmstudio_client.py      # LMStudio implementation
├── client_factory.py       # Factory pattern
├── chat_handler.py         # Message formatting (unchanged)
└── config_loader.py        # Config management (unchanged)
```

## 🎯 Quick Start

### Old Way (Deprecated)
```python
# ❌ Old
from llm.LLM_API import call_gemini
response = call_gemini(messages)
```

### New Way (OOP)
```python
# ✅ New
from llm.client_factory import LLMClientFactory

client = LLMClientFactory.create_gemini()
response = client.generate(messages)
```

## 📖 Detailed Documentation

Xem `README_v2.md` để đọc full documentation về:
- Architecture patterns
- Usage examples
- Integration guides
- Testing guidelines

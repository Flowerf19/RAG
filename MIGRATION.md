# Migration Guide - Cấu trúc mới

## 📋 Tổng quan thay đổi

Hệ thống đã được refactor hoàn toàn theo nguyên tắc OOP và Single Responsibility:

```
CŨ (llm/):                          MỚI (ui/ + llm/):
├── LLM_FE.py (UI+LLM)    →        ui/
├── LLM_API.py            →          ├── app.py (Main Streamlit)
├── LLM_LOCAL.py          →          └── components/ (UI components)
├── chat_handler.py       →        
├── chat_styles.css       →        llm/
└── config_loader.py      →          ├── base_client.py (Abstract)
                                     ├── gemini_client.py
                                     ├── lmstudio_client.py
                                     ├── client_factory.py
                                     ├── chat_handler.py (unchanged)
                                     └── config_loader.py (unchanged)
```

## 🚀 Cách chạy UI mới

### Cũ
```powershell
streamlit run llm/LLM_FE.py
```

### Mới
```powershell
streamlit run ui/app.py
```

## 🔧 Code Migration

### 1. LLM Client Usage

#### Cũ (Deprecated)
```python
from llm.LLM_API import call_gemini
from llm.LLM_LOCAL import call_lmstudio

# Call directly
response = call_gemini(messages)
response = call_lmstudio(messages)
```

#### Mới (OOP)
```python
from llm.client_factory import LLMClientFactory

# Create client via factory
client = LLMClientFactory.create_from_string("gemini")  # or "lmstudio"

# Generate response
response = client.generate(messages)
```

### 2. UI Components

#### Cũ (Monolithic)
```python
# All UI logic trong LLM_FE.py (500+ lines)
# Sidebar, chat, sources đều trong 1 file
```

#### Mới (Modular)
```python
from ui.components import ChatDisplay, SourceDisplay, Sidebar

# Each component is a class
chat_display = ChatDisplay()
chat_display.render(messages)

sidebar = Sidebar(data_dir=Path("data/pdf"))
settings = sidebar.render()

source_display = SourceDisplay()
source_display.render(sources, retrieval_info)
```

### 3. Main App Structure

#### Cũ (Procedural)
```python
# LLM_FE.py: Procedural code, top-to-bottom execution
st.sidebar...
st.markdown...
if prompt:
    # inline logic
```

#### Mới (OOP)
```python
# ui/app.py: OOP with clear responsibilities
class RAGChatApp:
    def __init__(self):
        self.chat_display = ChatDisplay()
        self.sidebar = Sidebar(...)
    
    def run(self):
        settings = self.sidebar.render()
        self.chat_display.render(messages)
        # ...
```

## 📝 Benefits của cấu trúc mới

### 1. Separation of Concerns
- **UI (ui/)**: Chỉ lo rendering
- **LLM (llm/)**: Chỉ lo gọi LLM
- **Backend (pipeline/)**: Chỉ lo retrieval

### 2. OOP Design Patterns
- **Factory Pattern**: `LLMClientFactory` để tạo clients
- **Strategy Pattern**: `BaseLLMClient` → swap providers dễ dàng
- **Component Pattern**: UI components reusable

### 3. Testability
```python
# Easy to test individual components
def test_gemini_client():
    client = GeminiClient(config={...})
    response = client.generate(test_messages)
    assert response == expected

def test_chat_display():
    display = ChatDisplay()
    # Test rendering logic
```

### 4. Maintainability
- Mỗi file < 300 lines
- Clear responsibilities
- Easy to locate bugs
- Easy to extend (thêm LLM provider mới)

## 🔄 Backward Compatibility

### Old Files (Giữ lại tạm thời)
Các file cũ vẫn được giữ trong `llm/` để backward compatibility:
- `LLM_API.py` (deprecated)
- `LLM_LOCAL.py` (deprecated)
- `LLM_FE.py` (deprecated)

**Lưu ý**: Những file này sẽ bị xóa trong version tiếp theo.

### Migration Path
1. ✅ **Giai đoạn 1** (Hiện tại): Cấu trúc mới được tạo song song
2. 🔄 **Giai đoạn 2** (Tiếp theo): Update tất cả code sử dụng file cũ
3. ❌ **Giai đoạn 3** (Cuối cùng): Xóa file cũ

## 🧪 Testing New Structure

### Test UI
```powershell
streamlit run ui/app.py
```

### Test LLM Clients
```python
# Test trong Python REPL
from llm.client_factory import LLMClientFactory

# Test Gemini
gemini = LLMClientFactory.create_gemini()
print(gemini.is_available())
response = gemini.generate([{"role": "user", "content": "Hello"}])
print(response)

# Test LMStudio
lmstudio = LLMClientFactory.create_lmstudio()
print(lmstudio.is_available())
response = lmstudio.generate([{"role": "user", "content": "Hello"}])
print(response)
```

## 📚 Documentation

- **UI Module**: `ui/README.md`
- **LLM Module**: `llm/README_NEW.md`
- **Components**: Inline docstrings trong mỗi class

## ❓ FAQ

### Q: File cũ có bị xóa không?
A: Chưa, giữ lại để backward compatibility. Sẽ xóa trong version sau.

### Q: Có cần update code hiện có không?
A: Nên update để sử dụng cấu trúc mới (OOP, modular). Old code vẫn chạy nhưng deprecated.

### Q: Cách thêm LLM provider mới?
A:
```python
# 1. Tạo class mới implement BaseLLMClient
class NewProviderClient(BaseLLMClient):
    def generate(self, messages, ...): ...
    def is_available(self): ...

# 2. Thêm vào factory
class LLMClientFactory:
    @staticmethod
    def create_newprovider(...):
        return NewProviderClient(...)
```

### Q: UI có thay đổi gì không?
A: Giao diện giống hệt, chỉ code architecture thay đổi.

## 🎯 Next Steps

1. ✅ Chạy thử UI mới: `streamlit run ui/app.py`
2. ✅ Test các LLM clients
3. 🔄 Update code của bạn để dùng factory pattern
4. 📝 Đọc README trong `ui/` và `llm/` để hiểu rõ hơn

---

**Cập nhật**: November 5, 2025
**Version**: 2.0.0
**Status**: ✅ Refactoring hoàn tất, cấu trúc mới đã sẵn sàng

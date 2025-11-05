# UI Module - Streamlit Frontend

## 🎯 Mục đích
Module UI chứa toàn bộ Streamlit frontend, tách biệt hoàn toàn khỏi logic LLM và backend.

## 🏗️ Cấu trúc
```
ui/
├── app.py                   # Main Streamlit app (entry point)
├── components/              # Reusable UI components
│   ├── chat_display.py     # Chat message rendering
│   ├── sidebar.py          # Sidebar controls and settings
│   └── source_display.py   # Source information display
├── styles/
│   └── chat_styles.css     # CSS styling
└── README.md
```

## 🚀 Chạy UI
```powershell
# Từ thư mục gốc của project
streamlit run ui/app.py

# Hoặc (nếu ở trong thư mục ui/)
streamlit run app.py
```

## 🔧 Architecture Pattern

### OOP Components
Mỗi UI component là một class với responsibility rõ ràng:

1. **ChatDisplay** - Render chat messages
   - `render(messages, is_generating, pending_prompt)`: Render chat log
   - `render_header(title)`: Render header

2. **SourceDisplay** - Display retrieval sources
   - `render(sources, retrieval_info, expanded_queries)`: Render all source info
   - `_render_retrieval_stats()`: Render metrics
   - `_render_sources()`: Render document sources

3. **Sidebar** - Sidebar controls
   - `render(on_embedding_clicked)`: Render sidebar and return settings
   - Returns settings dict: `{backend_mode, embedder_type, reranker_type, ...}`

### Main App Class
`RAGChatApp` orchestrates all components:
- Manages session state
- Coordinates UI flow
- Calls LLM clients (via factory)
- Calls backend retrieval (via `fetch_retrieval`)

## 📝 Nguyên tắc thiết kế

### Single Responsibility
- **UI components**: Chỉ lo rendering, không có business logic
- **Main app**: Orchestration và coordination
- **LLM clients**: Gọi LLM (trong `llm/`)
- **Backend**: Retrieval logic (trong `pipeline/`)

### Dependency Injection
Components nhận dependencies qua constructor:
```python
sidebar = Sidebar(data_dir=paths_data_dir())
```

### Factory Pattern
LLM clients được tạo qua factory:
```python
client = LLMClientFactory.create_from_string(backend_mode)
response = client.generate(messages)
```

## 🔗 Integration với các module khác

### LLM Module (`llm/`)
```python
from llm.client_factory import LLMClientFactory
from llm.chat_handler import build_messages

# Tạo client
client = LLMClientFactory.create_from_string("gemini")

# Build messages
messages = build_messages(query="Hello", context="...", history=[])

# Generate
response = client.generate(messages)
```

### Backend Module (`pipeline/`)
```python
from pipeline.backend_connector import fetch_retrieval

# Fetch context from retrieval
ret = fetch_retrieval(
    query_text="...",
    top_k=5,
    embedder_type="huggingface_local",
    reranker_type="bge_m3_hf_local"
)

context = ret["context"]
sources = ret["sources"]
```

## 📦 Dependencies
- `streamlit`: UI framework
- `llm`: LLM clients (internal)
- `pipeline`: Backend retrieval (internal)

## 🎨 Styling
CSS được load từ `styles/chat_styles.css` với các class:
- `.chat-header`: Chat window header
- `.chat-log`: Chat container
- `.chat-row`, `.chat-bubble`: Message styling
- `.typing`: Typing indicator animation

## 🔄 Session State Management
App sử dụng Streamlit session_state để quản lý:
- `messages`: Chat history (OpenAI format)
- `is_generating`: Generation status
- `pending_prompt`: Current prompt being processed
- `last_sources`: Last retrieval sources
- `last_retrieval_info`: Last retrieval metadata
- `last_queries`: Expanded queries from QEM

## ⚠️ Lưu ý
- **Không gọi LLM trực tiếp**: Dùng `LLMClientFactory`
- **Không gọi embedding trực tiếp**: Dùng `fetch_retrieval`
- **Components thuần túy**: Không có business logic trong UI components

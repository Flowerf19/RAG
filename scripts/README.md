# Scripts Directory

Thư mục chứa các script để chạy RAG Pipeline

## 📁 Cấu trúc

```text
scripts/
├── run_pipeline.py      # Script Python chính để chạy pipeline
├── run_pipeline.bat     # Batch script cho Windows CMD
├── run_pipeline.ps1     # PowerShell script với options nâng cao
└── README.md           # Tài liệu hướng dẫn
```

```
scripts/
├── run_pipeline.py      # Script Python chính để chạy pipeline
├── run_pipeline.bat     # Batch script cho Windows CMD
├── run_pipeline.ps1     # PowerShell script với options nâng cao
└── README.md           # Tài liệu hướng dẫn
```

## 🚀 Cách sử dụng

### 1. Chạy từ Python trực tiếp

```bash
# Từ thư mục gốc của project
python scripts/run_pipeline.py
```

### 2. Chạy bằng Batch script (Windows CMD)

```cmd
# Từ thư mục gốc của project
scripts\run_pipeline.bat
```

### 3. Chạy bằng PowerShell script

```powershell
# Từ thư mục gốc của project
.\scripts\run_pipeline.ps1

# Với tùy chọn model khác
.\scripts\run_pipeline.ps1 -Model BGE_M3

# Xem trợ giúp
.\scripts\run_pipeline.ps1 -Help
```

## 🔧 Chức năng

Script sẽ:

1. ✅ Khởi tạo RAG Pipeline với embedder Gemma
2. 📁 Tự động tìm và xử lý tất cả PDF trong `data/pdf/`
3. ✂️ Chia nhỏ nội dung thành chunks
4. 🧠 Tạo embeddings cho tất cả chunks
5. 💾 Lưu vector index (FAISS) và metadata
6. 📊 Hiển thị báo cáo chi tiết về quá trình xử lý

## 📊 Output

Dữ liệu được lưu vào các thư mục:

- `data/chunks/` - File text chứa chunks
- `data/embeddings/` - File JSON chứa embeddings
- `data/vectors/` - FAISS index và metadata
- `data/metadata/` - Thông tin metadata bổ sung

## ⚠️ Yêu cầu

- Virtual environment đã được tạo tại `.venv/`
- Ollama server đang chạy với model `embeddinggemma:latest`
- Các thư viện Python đã được cài đặt

## 🔍 Troubleshooting

### Lỗi "Không tìm thấy virtual environment"

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Lỗi "Ollama server not found"

```bash
# Kiểm tra Ollama đang chạy
ollama list

# Khởi động Ollama (nếu cần)
ollama serve

# Pull model cần thiết
ollama pull embeddinggemma:latest
```

### Lỗi "No PDF files found"

- Đảm bảo có file PDF trong thư mục `data/pdf/`
- Kiểm tra định dạng file (.pdf)

## 📝 Logs

Script sẽ hiển thị log chi tiết bao gồm:

- Số lượng PDF được xử lý
- Số trang, chunks, embeddings được tạo
- Thời gian xử lý
- Bất kỳ lỗi nào gặp phải

## 🎯 Pipeline Flow

```text
PDF Files → PDFLoader → HybridChunker → OllamaEmbedder → VectorStore → FAISS Index
```

Script này tự động hóa toàn bộ quy trình trên cho tất cả PDF trong thư mục.
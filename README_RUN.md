# Hướng dẫn chạy hệ thống RAG Pipeline

## 1. Chuẩn bị môi trường

```powershell
# Bật venv (nếu chưa bật)
. .\.venv\Scripts\Activate.ps1

# Cài đặt các package cần thiết
pip install -r requirements.txt

# Cài đặt các model spaCy (nếu chưa có)
python install_spacy_models.py
```

## Các model spaCy cần thiết

- `en_core_web_sm` - Model nhỏ gọn cho xử lý văn bản cơ bản
- `en_core_web_md` - Model trung bình với word vectors
- `en_core_web_lg` - Model lớn với word vectors chất lượng cao

## 2. Chạy pipeline xử lý PDF → FAISS

```powershell
# Xử lý toàn bộ PDF trong thư mục data/pdf/
python run_pipeline.py
```

- Kết quả sẽ sinh ra các file:
  - data/chunks/…_chunks_*.txt
  - data/embeddings/…_embeddings_*.json
  - data/vectors/…_vectors_*.faiss
  - data/vectors/…_metadata_map_*.pkl
  - data/metadata/…_summary_*.json

## 3. Chạy giao diện FE (Streamlit)

```powershell
# Chạy giao diện FE (ví dụ với file llm/LLM_FE.py)
streamlit run llm/LLM_FE.py
```

- Truy cập trình duyệt tại địa chỉ được in ra (thường là `http://localhost:8501`)

## 4. Debug nhanh

```powershell
# Kiểm tra chunk/page/debug
python debug_chunks.py
python debug_blocks.py
```

---

## Lưu ý

- Đảm bảo Ollama server đã chạy và đã pull đủ model embedding (gemma, bge-m3)
- Nếu muốn chạy lại pipeline từ đầu, xóa file cache: `data/cache/processed_chunks.json`
- Nếu gặp lỗi page_number N/A, kiểm tra lại provenance và metadata của block khi chunking

---

**Mọi thắc mắc liên hệ nhóm phát triển!**

# Luồng Pipeline RAG Hoàn Chỉnh

## 1. LOAD & PREPROCESSING (Module: loaders)
```
PDF File
   ↓
[PDFLoader] → Đọc PDF
   ↓
[Normalize] → Chuẩn hóa blocks, tables
   ↓
PDFDocument (chứa: pages, blocks, tables, metadata)
```

## 2. CHUNKING (Module: chunkers)
```
PDFDocument
   ↓
[Chunker] → Chia nhỏ theo chiến lược (semantic, fixed_size, etc.)
   ↓
ChunkDocument (danh sách chunks với metadata)
```

## 3. INDEXING & RETRIEVAL (Module: retriever)

### 3.1 Xây dựng chỉ mục
```
Chunks
   ↓
[GemmaEmbedder] → Tính embedding vector cho mỗi chunk
   ↓
Embedding Index (vector store)

Chunks
   ↓
[BM25Retriever] → Xây dựng BM25 index
   ↓
BM25 Index (term frequency)
```

### 3.2 Truy vấn (Query)
```
User Query
   ↓
   ├─→ [GemmaEmbedder] → Query embedding
   │      ↓
   │   [Embedding Search] → Cosine similarity với tất cả chunks
   │      ↓
   │   Embedding scores (semantic)
   │
   └─→ [BM25Retriever] → BM25 scoring
          ↓
       BM25 scores (keyword matching)
```

## 4. HYBRID SCORING (Module: retriever/hybrid.py)
```
Embedding scores + BM25 scores
   ↓
[HybridRetriever]
   score = α * embedding_score + (1-α) * bm25_score
   ↓
Top 50-100 chunks (ranked by hybrid score)
```

## 5. RERANKING (Module: retriever/reranker.py)
```
Top 50-100 chunks
   ↓
[SimpleReranker] → Re-score lại với embedding cosine similarity
   (hoặc [OllamaReranker] nếu có BGE/Qwen3 reranker model)
   ↓
Top 5-10 chunks (final results)
```

## 6. GENERATION (Module: generator - chưa có)
```
Top 5-10 chunks
   ↓
[LLM Generator] → Tổng hợp chunks + query để sinh câu trả lời
   ↓
Final Answer
```

---

## TÓM TẮT LUỒNG:

1. **PDF** → 2. **Chunks** → 3. **Embedding + BM25 Index** → 4. **Query** → 5. **Hybrid Scoring** → 6. **Reranking** → 7. **Top Results** → 8. **LLM Generation** → **Answer**

---

## CÁC MODULE ĐÃ CÓ:

✅ **loaders**: Load & normalize PDF
✅ **chunkers**: Chia nhỏ documents
✅ **retriever/embedding.py**: Gemma embedding
✅ **retriever/bm25.py**: BM25 scoring
✅ **retriever/hybrid.py**: Hybrid scoring
✅ **retriever/reranker.py**: Simple reranking

---

## MODULE CÒN THIẾU:

❌ **Vector Store**: Lưu trữ embeddings (có thể dùng FAISS, ChromaDB, hoặc simple dict)
❌ **Generator**: LLM để sinh câu trả lời từ chunks

---

## CÁCH SỬ DỤNG:

```python
# 1. Load PDF
from loaders.pdf_loader import PDFLoader
pdf_doc = PDFLoader().load("document.pdf")

# 2. Chunk
from chunkers.chunker import SemanticChunker
chunks = SemanticChunker().chunk(pdf_doc)

# 3. Tạo index
from retriever.embedding import GemmaEmbedder
from retriever.bm25 import BM25Retriever

embedder = GemmaEmbedder()
chunk_texts = [c.content for c in chunks.chunks]
chunk_embeddings = embedder.embed(chunk_texts)

bm25 = BM25Retriever(chunk_texts)

# 4. Query với Hybrid
from retriever.hybrid import HybridRetriever

query = "What is machine learning?"
hybrid = HybridRetriever(embedder, bm25, alpha=0.5)
hybrid_scores = hybrid.score(query, chunk_texts)

# Lấy top 50
top_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)[:50]
top_chunks = [chunk_texts[i] for i in top_indices]

# 5. Rerank
from retriever.reranker import SimpleReranker

reranker = SimpleReranker(embedder)
final_results = reranker.rerank(query, top_chunks, top_n=5)

# 6. Hiển thị kết quả
for r in final_results["results"]:
    print(f"Score: {r['relevance_score']:.4f}")
    print(f"Chunk: {r['document']}")
    print()
```

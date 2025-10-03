# CÁC HOẠT ĐỘNG CỦA SIMPLE RERANKER

## 1. INPUT
```
Query: "What is machine learning?"

Documents (đã được lọc từ Hybrid Search - top 50-100):
1. "Angela Merkel was Chancellor"
2. "Machine learning is a subset of AI"
3. "Pizza is made with tomatoes"
4. "Deep learning uses neural networks"
5. "Weather is sunny today"
```

---

## 2. BƯỚC 1: Tính Embedding cho Query
```python
query_emb = embedder.embed(["What is machine learning?"])
# Kết quả: vector [0.12, -0.34, 0.56, ..., 0.78]  (ví dụ 768 chiều)
```

---

## 3. BƯỚC 2: Tính Embedding cho tất cả Documents
```python
doc_embs = embedder.embed([
    "Angela Merkel was Chancellor",
    "Machine learning is a subset of AI",
    "Pizza is made with tomatoes",
    "Deep learning uses neural networks",
    "Weather is sunny today"
])

# Kết quả:
# Doc 1: [0.23, -0.11, 0.44, ..., 0.33]
# Doc 2: [0.15, -0.32, 0.58, ..., 0.75]  <- Gần với query
# Doc 3: [-0.45, 0.67, -0.22, ..., 0.11]
# Doc 4: [0.18, -0.29, 0.52, ..., 0.71]  <- Khá gần với query
# Doc 5: [-0.33, 0.55, -0.18, ..., 0.22]
```

---

## 4. BƯỚC 3: Tính Cosine Similarity giữa Query và mỗi Document

### Công thức Cosine Similarity:
```
similarity = (A · B) / (||A|| * ||B||)

Trong đó:
- A · B: Tích vô hướng (dot product)
- ||A||: Độ dài vector A
- ||B||: Độ dài vector B
```

### Ví dụ tính toán:
```python
query_vec = [0.12, -0.34, 0.56]
doc2_vec = [0.15, -0.32, 0.58]

# Dot product
dot = 0.12*0.15 + (-0.34)*(-0.32) + 0.56*0.58
    = 0.018 + 0.1088 + 0.3248
    = 0.4516

# Norms
||query|| = sqrt(0.12² + 0.34² + 0.56²) = 0.67
||doc2||  = sqrt(0.15² + 0.32² + 0.58²) = 0.69

# Cosine similarity
similarity = 0.4516 / (0.67 * 0.69) = 0.98  (rất cao!)
```

### Kết quả cho tất cả documents:
```
Doc 1: cosine_sim(query, doc1) = 0.25  (thấp - không liên quan)
Doc 2: cosine_sim(query, doc2) = 0.98  (cao - rất liên quan!)
Doc 3: cosine_sim(query, doc3) = 0.05  (thấp - không liên quan)
Doc 4: cosine_sim(query, doc4) = 0.87  (cao - khá liên quan)
Doc 5: cosine_sim(query, doc5) = 0.12  (thấp - không liên quan)
```

---

## 5. BƯỚC 4: Sắp xếp theo điểm giảm dần
```python
results = [
    {"index": 0, "document": "Angela Merkel...", "score": 0.25},
    {"index": 1, "document": "Machine learning...", "score": 0.98},
    {"index": 2, "document": "Pizza...", "score": 0.05},
    {"index": 3, "document": "Deep learning...", "score": 0.87},
    {"index": 4, "document": "Weather...", "score": 0.12},
]

# Sau khi sort (giảm dần):
results = [
    {"index": 1, "document": "Machine learning...", "score": 0.98},  # ← Rank 1
    {"index": 3, "document": "Deep learning...", "score": 0.87},     # ← Rank 2
    {"index": 0, "document": "Angela Merkel...", "score": 0.25},     # ← Rank 3
    {"index": 4, "document": "Weather...", "score": 0.12},           # ← Rank 4
    {"index": 2, "document": "Pizza...", "score": 0.05},             # ← Rank 5
]
```

---

## 6. BƯỚC 5: Lấy top_n (ví dụ top_n=3)
```python
final_results = results[:3]

# Output:
{
    "results": [
        {"index": 1, "document": "Machine learning is a subset of AI", "score": 0.98},
        {"index": 3, "document": "Deep learning uses neural networks", "score": 0.87},
        {"index": 0, "document": "Angela Merkel was Chancellor", "score": 0.25}
    ]
}
```

---

## TẠI SAO RERANKER LẠI QUAN TRỌNG?

### Trước Reranker (chỉ dùng Hybrid):
```
Top 5 từ Hybrid (BM25 + Embedding):
1. "Angela Merkel..." (score: 0.65) ← Có từ khóa nhưng không liên quan
2. "Machine learning..." (score: 0.62)
3. "Weather..." (score: 0.58)
4. "Deep learning..." (score: 0.55)
5. "Pizza..." (score: 0.52)
```

### Sau Reranker (tính toán lại với embedding):
```
Top 5 sau Rerank:
1. "Machine learning..." (score: 0.98) ← Đúng nhất!
2. "Deep learning..." (score: 0.87)
3. "Angela Merkel..." (score: 0.25)
4. "Weather..." (score: 0.12)
5. "Pizza..." (score: 0.05)
```

**→ Reranker sửa lại thứ tự, đưa kết quả tốt nhất lên top!**

---

## SO SÁNH VỚI BM25

| Khía cạnh | BM25 | Embedding Reranker |
|-----------|------|-------------------|
| **Input** | Query text + Doc text | Query embedding + Doc embedding |
| **Tính toán** | Đếm từ, TF-IDF | Vector cosine similarity |
| **Kết quả** | Điểm dựa trên tần suất từ | Điểm dựa trên ngữ nghĩa |
| **Ví dụ** | "ML" ≠ "machine learning" | "ML" ≈ "machine learning" |

---

## ĐIỂM YẾU CỦA SIMPLE RERANKER

1. **Chỉ dùng cosine similarity** - không mạnh bằng cross-encoder
2. **Phải tính embedding lại** - chậm nếu có nhiều documents
3. **Không học được pattern** - không như BGE Reranker (được train riêng)

→ Tuy nhiên, đây là giải pháp **đơn giản, không cần model phức tạp, không cần HuggingFace**!

---

## KẾT LUẬN

**SimpleReranker hoạt động bằng cách:**
1. Chuyển query và documents thành vector (embedding)
2. Tính độ tương đồng góc (cosine similarity) giữa các vector
3. Sắp xếp lại documents theo độ tương đồng
4. Trả về top-k documents có độ tương đồng cao nhất

**Ưu điểm:** Đơn giản, hiểu ngữ nghĩa, không cần model phức tạp
**Nhược điểm:** Chậm hơn BM25, không mạnh bằng cross-encoder reranker

# RAG Evaluation Metrics Implementation Plan

## 🎯 3 Core Metrics: Ground-truth, Recall, Relevance

### 📋 Current State Analysis
- **Ground-truth**: ✅ Đã có sẵn trong database (`ground_truth_qa` table với field `source`)
- **Semantic Similarity**: ✅ Đã implement (đo độ tương đồng ngữ nghĩa)
- **Recall**: ❌ Chưa có - cần implement
- **Relevance**: ⚠️ Partial - có semantic similarity nhưng cần bổ sung

### 📝 Implementation Todo List

#### 1. **Ground-truth Infrastructure** (Tận dụng existing)
- ✅ Database table: `ground_truth_qa` với fields: `question`, `answer`, `source`
- ✅ Import functionality: Excel/CSV upload với column mapping
- ✅ UI component: GroundTruthComponent với import và evaluation buttons
- 🔄 **Cần bổ sung**: Validation logic để đảm bảo source field có nội dung

#### 2. **Recall Metric** (Implement mới)
- **Định nghĩa**: Tỷ lệ chunks liên quan được tìm thấy / tổng số chunks liên quan
- **Công thức**: `Recall = (Số chunks retrieved có semantic similarity > threshold) / (Tổng số chunks liên quan trong DB)`
- **Cách tính**:
  - Lấy tất cả chunks từ database có source tương tự ground truth
  - So sánh với retrieved chunks qua semantic similarity
  - Đếm True Positives (chunks liên quan được tìm thấy)
  - Tính tỷ lệ: TP / (TP + FN)

#### 3. **Relevance Metric** (Tận dụng + bổ sung semantic similarity)
- **Định nghĩa**: Độ liên quan của retrieved content với query
- **Tận dụng**: Semantic similarity đã có (cosine similarity với source)
- **Bổ sung**:
  - Query-chunk relevance (không chỉ source-chunk)
  - Multi-level relevance scoring (0-1 scale)
  - Relevance threshold configuration

### 🛠 Technical Implementation Plan

#### Phase 1: Enhance Backend API (`evaluation/backend_dashboard/api.py`)
1. **Thêm method `evaluate_recall()`**:
   - Input: embedder_type, reranker_type, use_qem, top_k, threshold
   - Logic: So sánh retrieved chunks với ground truth sources
   - Output: recall_score, precision, f1_score

2. **Thêm method `evaluate_relevance()`**:
   - Input: embedder_type, reranker_type, use_qem, top_k
   - Logic: Tính relevance score cho từng retrieved chunk
   - Output: avg_relevance, relevance_distribution

3. **Cập nhật `evaluate_ground_truth_with_semantic_similarity()`**:
   - Thêm recall và relevance metrics vào kết quả

#### Phase 2: Update UI Component (`ui/dashboard/components/ground_truth.py`)
1. **Thêm buttons mới**:
   - "📈 Evaluate Recall" - chạy recall evaluation
   - "🎯 Evaluate Relevance" - chạy relevance evaluation
   - "📊 Full Evaluation Suite" - chạy cả 3 metrics

2. **Thêm display components**:
   - Recall metrics dashboard (precision, recall, F1)
   - Relevance score distribution charts
   - Comparative analysis across models

#### Phase 3: Database Schema Enhancement
1. **Thêm metadata cho chunks**:
   - Chunk relevance scores
   - Source mapping cho recall calculation
   - Evaluation timestamps

#### Phase 4: Configuration & Thresholds
1. **Thêm config file**: `evaluation_config.yaml`
   - Semantic similarity thresholds
   - Relevance scoring weights
   - Recall calculation parameters

### 📊 Expected Output Metrics

#### Recall Evaluation:
```
{
  "recall_score": 0.75,        # 75% chunks liên quan được tìm thấy
  "precision": 0.82,           # 82% retrieved chunks là liên quan
  "f1_score": 0.78,            # Harmonic mean
  "true_positives": 15,        # Chunks liên quan được tìm thấy
  "false_positives": 3,        # Chunks không liên quan được retrieve
  "false_negatives": 5         # Chunks liên quan bị bỏ sót
}
```

#### Relevance Evaluation:
```
{
  "avg_relevance": 0.68,       # Trung bình relevance score
  "high_relevance_ratio": 0.45, # Tỷ lệ chunks có relevance > 0.8
  "relevance_distribution": {
    "0-0.2": 5,
    "0.2-0.4": 12,
    "0.4-0.6": 18,
    "0.6-0.8": 22,
    "0.8-1.0": 13
  }
}
```

### 🔄 Integration Points
- **Tận dụng existing retrieval pipeline**: Sử dụng `retrieval_orchestrator.py`
- **Tận dụng existing embedders**: Sử dụng `embedder_factory.py`
- **Tận dụng existing ground truth**: Sử dụng `ground_truth_qa` table
- **Tận dụng existing UI framework**: Streamlit components

### 📈 Success Criteria
1. **Ground-truth**: Import và validation hoạt động hoàn hảo
2. **Recall**: Tính toán chính xác với ground truth sources
3. **Relevance**: Semantic similarity scores có ý nghĩa
4. **UI**: Dashboard hiển thị cả 3 metrics với charts và comparisons
5. **Performance**: Evaluation chạy trong thời gian hợp lý (< 5 min cho 50 queries)

### 🚀 Next Steps
1. Bắt đầu với Phase 1: Implement recall method trong backend API
2. Test với sample data
3. Implement relevance enhancements
4. Update UI components
5. Full integration testing

---
**Priority**: Recall → Relevance → UI Enhancements</content>
<parameter name="filePath">d:\Project\RAG-2\RAG_EVALUATION_TODO.md
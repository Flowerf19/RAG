# Table Schema-Aware Chunking - Embedding Readiness Report

**Generated:** 2025-10-09  
**Status:** ✅ READY với một số cải tiến khuyến nghị

---

## Checklist Assessment

### ✅ 1. Bảng → chunk riêng (table / table_row), không trộn paragraph
**Status:** PASSED

- TableBlock được detect và chunk riêng biệt với `group_type: 'table'`
- Không bị merge với paragraph chunks
- Evidence: Chunk #2 trong output có `group_type: 'table'`, riêng biệt với Chunk #1 (paragraph)

```python
# RuleBasedChunker._group_blocks() line 89-91
if btype in {"table", "code"}:
    if cur_blocks:
        groups.append((cur_type, cur_blocks, cur_title))
    groups.append((btype, [b], cur_title))  # Table chunk riêng
```

### ✅ 2. Lưu payload có cấu trúc trong metadata.table_payload
**Status:** PASSED

- TableSchema được lưu đầy đủ trong `chunk.metadata['table_payload']`
- Bao gồm: header, rows (với cells), bbox, caption, stable_id, content_sha256
- Evidence:
```python
'table_payload': TableSchema(
    id='1d6a79d7-017d-4b94-8b1b-4a94ca1f143e',
    header=['date', 'version', 'description', 'author', 'reviewer'],
    rows=[TableRow(cells=[...])],
    bbox=(73.89, 133.22, 552.93, 310.00),
    metadata={'table_caption': 'table 2.1 - process characteristics'},
    stable_id='6d8b340bf2c1a9a4ba35ed6d',
    content_sha256='a0d6aed3f...'
)
```

### ✅ 3. Sinh textForEmbedding "schema-aware" (Markdown/CSV/KV) deterministic
**Status:** PASSED

- `embedding_text` được sinh tự động theo format KV (pipe-separated)
- Format deterministic: `header | header | header\nval | val | val`
- Evidence:
```python
'embedding_text': 'date | version | description | author | reviewer\n10-05-2024 | 1.0 | issued version |  | '
```

**Implementation:**
```python
# rule_based_chunker.py line 243-255
if hasattr(table_schema, "markdown") and table_schema.markdown:
    metadata["embedding_text"] = table_schema.markdown
else:
    rows = getattr(table_schema, "rows", [])
    header = getattr(table_schema, "header", [])
    embedding_lines = []
    for r in rows:
        if hasattr(r, "cells"):
            line = " | ".join([str(c.value) for c in r.cells])
            embedding_lines.append(line)
    metadata["embedding_text"] = "\n".join([" | ".join(header)] + embedding_lines)
```

### ⚠️ 4. Áp token budget + quy tắc cắt gọt ngay ở chunking
**Status:** PARTIAL - Cần cải tiến

**Hiện tại:**
- Token counting có (`token_count` được tính)
- Max token limit được check trong HybridChunker (max_tokens=200)
- Nhưng CHƯA có logic cắt gọt table nếu vượt quá token budget

**Issue:** Bảng lớn có thể vượt quá max_tokens mà không được cắt

**Recommended Fix:**
```python
def _build_chunk(self, text, blocks, gtype, ...):
    # ... existing code ...
    
    if gtype == "table" and table_schema:
        # Check token budget
        estimated_tokens = self.estimate_tokens(table_text)
        if estimated_tokens > self.max_tokens:
            # Apply row truncation strategy
            rows = table_schema.rows[:N]  # Keep first N rows
            metadata["truncated"] = True
            metadata["original_row_count"] = len(table_schema.rows)
            metadata["kept_row_count"] = N
```

### ❌ 5. Lưu cell-level provenance nếu có
**Status:** NOT IMPLEMENTED

**Hiện tại:**
- Provenance chỉ có block-level (BlockSpan)
- KHÔNG có cell-level provenance (row, col)

**Current Provenance:**
```python
ProvenanceAgg(
    source_blocks=['0b80e483b9cf951be40f7188'],
    spans=[BlockSpan(block_id='...', start_char=0, end_char=86, page_number=1)],
    page_numbers={1},
    doc_id='unknown'
)
```

**Recommended Enhancement:**
```python
# Extend BlockSpan or add TableCellSpan
@dataclass
class TableCellSpan:
    block_id: str
    row_idx: int
    col_idx: int
    cell_value: str
    page_number: Optional[int] = None

# In chunk.metadata
metadata["cell_provenance"] = [
    {"row": 1, "col": 0, "value": "10-05-2024", "page": 1},
    {"row": 1, "col": 1, "value": "1.0", "page": 1},
    # ...
]
```

### ❌ 6. Ghi table_text_mode + row_priority vào metadata để audit
**Status:** NOT IMPLEMENTED

**Hiện tại:**
- Không có `table_text_mode` trong metadata
- Không có `row_priority` hoặc row selection strategy

**Recommended Addition:**
```python
metadata.update({
    "table_text_mode": "pipe_separated",  # or "markdown", "csv"
    "row_priority": "sequential",  # or "header_priority", "truncated"
    "header_included": True,
    "row_range": [0, len(rows)],  # [start_idx, end_idx]
    "generation_strategy": "auto"
})
```

---

## Current Embedding Flow Assessment

### ✅ Strengths

1. **Clean Separation:** Tables are properly isolated in separate chunks
2. **Structured Payload:** Full TableSchema preserved for downstream use
3. **Deterministic Text:** `embedding_text` is consistent and reproducible
4. **Caption Extraction:** Table captions are properly extracted and stored
5. **Metadata Rich:** Chunks contain comprehensive metadata

### ⚠️ Weaknesses & Recommendations

1. **Large Table Handling:** 
   - **Issue:** No truncation strategy for tables exceeding token budget
   - **Impact:** May cause embedding errors or truncation at embedding layer
   - **Fix Priority:** HIGH

2. **Cell-Level Provenance:**
   - **Issue:** Cannot trace back to specific cells after retrieval
   - **Impact:** Less precise citation in RAG responses
   - **Fix Priority:** MEDIUM

3. **Audit Trail:**
   - **Issue:** No metadata about text generation method
   - **Impact:** Hard to debug or optimize different table formats
   - **Fix Priority:** LOW

4. **No Chunk.textForEmbedding Property:**
   - **Issue:** `embedding_text` only in metadata, not as first-class property
   - **Impact:** Embedder needs to check both `chunk.text` and `chunk.metadata['embedding_text']`
   - **Fix Priority:** MEDIUM

---

## Embedding Integration Checklist

### For Embedder Implementation:

```python
def get_text_for_embedding(chunk: Chunk) -> str:
    """Get appropriate text for embedding from chunk."""
    
    # For table chunks, use embedding_text if available
    if chunk.metadata.get('group_type') == 'table':
        embedding_text = chunk.metadata.get('embedding_text')
        if embedding_text:
            return embedding_text
    
    # For regular chunks, use chunk.text
    return chunk.text

def embed_chunk(chunk: Chunk, embedder) -> dict:
    """Embed a chunk and return result with metadata."""
    
    text = get_text_for_embedding(chunk)
    
    # Check token limit
    if chunk.token_count > embedder.max_tokens:
        logger.warning(f"Chunk {chunk.chunk_id} exceeds token limit")
        # Handle truncation at embedding layer as fallback
    
    embedding = embedder.embed(text)
    
    return {
        'chunk_id': chunk.chunk_id,
        'embedding': embedding,
        'text': text,  # Text that was actually embedded
        'metadata': chunk.metadata,
        'table_payload': chunk.metadata.get('table_payload'),  # Preserve structure
    }
```

### For Retrieval:

```python
def retrieve_and_format(query_embedding, chunk_db):
    """Retrieve chunks and format for display."""
    
    results = chunk_db.similarity_search(query_embedding, k=5)
    
    formatted_results = []
    for chunk_id, similarity in results:
        chunk = chunk_db.get_chunk(chunk_id)
        
        if chunk.metadata.get('group_type') == 'table':
            # Reconstruct table from table_payload
            table_payload = chunk.metadata.get('table_payload')
            formatted_text = format_table_as_markdown(table_payload)
            
            formatted_results.append({
                'text': formatted_text,  # Display as table
                'type': 'table',
                'caption': table_payload.metadata.get('table_caption'),
                'similarity': similarity
            })
        else:
            formatted_results.append({
                'text': chunk.text,
                'type': 'text',
                'similarity': similarity
            })
    
    return formatted_results
```

---

## Test Verification

### Sample Table Chunk:
```
Chunk ID: [generated]
Group Type: table
Token Count: 28
Content (text): Date | Version | Description | Author | Reviewer\n10-05-2024 | 1.0 | Issued version | |
Embedding Text: date | version | description | author | reviewer\n10-05-2024 | 1.0 | issued version |  | 
Table Payload: TableSchema(header=[...], rows=[...], bbox=(...))
Caption: None
Provenance: page=1, block_id=0b80e483b9cf951be40f7188
```

### Embedding Simulation:
```python
# Text sent to embedder
text_for_embedding = "date | version | description | author | reviewer\n10-05-2024 | 1.0 | issued version |  | "

# After embedding
vector = embedder.embed(text_for_embedding)  # [0.123, -0.456, ...]

# After retrieval
retrieved_chunk = get_chunk_by_id(chunk_id)
table_payload = retrieved_chunk.metadata['table_payload']

# Reconstruct full table
display_table = format_table(table_payload)
# → Can display original table structure with all cells
```

---

## Final Verdict

### ✅ READY FOR EMBEDDING with caveats:

**Can proceed with embedding now:**
- Table chunks are properly isolated
- Structured payload is preserved
- Embedding text is deterministic
- Table structure can be reconstructed after retrieval

**Recommended improvements before production:**
1. **HIGH Priority:** Implement token budget enforcement with table row truncation
2. **MEDIUM Priority:** Add `textForEmbedding` as Chunk property
3. **MEDIUM Priority:** Add cell-level provenance for precise citation
4. **LOW Priority:** Add audit metadata (table_text_mode, row_priority)

**Workaround for current limitations:**
- Embedder should implement fallback truncation for oversized tables
- Document that cell-level citation is not yet available
- Monitor for tables exceeding token limits in logs

---

## Next Steps

1. **Test with embedder:** Run actual embedding on chunk_output.txt
2. **Verify retrieval:** Test that table_payload survives embedding → storage → retrieval
3. **Implement HIGH priority fixes:** Token budget enforcement
4. **Create embedding utilities:** Helper functions for get_text_for_embedding()
5. **Integration testing:** End-to-end RAG pipeline with table queries


# TODO: Cập nhật Block.normalize() - Improvement Roadmap

## Phase 1: Basic Filtering & Cleaning (Priority: HIGH)

- [ ] **1.1** Lọc block header/footer lặp lại
  - Phát hiện block có nội dung lặp lại trên nhiều trang
  - Lọc block chứa keyword: "Classification", "Owner", "Company", "Version", "ISMS", "QMS"
  - Thêm tham số `repeated_block_hashes` vào normalize để track

- [ ] **1.2** Lọc block chỉ chứa số trang hoặc trống
  - Lọc block match pattern: "Page X/Y", "Page X of Y"
  - Lọc block chỉ có whitespace/newline (sau strip())
  - Lọc block có độ dài text < ngưỡng (ví dụ < 3 ký tự)

- [ ] **1.3** Chuẩn hóa whitespace và unicode
  - Áp dụng ftfy.fix_text() (đã có)
  - Áp dụng clean-text (đã có)
  - Normalize multiple spaces/newlines thành single
  - Loại bỏ zero-width characters (đã có)

## Phase 2: Advanced Text Processing (Priority: MEDIUM)

- [ ] **2.1** Tích hợp spaCy sentence splitting
  - Tách câu cho block text dài (đã có)
  - Thêm language detection nếu cần (vi/en auto-detect)
  - Cache spaCy model để tăng performance

- [ ] **2.2** De-hyphenation nâng cao
  - Xử lý từ bị ngắt dòng với dấu gạch ngang (đã có cơ bản)
  - Xử lý trường hợp đặc biệt: từ ghép hợp pháp
  - Validate sau khi dehyphenate bằng dictionary

- [ ] **2.3** Loại bỏ repeated characters
  - Phát hiện và normalize repeated chars (e.g., "helllllo" -> "hello")
  - Áp dụng threshold cho repeated space/tab

## Phase 3: Bbox & Layout Analysis (Priority: MEDIUM)

- [ ] **3.1** Lọc block có bbox quá nhỏ
  - Tính area = (x1-x0) * (y1-y0)
  - Lọc block có area < ngưỡng (noise)
  - Lọc block có width hoặc height < ngưỡng

- [ ] **3.2** Phát hiện block ngoài vùng nội dung
  - Xác định margin/boundary của trang
  - Lọc block ở header/footer area (dựa vào y-coordinate)
  - Lọc block ở sidebar/watermark area

- [ ] **3.3** Gộp block liền kề
  - Phát hiện block liền kề theo y-axis (cùng dòng)
  - Gộp block nếu khoảng cách < ngưỡng
  - Rebuild bbox sau khi merge

## Phase 4: Content Type Detection (Priority: LOW)

- [ ] **4.1** Phân loại block type
  - Detect block type: header, title, paragraph, list, table_fragment
  - Thêm trường `block_type` vào metadata
  - Xử lý khác nhau theo type

- [ ] **4.2** Extract structured information
  - Phát hiện numbered list items (1., 2., 3., ...)
  - Phát hiện bullet points (•, -, *, ...)
  - Extract dates, emails, URLs (nếu cần)

## Phase 5: Performance & Optimization (Priority: LOW)

- [ ] **5.1** Optimize normalize performance
  - Cache repeated regex compile
  - Batch processing cho multiple blocks
  - Lazy evaluation cho expensive operations

- [ ] **5.2** Add configuration support
  - Thêm config params cho mọi threshold
  - Support enable/disable từng feature
  - Profile và tune default values

## Implementation Notes

- Mỗi bước nên có unit test riêng
- Mỗi feature nên có flag enable/disable trong config
- Log warning khi filter block để debug
- Maintain backward compatibility với existing code

## Current Status

✅ Basic text normalization (ftfy, clean-text)
✅ De-hyphenation cơ bản
✅ Zero-width character removal
✅ Basic bbox normalization
✅ spaCy sentence tokenization
⏳ Header/footer filtering (TODO)
⏳ Page number filtering (TODO)
⏳ Bbox-based filtering (TODO)

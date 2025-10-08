# PDFLoader Usage Examples

Sau khi refactor, PDFLoader đã được chuẩn hóa theo OOP và loại bỏ dependency vào YAML config.

## Cách sử dụng cơ bản

### 1. Sử dụng với cấu hình mặc định

```python
from loaders import PDFLoader

# Sử dụng factory method với cấu hình mặc định (equivalent với YAML cũ)
loader = PDFLoader.create_default()

# Hoặc khởi tạo trực tiếp với default values
loader = PDFLoader()  # Tất cả parameters đều có default values

# Load PDF
document = loader.load("path/to/file.pdf")
```

### 2. Sử dụng với cấu hình tùy chỉnh

```python
# Khởi tạo với cấu hình custom
loader = PDFLoader(
    extract_text=True,
    extract_tables=False,  # Không trích xuất bảng
    min_repeated_text_threshold=5,  # Tăng ngưỡng lặp lại
    min_text_length=20,  # Text tối thiểu 20 chars
    enable_repeated_block_filter=False  # Tắt filter block lặp lại
)

document = loader.load("path/to/file.pdf")
```

### 3. Sử dụng factory methods

```python
# Chỉ trích xuất text
text_loader = PDFLoader.create_text_only()

# Chỉ trích xuất bảng
table_loader = PDFLoader.create_tables_only()

# Cấu hình mặc định
default_loader = PDFLoader.create_default()
```

## Quản lý cấu hình runtime

### 1. Xem cấu hình hiện tại

```python
loader = PDFLoader.create_default()

# Xem tất cả cấu hình
config = loader.get_config()
print("Current config:", config)

# Xem thông tin loader
print("Loader info:", repr(loader))
```

### 2. Cập nhật cấu hình runtime

```python
loader = PDFLoader.create_default()

# Cập nhật một số thuộc tính
loader.update_config(
    extract_tables=False,
    min_text_length=15,
    tables_engine="camelot"
)

# Bật/tắt tất cả filters
loader.enable_all_filters()
loader.disable_all_filters()
```

## So sánh với cách cũ

### Cách mới (OOP chuẩn):

```python
# ✅ New way - dependency injection, flexible, testable
loader = PDFLoader(
    extract_text=True,
    extract_tables=True,
    min_repeated_text_threshold=3
)

# Hoặc sử dụng factory methods
loader = PDFLoader.create_default()
```

## Lợi ích của cách mới

1. **Chuẩn OOP**: Tất cả thuộc tính được định nghĩa rõ ràng trong class
2. **Dependency Injection**: Có thể inject config từ bên ngoài
3. **Testability**: Dễ dàng mock và test với các config khác nhau
4. **Type Safety**: Type hints rõ ràng cho tất cả parameters
5. **Flexibility**: Có thể tạo nhiều instance với config khác nhau
6. **Runtime Configuration**: Có thể thay đổi config trong runtime
7. **No External Dependencies**: Không phụ thuộc vào file YAML
8. **Factory Methods**: Các preset configs phổ biến
9. **Validation**: Validate config ngay khi khởi tạo
10. **Debugging**: Dễ dàng debug và inspect config

## Migration từ code cũ

Nếu bạn có code cũ sử dụng YAML config:

```python
# Old code
loader = PDFLoader()  # Loads from YAML

# New equivalent code
loader = PDFLoader.create_default()  # Same behavior
```

Hoặc chuyển sang cấu hình explicit:

```python
# New explicit configuration
loader = PDFLoader(
    extract_text=True,
    extract_tables=True,
    tables_engine="auto",
    min_repeated_text_threshold=3,
    # ... other specific settings
)
```

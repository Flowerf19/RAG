# RAG System - Copilot Instructions

## Project Overview
Đây là hệ thống RAG (Retrieval-Augmented Generation) được thiết kế theo chuẩn OOP với focus vào PDF processing và document loading.

## Development Environment Setup

### 1. Virtual Environment Setup
```powershell
# Activate virtual environment
& C:/Users/ENGUYEHWC/Downloads/RAG/RAG/.venv/Scripts/Activate.ps1

# Verify activation (should show (.venv) in prompt)
# Install dependencies if needed
pip install -r requirements.txt
```

### 2. Project Structure
```
RAG/
├── .venv/                          # Virtual environment
├── loaders/                        # PDF loading module (CURRENT FOCUS)
│   ├── __init__.py
│   ├── pdf_loader.py              # Main PDFLoader class (OOP refactored)
│   ├── config.py                  # Config management (deprecated)
│   ├── model/                     # Data models
│   └── normalizers/               # Data normalization utilities
├── chunkers/                      # Text chunking (NOT CURRENT FOCUS)
├── tests/                         # Test directory
│   └── test_loader.py            # Loader tests
├── requirements.txt
└── test_pdfloader_refactor.py    # Manual test file
```

## Current Development Focus: LOADERS ONLY

### PDFLoader Class (Refactored to OOP)
- **Location**: `loaders/pdf_loader.py`
- **Design**: Single class với dependency injection
- **Config**: No YAML dependencies, all config as constructor parameters
- **Features**:
  - PDF text extraction
  - Table extraction with multiple engines
  - Block filtering capabilities
  - Caption assignment
  - Factory methods for common configurations

### Key Design Principles Applied:
1. **Single Responsibility**: PDFLoader chỉ load và parse PDF
2. **Dependency Injection**: Config được inject qua constructor
3. **Factory Pattern**: `create_default()`, `create_text_only()`, `create_tables_only()`
4. **OOP Encapsulation**: Utility functions thành static methods
5. **Configuration Management**: Runtime config updates

## Testing Guidelines

### Running Tests
```powershell
# Make sure venv is activated
& C:/Users/ENGUYEHWC/Downloads/RAG/RAG/.venv/Scripts/Activate.ps1

# Run pytest from project root
cd C:\Users\ENGUYEHWC\Downloads\RAG\RAG
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_loader.py -v

# Run with coverage
python -m pytest tests/ --cov=loaders --cov-report=html
```

### Test Structure
- **Location**: `tests/test_loader.py`
- **Framework**: pytest
- **Design**: Single test class `TestPDFLoader`
- **Coverage**: PDFLoader initialization, config management, static methods

### Manual Test
```powershell
# Run manual test
python test_pdfloader_refactor.py
```

## Code Standards

### OOP Guidelines
1. **Class-First Design**: All functionality trong classes
2. **No Global Functions**: Utility functions thành static methods
3. **Clear Interfaces**: Type hints cho all methods
4. **Validation**: Config validation trong constructor
5. **Factory Methods**: For common use cases

### Testing Standards
1. **Single Test Class**: One class per module under test
2. **Descriptive Names**: Test methods describe what they test
3. **Setup/Teardown**: Use pytest fixtures
4. **Mocking**: Mock external dependencies
5. **Coverage**: Aim for >90% coverage

## Current Development Tasks

### ✅ Completed
- [x] Refactored PDFLoader to pure OOP
- [x] Removed YAML config dependency
- [x] Added factory methods
- [x] Moved utility functions to static methods
- [x] Added config validation
- [x] Updated all usage examples

### 🔄 In Progress
- [ ] Complete pytest test suite for PDFLoader
- [ ] Add proper test fixtures
- [ ] Test coverage reporting

### 📋 TODO (Loader Module Only)
- [ ] Performance benchmarking
- [ ] Memory usage optimization
- [ ] Error handling improvements
- [ ] Documentation completion

## DO NOT WORK ON
- chunkers/ module
- retriever/ module  
- pipeline.py integration
- UI components
- Other modules outside loaders/

## Development Commands Quick Reference

```powershell
# Environment
& C:/Users/ENGUYEHWC/Downloads/RAG/RAG/.venv/Scripts/Activate.ps1

# Testing
python -m pytest tests/test_loader.py -v
python test_pdfloader_refactor.py

# Code Quality
python -m pylint loaders/pdf_loader.py
python -m mypy loaders/pdf_loader.py

# Dependencies
pip list
pip install -r requirements.txt
```

## Usage Examples

### Basic Usage
```python
from loaders import PDFLoader

# Default configuration
loader = PDFLoader.create_default()
document = loader.load("path/to/file.pdf")

# Custom configuration
loader = PDFLoader(
    extract_text=True,
    extract_tables=False,
    min_repeated_text_threshold=5
)
document = loader.load("path/to/file.pdf")
```

### Testing Usage
```python
# In tests
loader = PDFLoader(extract_text=True, extract_tables=False)
assert loader.extract_text == True
assert loader.extract_tables == False
```
# Testing Guide - Loader Module Only

## Quick Start

### 1. Activate Virtual Environment
```powershell
& C:/Users/ENGUYEHWC/Downloads/RAG/RAG/.venv/Scripts/Activate.ps1
```

### 2. Install Test Dependencies
```powershell
pip install pytest pytest-cov pytest-mock
```

### 3. Run Tests
```powershell
# Basic test run
python -m pytest tests/test_loader.py -v

# With coverage
python -m pytest tests/test_loader.py --cov=loaders --cov-report=term-missing

# Quick run (quiet mode)
python -m pytest tests/test_loader.py -q
```

### 4. Using PowerShell Script
```powershell
# Basic run
.\run_tests.ps1

# With coverage
.\run_tests.ps1 -Coverage

# Verbose with coverage
.\run_tests.ps1 -Verbose -Coverage
```

## Test Structure

### Single Test Class Design
```
tests/
└── test_loader.py          # Single class TestPDFLoader
    ├── Initialization tests
    ├── Factory method tests
    ├── Configuration tests
    ├── Static method tests
    ├── Integration tests
    └── Edge case tests
```

### Test Categories

1. **Initialization Tests**
   - Default parameters
   - Custom parameters
   - Validation errors

2. **Factory Methods**
   - `create_default()`
   - `create_text_only()`
   - `create_tables_only()`

3. **Configuration Management**
   - `get_config()`
   - `update_config()`
   - Filter management

4. **Static Methods**
   - `_extract_leading_number()`
   - `_make_row()`
   - `_rebuild_markdown()`
   - `_reindex_rows()`

5. **Integration Tests**
   - File operations (mocked)
   - PDF loading (mocked)

## Expected Output

### Successful Test Run
```
================================ test session starts ================================
tests/test_loader.py::TestPDFLoader::test_default_initialization PASSED      [ 8%]
tests/test_loader.py::TestPDFLoader::test_custom_initialization PASSED       [16%]
tests/test_loader.py::TestPDFLoader::test_validation_on_initialization PASSED [25%]
...
tests/test_loader.py::TestPDFLoader::test_config_consistency PASSED          [100%]

================================ 25 passed in 2.34s ================================
```

### With Coverage
```
Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
loaders/__init__.py          3      0   100%
loaders/pdf_loader.py      180     12    93%   45-48, 234-237
------------------------------------------------------
TOTAL                      183     12    93%
```

## Focus Areas

### ✅ What We Test
- PDFLoader class initialization
- Configuration management
- Factory methods
- Static utility methods
- Basic integration (mocked)

### ❌ What We DON'T Test
- chunkers/ module
- retriever/ module
- pipeline.py integration
- UI components
- Actual PDF file processing (use mocks)

## Troubleshooting

### Common Issues

1. **Virtual Environment Not Activated**
   ```
   Error: python: command not found
   ```
   **Solution**: Activate venv first

2. **Missing Dependencies**
   ```
   ModuleNotFoundError: No module named 'pytest'
   ```
   **Solution**: `pip install pytest pytest-cov pytest-mock`

3. **Import Errors**
   ```
   ImportError: No module named 'loaders'
   ```
   **Solution**: Run from project root directory

### Debug Mode
```powershell
# Run with full traceback
python -m pytest tests/test_loader.py -v --tb=long

# Run specific test
python -m pytest tests/test_loader.py::TestPDFLoader::test_default_initialization -v

# Run with prints visible
python -m pytest tests/test_loader.py -v -s
```

## Test Coverage Goals

- **Target**: >90% coverage for loaders/pdf_loader.py
- **Focus**: Core functionality, not edge cases
- **Priority**: Constructor, factory methods, config management
#!/usr/bin/env python3
"""
Test file để validate PDFLoader refactor hoạt động đúng.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from loaders.pdf_loader import PDFLoader

def test_pdfloader_initialization():
    """Test khởi tạo PDFLoader với các cách khác nhau."""
    print("=== Testing PDFLoader Initialization ===")
    
    # Test default constructor
    loader1 = PDFLoader()
    print(f"✓ Default constructor: {repr(loader1)}")
    
    # Test factory methods
    loader2 = PDFLoader.create_default()
    print(f"✓ Factory default: {repr(loader2)}")
    
    loader3 = PDFLoader.create_text_only()
    print(f"✓ Factory text-only: {repr(loader3)}")
    
    loader4 = PDFLoader.create_tables_only()
    print(f"✓ Factory tables-only: {repr(loader4)}")
    
    # Test custom config
    loader5 = PDFLoader(
        extract_text=True,
        extract_tables=False,
        min_repeated_text_threshold=5,
        tables_engine="camelot"
    )
    print(f"✓ Custom config: {repr(loader5)}")

def test_config_management():
    """Test quản lý cấu hình runtime."""
    print("\n=== Testing Config Management ===")
    
    loader = PDFLoader.create_default()
    
    # Get current config
    config = loader.get_config()
    print(f"✓ Current config keys: {list(config.keys())}")
    
    # Update config
    loader.update_config(min_text_length=15, extract_tables=False)
    print(f"✓ Updated config - min_text_length: {loader.min_text_length}")
    print(f"✓ Updated config - extract_tables: {loader.extract_tables}")
    
    # Test filter management
    loader.disable_all_filters()
    print(f"✓ All filters disabled - repeated_block_filter: {loader.enable_repeated_block_filter}")
    
    loader.enable_all_filters()
    print(f"✓ All filters enabled - repeated_block_filter: {loader.enable_repeated_block_filter}")

def test_static_methods():
    """Test các static utility methods."""
    print("\n=== Testing Static Utility Methods ===")
    
    # Test _extract_leading_number
    assert PDFLoader._extract_leading_number("123 abc") == 123
    assert PDFLoader._extract_leading_number("  456") == 456
    assert PDFLoader._extract_leading_number("abc 123") is None
    assert PDFLoader._extract_leading_number("") is None
    print("✓ _extract_leading_number works correctly")
    
    # Test _rebuild_markdown
    from loaders.model.table import TableRow, TableCell
    
    # Create test data
    header = ["Col1", "Col2", "Col3"]
    cells1 = [
        TableCell(value="A", row=1, col=1, bbox=None, metadata={}),
        TableCell(value="B", row=1, col=2, bbox=None, metadata={}),
        TableCell(value="C", row=1, col=3, bbox=None, metadata={})
    ]
    row1 = TableRow(cells=cells1, row_idx=1)
    
    markdown = PDFLoader._rebuild_markdown(header, [row1])
    expected_lines = [
        "| Col1 | Col2 | Col3 |",
        "|---|---|---|",
        "| A | B | C |"
    ]
    assert all(line in markdown for line in expected_lines)
    print("✓ _rebuild_markdown works correctly")

def test_validation():
    """Test validation logic."""
    print("\n=== Testing Validation ===")
    
    try:
        # Should raise ValueError
        PDFLoader(min_repeated_text_threshold=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Validation caught invalid min_repeated_text_threshold: {e}")
    
    try:
        # Should raise ValueError
        PDFLoader(min_text_length=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Validation caught invalid min_text_length: {e}")
    
    # Valid config should work
    loader = PDFLoader(min_repeated_text_threshold=1, min_text_length=0)
    print("✓ Valid config accepted")

def main():
    """Run all tests."""
    print("Testing PDFLoader Refactor")
    print("=" * 50)
    
    try:
        test_pdfloader_initialization()
        test_config_management()
        test_static_methods()
        test_validation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! PDFLoader refactor is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
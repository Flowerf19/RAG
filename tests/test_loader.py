"""
Test suite for PDFLoader class.
Single class design theo chuẩn OOP, focus vào loaders module only.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import the class under test
from loaders.pdf_loader import PDFLoader
from loaders.model.document import PDFDocument
from loaders.model.page import PDFPage
from loaders.model.table import TableSchema, TableRow, TableCell


class TestPDFLoader:
    """
    Single test class for PDFLoader theo chuẩn OOP.
    Covers initialization, configuration, static methods, và core functionality.
    """
    
    # ========== SETUP/TEARDOWN ==========
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup cho mỗi test method."""
        self.default_config = {
            'extract_text': True,
            'extract_tables': True,
            'tables_engine': 'auto',
            'min_repeated_text_threshold': 3,
            'min_text_length': 10,
            'repeated_block_threshold': 3,
            'enable_repeated_block_filter': True,
            'enable_position_filter': True,
            'enable_page_number_filter': True,
            'enable_empty_filter': True,
            'enable_bbox_filter': True,
            'min_bbox_area': 10.0
        }
    
    # ========== INITIALIZATION TESTS ==========
    
    def test_default_initialization(self):
        """Test khởi tạo PDFLoader với default parameters."""
        loader = PDFLoader()
        
        assert loader.extract_text == True
        assert loader.extract_tables == True
        assert loader.tables_engine == "auto"
        assert loader.min_repeated_text_threshold == 3
        assert loader.min_text_length == 10
        assert loader.repeated_block_threshold == 3
        assert loader.enable_repeated_block_filter == True
        assert loader.enable_position_filter == True
        assert loader.enable_page_number_filter == True
        assert loader.enable_empty_filter == True
        assert loader.enable_bbox_filter == True
        assert loader.min_bbox_area == 10.0
    
    def test_custom_initialization(self):
        """Test khởi tạo PDFLoader với custom parameters."""
        loader = PDFLoader(
            extract_text=False,
            extract_tables=True,
            tables_engine="camelot",
            min_repeated_text_threshold=5,
            min_text_length=20,
            enable_repeated_block_filter=False
        )
        
        assert loader.extract_text == False
        assert loader.extract_tables == True
        assert loader.tables_engine == "camelot"
        assert loader.min_repeated_text_threshold == 5
        assert loader.min_text_length == 20
        assert loader.enable_repeated_block_filter == False
    
    def test_validation_on_initialization(self):
        """Test validation khi khởi tạo với invalid parameters."""
        
        # Test invalid min_repeated_text_threshold
        with pytest.raises(ValueError, match="min_repeated_text_threshold must be >= 1"):
            PDFLoader(min_repeated_text_threshold=0)
        
        # Test invalid min_text_length
        with pytest.raises(ValueError, match="min_text_length must be >= 0"):
            PDFLoader(min_text_length=-1)
        
        # Test invalid repeated_block_threshold
        with pytest.raises(ValueError, match="repeated_block_threshold must be >= 1"):
            PDFLoader(repeated_block_threshold=0)
        
        # Test invalid min_bbox_area
        with pytest.raises(ValueError, match="min_bbox_area must be >= 0"):
            PDFLoader(min_bbox_area=-1)
    
    def test_tables_engine_validation(self):
        """Test validation cho tables_engine parameter."""
        # Unknown engine should fallback to 'auto'
        loader = PDFLoader(tables_engine="unknown_engine")
        assert loader.tables_engine == "auto"
    
    # ========== FACTORY METHODS TESTS ==========
    
    def test_create_default_factory(self):
        """Test factory method create_default()."""
        loader = PDFLoader.create_default()
        
        assert loader.extract_text == True
        assert loader.extract_tables == True
        assert loader.tables_engine == "auto"
        assert loader.min_repeated_text_threshold == 3
        assert loader.min_text_length == 10
    
    def test_create_text_only_factory(self):
        """Test factory method create_text_only()."""
        loader = PDFLoader.create_text_only()
        
        assert loader.extract_text == True
        assert loader.extract_tables == False
        assert loader.tables_engine == "auto"
        assert loader.min_repeated_text_threshold == 3
        assert loader.min_text_length == 10
    
    def test_create_tables_only_factory(self):
        """Test factory method create_tables_only()."""
        loader = PDFLoader.create_tables_only()
        
        assert loader.extract_text == False
        assert loader.extract_tables == True
        assert loader.tables_engine == "auto"
        assert loader.min_repeated_text_threshold == 3
    
    # ========== CONFIGURATION MANAGEMENT TESTS ==========
    
    def test_get_config(self):
        """Test method get_config()."""
        loader = PDFLoader(extract_text=False, min_text_length=15)
        config = loader.get_config()
        
        assert isinstance(config, dict)
        assert config['extract_text'] == False
        assert config['min_text_length'] == 15
        assert config['extract_tables'] == True  # default value
        
        # Check all expected keys are present
        expected_keys = {
            'extract_text', 'extract_tables', 'tables_engine',
            'min_repeated_text_threshold', 'min_text_length', 'repeated_block_threshold',
            'enable_repeated_block_filter', 'enable_position_filter', 'enable_page_number_filter',
            'enable_empty_filter', 'enable_bbox_filter', 'min_bbox_area'
        }
        assert set(config.keys()) == expected_keys
    
    def test_update_config(self):
        """Test method update_config()."""
        loader = PDFLoader()
        
        # Update valid config
        loader.update_config(
            extract_tables=False,
            min_text_length=25,
            tables_engine="pdfplumber"
        )
        
        assert loader.extract_tables == False
        assert loader.min_text_length == 25
        assert loader.tables_engine == "pdfplumber"
    
    def test_update_config_validation(self):
        """Test validation trong update_config()."""
        loader = PDFLoader()
        
        # Invalid config should raise error after validation
        with pytest.raises(ValueError):
            loader.update_config(min_repeated_text_threshold=-1)
    
    def test_filter_management(self):
        """Test enable/disable all filters."""
        loader = PDFLoader()
        
        # Disable all filters
        loader.disable_all_filters()
        assert loader.enable_repeated_block_filter == False
        assert loader.enable_position_filter == False
        assert loader.enable_page_number_filter == False
        assert loader.enable_empty_filter == False
        assert loader.enable_bbox_filter == False
        
        # Enable all filters
        loader.enable_all_filters()
        assert loader.enable_repeated_block_filter == True
        assert loader.enable_position_filter == True
        assert loader.enable_page_number_filter == True
        assert loader.enable_empty_filter == True
        assert loader.enable_bbox_filter == True
    
    # ========== STATIC METHODS TESTS ==========
    
    def test_extract_leading_number_static_method(self):
        """Test static method _extract_leading_number()."""
        
        # Valid cases
        assert PDFLoader._extract_leading_number("123 abc") == 123
        assert PDFLoader._extract_leading_number("  456") == 456
        assert PDFLoader._extract_leading_number("789") == 789
        assert PDFLoader._extract_leading_number("0 test") == 0
        
        # Invalid cases
        assert PDFLoader._extract_leading_number("abc 123") is None
        assert PDFLoader._extract_leading_number("") is None
        assert PDFLoader._extract_leading_number("   ") is None
        assert PDFLoader._extract_leading_number("no numbers here") is None
        assert PDFLoader._extract_leading_number(None) is None
        assert PDFLoader._extract_leading_number(123) is None  # not string
    
    def test_make_row_static_method(self):
        """Test static method _make_row()."""
        values = ["A", "B", "C"]
        row = PDFLoader._make_row(values, 1)
        
        assert isinstance(row, TableRow)
        assert row.row_idx == 1
        assert len(row.cells) == 3
        
        # Check cells
        assert row.cells[0].value == "A"
        assert row.cells[0].row == 1
        assert row.cells[0].col == 1
        
        assert row.cells[1].value == "B"
        assert row.cells[1].row == 1
        assert row.cells[1].col == 2
        
        assert row.cells[2].value == "C"
        assert row.cells[2].row == 1
        assert row.cells[2].col == 3
    
    def test_rebuild_markdown_static_method(self):
        """Test static method _rebuild_markdown()."""
        header = ["Col1", "Col2", "Col3"]
        
        # Create test row
        cells = [
            TableCell(value="A", row=1, col=1, bbox=None, metadata={}),
            TableCell(value="B", row=1, col=2, bbox=None, metadata={}),
            TableCell(value="C", row=1, col=3, bbox=None, metadata={})
        ]
        row = TableRow(cells=cells, row_idx=1)
        
        markdown = PDFLoader._rebuild_markdown(header, [row])
        
        # Check expected content
        assert "| Col1 | Col2 | Col3 |" in markdown
        assert "|---|---|---|" in markdown
        assert "| A | B | C |" in markdown
        
        # Test empty case
        empty_md = PDFLoader._rebuild_markdown([], [])
        assert empty_md == ""
    
    def test_reindex_rows_static_method(self):
        """Test static method _reindex_rows()."""
        # Create test rows with wrong indices
        cells1 = [TableCell(value="A", row=999, col=999, bbox=None, metadata={})]
        row1 = TableRow(cells=cells1, row_idx=999)
        
        cells2 = [TableCell(value="B", row=888, col=888, bbox=None, metadata={})]
        row2 = TableRow(cells=cells2, row_idx=888)
        
        rows = [row1, row2]
        PDFLoader._reindex_rows(rows)
        
        # Check reindexing
        assert rows[0].row_idx == 1
        assert rows[0].cells[0].row == 1
        assert rows[0].cells[0].col == 1
        
        assert rows[1].row_idx == 2
        assert rows[1].cells[0].row == 2
        assert rows[1].cells[0].col == 1
    
    # ========== INTEGRATION TESTS ==========
    
    @patch('loaders.pdf_loader.fitz')
    @patch('loaders.pdf_loader.pdfplumber')
    def test_file_operations_mocked(self, mock_pdfplumber, mock_fitz):
        """Test file operations với mock để tránh dependency on actual PDF files."""
        # Mock fitz document
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = MagicMock()
        mock_fitz.open.return_value = mock_doc
        
        # Mock pdfplumber
        mock_pdfplumber.open.return_value = MagicMock()
        
        # Mock PDFDocument static methods
        with patch('loaders.pdf_loader.PDFDocument.extract_metadata') as mock_metadata:
            mock_metadata.return_value = ("test.pdf", {}, {}, 1)
            
            with patch('loaders.pdf_loader.PDFDocument.collect_all_blocks') as mock_blocks:
                mock_blocks.return_value = [[]]  # Empty blocks
                
                loader = PDFLoader.create_default()
                
                # This should not raise an exception
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    
                try:
                    # Mock the actual load call
                    with patch.object(loader, '_open_documents') as mock_open:
                        mock_open.return_value = (mock_doc, None, [])
                        
                        result = loader.load_pdf(tmp_path)
                        
                        assert isinstance(result, PDFDocument)
                        assert result.file_path == tmp_path
                        assert result.num_pages == 1
                        
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    # ========== REPR AND STRING METHODS ==========
    
    def test_repr_method(self):
        """Test __repr__ method."""
        loader = PDFLoader(
            extract_text=False,
            extract_tables=True,
            tables_engine="camelot",
            min_repeated_text_threshold=5
        )
        
        repr_str = repr(loader)
        
        assert "PDFLoader(" in repr_str
        assert "extract_text=False" in repr_str
        assert "extract_tables=True" in repr_str
        assert "tables_engine='camelot'" in repr_str
        assert "min_repeated_text_threshold=5" in repr_str
    
    # ========== EDGE CASES ==========
    
    def test_update_config_unknown_parameter(self):
        """Test update_config với unknown parameter."""
        loader = PDFLoader()
        
        # Should not raise error, just ignore unknown parameter
        loader.update_config(unknown_param="value")
        
        # Should not have the unknown attribute
        assert not hasattr(loader, 'unknown_param')
    
    def test_config_consistency(self):
        """Test config consistency between different initialization methods."""
        loader1 = PDFLoader()
        loader2 = PDFLoader.create_default()
        
        # Both should have same default config
        config1 = loader1.get_config()
        config2 = loader2.get_config()
        
        assert config1 == config2
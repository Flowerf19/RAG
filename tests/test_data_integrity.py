"""
Test tính toàn vẹn dữ liệu cho PDFLoader với file PDF thật.
Load file PDF và xuất dữ liệu ra JSON để kiểm tra blocks và tables.
"""

import pytest
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from loaders.pdf_loader import PDFLoader
from loaders.model.document import PDFDocument
from loaders.model.page import PDFPage


class TestPDFLoaderDataIntegrity:
    """
    Test class để kiểm tra tính toàn vẹn dữ liệu của PDFLoader.
    Load file PDF thật và validate data integrity.
    """
    
    # Test file path
    TEST_PDF_PATH = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\data\pdf\Process_Risk Management.pdf"
    OUTPUT_DIR = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\test_outputs"
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup cho mỗi test method."""
        # Tạo output directory nếu chưa có
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Check if test file exists
        if not os.path.exists(self.TEST_PDF_PATH):
            pytest.skip(f"Test PDF file not found: {self.TEST_PDF_PATH}")
    
    def _save_json_output(self, data: Dict[str, Any], filename: str) -> str:
        """Save data to JSON file for inspection."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_{filename}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Data saved to: {output_file}")
        return output_file
    
    def _extract_document_summary(self, document: PDFDocument) -> Dict[str, Any]:
        """Extract summary statistics from document."""
        summary = {
            "file_path": document.file_path,
            "num_pages": document.num_pages,
            "total_pages_loaded": len(document.pages),
            "meta": document.meta,
            "warnings": document.warnings,
            "pages_summary": []
        }
        
        total_blocks = 0
        total_tables = 0
        
        for page in document.pages:
            page_blocks = len(page.blocks) if page.blocks else 0
            page_tables = len(page.tables) if page.tables else 0
            
            total_blocks += page_blocks
            total_tables += page_tables
            
            page_summary = {
                "page_number": page.page_number,
                "blocks_count": page_blocks,
                "tables_count": page_tables,
                "has_text": bool(page.text),
                "source": page.source
            }
            summary["pages_summary"].append(page_summary)
        
        summary["totals"] = {
            "total_blocks": total_blocks,
            "total_tables": total_tables
        }
        
        return summary
    
    def _extract_detailed_blocks(self, document: PDFDocument) -> List[Dict[str, Any]]:
        """Extract detailed block information."""
        all_blocks = []
        
        for page in document.pages:
            if not page.blocks:
                continue
                
            for i, block in enumerate(page.blocks):
                block_info = {
                    "page_number": page.page_number,
                    "block_index": i,
                    "block_type": type(block).__name__,
                    "block_data": None
                }
                
                # Handle different block types
                if hasattr(block, '__dict__'):
                    # If it's an object, extract its attributes
                    block_info["block_data"] = {
                        key: str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
                        for key, value in block.__dict__.items()
                    }
                elif isinstance(block, (tuple, list)):
                    # If it's a tuple/list (raw block format)
                    if len(block) >= 5:
                        block_info["block_data"] = {
                            "x0": block[0] if len(block) > 0 else None,
                            "y0": block[1] if len(block) > 1 else None,
                            "x1": block[2] if len(block) > 2 else None,
                            "y1": block[3] if len(block) > 3 else None,
                            "text": block[4] if len(block) > 4 else None,
                            "block_no": block[5] if len(block) > 5 else None,
                            "block_type": block[6] if len(block) > 6 else None,
                            "text_length": len(str(block[4])) if len(block) > 4 and block[4] else 0
                        }
                else:
                    block_info["block_data"] = str(block)
                
                all_blocks.append(block_info)
        
        return all_blocks
    
    def _extract_detailed_tables(self, document: PDFDocument) -> List[Dict[str, Any]]:
        """Extract detailed table information."""
        all_tables = []
        
        for page in document.pages:
            if not page.tables:
                continue
                
            for i, table in enumerate(page.tables):
                table_info = {
                    "page_number": page.page_number,
                    "table_index": i,
                    "table_type": type(table).__name__,
                    "table_data": None
                }
                
                if isinstance(table, dict):
                    # Dict format with matrix and bbox
                    matrix = table.get('matrix', [])
                    bbox = table.get('bbox')
                    metadata = table.get('metadata', {})
                    
                    table_info["table_data"] = {
                        "matrix_rows": len(matrix),
                        "matrix_cols": len(matrix[0]) if matrix and len(matrix) > 0 else 0,
                        "bbox": bbox,
                        "metadata": metadata,
                        "sample_matrix": matrix[:3] if matrix else [],  # First 3 rows for inspection
                        "has_caption": bool(metadata.get('table_caption')) if metadata else False,
                        "caption": metadata.get('table_caption') if metadata else None
                    }
                else:
                    # Other table formats
                    table_info["table_data"] = str(table)
                
                all_tables.append(table_info)
        
        return all_tables
    
    def test_default_loader_data_integrity(self):
        """Test data integrity với default loader configuration."""
        print(f"\n🔍 Testing DEFAULT loader with: {self.TEST_PDF_PATH}")
        
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        # Basic assertions
        assert isinstance(document, PDFDocument)
        assert document.file_path == self.TEST_PDF_PATH
        assert document.num_pages > 0
        assert len(document.pages) > 0
        
        # Extract and save detailed data
        summary = self._extract_document_summary(document)
        blocks = self._extract_detailed_blocks(document)
        tables = self._extract_detailed_tables(document)
        
        output_data = {
            "test_name": "default_loader",
            "loader_config": loader.get_config(),
            "summary": summary,
            "detailed_blocks": blocks,
            "detailed_tables": tables
        }
        
        self._save_json_output(output_data, "default_loader_data.json")
        
        # Print summary to console
        print(f"📊 SUMMARY - Default Loader:")
        print(f"   Pages: {summary['total_pages_loaded']}/{summary['num_pages']}")
        print(f"   Total Blocks: {summary['totals']['total_blocks']}")
        print(f"   Total Tables: {summary['totals']['total_tables']}")
        
        # Data integrity checks
        assert summary['totals']['total_blocks'] > 0, "Should have extracted some blocks"
        if loader.extract_tables:
            print(f"   Tables extraction enabled, found: {summary['totals']['total_tables']} tables")
    
    def test_text_only_loader_data_integrity(self):
        """Test data integrity với text-only loader."""
        print(f"\n🔍 Testing TEXT-ONLY loader with: {self.TEST_PDF_PATH}")
        
        loader = PDFLoader.create_text_only()
        document = loader.load(self.TEST_PDF_PATH)
        
        # Basic assertions
        assert isinstance(document, PDFDocument)
        assert document.num_pages > 0
        
        # Extract data
        summary = self._extract_document_summary(document)
        blocks = self._extract_detailed_blocks(document)
        tables = self._extract_detailed_tables(document)
        
        output_data = {
            "test_name": "text_only_loader",
            "loader_config": loader.get_config(),
            "summary": summary,
            "detailed_blocks": blocks,
            "detailed_tables": tables
        }
        
        self._save_json_output(output_data, "text_only_loader_data.json")
        
        print(f"📊 SUMMARY - Text-Only Loader:")
        print(f"   Pages: {summary['total_pages_loaded']}/{summary['num_pages']}")
        print(f"   Total Blocks: {summary['totals']['total_blocks']}")
        print(f"   Total Tables: {summary['totals']['total_tables']}")
        
        # Text-only should have no tables
        assert summary['totals']['total_tables'] == 0, "Text-only loader should not extract tables"
        assert summary['totals']['total_blocks'] > 0, "Should have extracted text blocks"
    
    def test_tables_only_loader_data_integrity(self):
        """Test data integrity với tables-only loader."""
        print(f"\n🔍 Testing TABLES-ONLY loader with: {self.TEST_PDF_PATH}")
        
        loader = PDFLoader.create_tables_only()
        document = loader.load(self.TEST_PDF_PATH)
        
        # Basic assertions
        assert isinstance(document, PDFDocument)
        assert document.num_pages > 0
        
        # Extract data
        summary = self._extract_document_summary(document)
        blocks = self._extract_detailed_blocks(document)
        tables = self._extract_detailed_tables(document)
        
        output_data = {
            "test_name": "tables_only_loader",
            "loader_config": loader.get_config(),
            "summary": summary,
            "detailed_blocks": blocks,
            "detailed_tables": tables
        }
        
        self._save_json_output(output_data, "tables_only_loader_data.json")
        
        print(f"📊 SUMMARY - Tables-Only Loader:")
        print(f"   Pages: {summary['total_pages_loaded']}/{summary['num_pages']}")
        print(f"   Total Blocks: {summary['totals']['total_blocks']}")
        print(f"   Total Tables: {summary['totals']['total_tables']}")
        
        # Tables-only should focus on tables
        print(f"   Tables extraction: {'enabled' if loader.extract_tables else 'disabled'}")
        print(f"   Text extraction: {'enabled' if loader.extract_text else 'disabled'}")
    
    def test_custom_config_data_integrity(self):
        """Test data integrity với custom configuration."""
        print(f"\n🔍 Testing CUSTOM CONFIG loader with: {self.TEST_PDF_PATH}")
        
        loader = PDFLoader(
            extract_text=True,
            extract_tables=True,
            tables_engine="pdfplumber",
            min_repeated_text_threshold=2,
            min_text_length=5
        )
        document = loader.load(self.TEST_PDF_PATH)
        
        # Extract data
        summary = self._extract_document_summary(document)
        blocks = self._extract_detailed_blocks(document)
        tables = self._extract_detailed_tables(document)
        
        output_data = {
            "test_name": "custom_config_loader",
            "loader_config": loader.get_config(),
            "summary": summary,
            "detailed_blocks": blocks,
            "detailed_tables": tables
        }
        
        self._save_json_output(output_data, "custom_config_loader_data.json")
        
        print(f"📊 SUMMARY - Custom Config Loader:")
        print(f"   Pages: {summary['total_pages_loaded']}/{summary['num_pages']}")
        print(f"   Total Blocks: {summary['totals']['total_blocks']}")
        print(f"   Total Tables: {summary['totals']['total_tables']}")
        print(f"   Config: {loader.get_config()}")
    
    def test_compare_loader_configurations(self):
        """So sánh kết quả từ các configuration khác nhau."""
        print(f"\n🔍 COMPARISON TEST - Multiple configurations")
        
        configs = [
            ("default", PDFLoader.create_default()),
            ("text_only", PDFLoader.create_text_only()),
            ("tables_only", PDFLoader.create_tables_only()),
        ]
        
        comparison_data = {
            "file_path": self.TEST_PDF_PATH,
            "test_timestamp": datetime.now().isoformat(),
            "configurations": {}
        }
        
        for config_name, loader in configs:
            print(f"   Loading with {config_name} config...")
            document = loader.load(self.TEST_PDF_PATH)
            summary = self._extract_document_summary(document)
            
            comparison_data["configurations"][config_name] = {
                "loader_config": loader.get_config(),
                "summary": summary
            }
            
            print(f"      {config_name}: {summary['totals']['total_blocks']} blocks, {summary['totals']['total_tables']} tables")
        
        self._save_json_output(comparison_data, "loader_comparison.json")
        
        # Validate consistency
        page_counts = [data["summary"]["total_pages_loaded"] for data in comparison_data["configurations"].values()]
        assert all(count == page_counts[0] for count in page_counts), "All loaders should load same number of pages"
        
        print(f"✅ Comparison complete - all loaders consistent on page count: {page_counts[0]}")
    
    def test_block_content_analysis(self):
        """Phân tích chi tiết nội dung blocks."""
        print(f"\n🔍 BLOCK CONTENT ANALYSIS")
        
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        block_analysis = {
            "file_path": self.TEST_PDF_PATH,
            "analysis": {
                "total_blocks": 0,
                "blocks_by_page": {},
                "text_blocks": [],
                "empty_blocks": [],
                "large_blocks": [],
                "block_types": {}
            }
        }
        
        for page in document.pages:
            page_num = page.page_number
            block_analysis["analysis"]["blocks_by_page"][page_num] = {
                "count": len(page.blocks) if page.blocks else 0,
                "blocks": []
            }
            
            if not page.blocks:
                continue
            
            for i, block in enumerate(page.blocks):
                block_analysis["analysis"]["total_blocks"] += 1
                
                # Analyze block based on its type
                if isinstance(block, (tuple, list)) and len(block) >= 5:
                    text = block[4] if len(block) > 4 else ""
                    text_len = len(str(text)) if text else 0
                    
                    block_info = {
                        "page": page_num,
                        "index": i,
                        "bbox": [block[0], block[1], block[2], block[3]] if len(block) >= 4 else None,
                        "text_length": text_len,
                        "text_preview": str(text)[:100] + "..." if text_len > 100 else str(text),
                        "block_type": block[6] if len(block) > 6 else "unknown"
                    }
                    
                    # Categorize blocks
                    if text_len == 0:
                        block_analysis["analysis"]["empty_blocks"].append(block_info)
                    elif text_len > 500:
                        block_analysis["analysis"]["large_blocks"].append(block_info)
                    else:
                        block_analysis["analysis"]["text_blocks"].append(block_info)
                    
                    # Track block types
                    block_type = block_info["block_type"]
                    if block_type not in block_analysis["analysis"]["block_types"]:
                        block_analysis["analysis"]["block_types"][block_type] = 0
                    block_analysis["analysis"]["block_types"][block_type] += 1
                    
                    block_analysis["analysis"]["blocks_by_page"][page_num]["blocks"].append(block_info)
        
        self._save_json_output(block_analysis, "block_content_analysis.json")
        
        # Print analysis summary
        print(f"📊 BLOCK ANALYSIS SUMMARY:")
        print(f"   Total blocks: {block_analysis['analysis']['total_blocks']}")
        print(f"   Text blocks: {len(block_analysis['analysis']['text_blocks'])}")
        print(f"   Empty blocks: {len(block_analysis['analysis']['empty_blocks'])}")
        print(f"   Large blocks (>500 chars): {len(block_analysis['analysis']['large_blocks'])}")
        print(f"   Block types: {block_analysis['analysis']['block_types']}")
        
        # Assertions
        assert block_analysis['analysis']['total_blocks'] > 0, "Should have found some blocks"
    
    def test_table_content_analysis(self):
        """Phân tích chi tiết nội dung tables."""
        print(f"\n🔍 TABLE CONTENT ANALYSIS")
        
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        table_analysis = {
            "file_path": self.TEST_PDF_PATH,
            "analysis": {
                "total_tables": 0,
                "tables_by_page": {},
                "table_sizes": [],
                "tables_with_captions": [],
                "sample_table_data": []
            }
        }
        
        for page in document.pages:
            page_num = page.page_number
            table_analysis["analysis"]["tables_by_page"][page_num] = {
                "count": len(page.tables) if page.tables else 0,
                "tables": []
            }
            
            if not page.tables:
                continue
            
            for i, table in enumerate(page.tables):
                table_analysis["analysis"]["total_tables"] += 1
                
                if isinstance(table, dict):
                    matrix = table.get('matrix', [])
                    bbox = table.get('bbox')
                    metadata = table.get('metadata', {})
                    
                    rows = len(matrix)
                    cols = len(matrix[0]) if matrix and len(matrix) > 0 else 0
                    
                    table_info = {
                        "page": page_num,
                        "index": i,
                        "rows": rows,
                        "cols": cols,
                        "bbox": bbox,
                        "has_caption": bool(metadata.get('table_caption')),
                        "caption": metadata.get('table_caption', ""),
                        "sample_data": matrix[:2] if matrix else []  # First 2 rows
                    }
                    
                    table_analysis["analysis"]["table_sizes"].append({"rows": rows, "cols": cols})
                    
                    if table_info["has_caption"]:
                        table_analysis["analysis"]["tables_with_captions"].append(table_info)
                    
                    if len(table_analysis["analysis"]["sample_table_data"]) < 3:  # Keep first 3 tables
                        table_analysis["analysis"]["sample_table_data"].append(table_info)
                    
                    table_analysis["analysis"]["tables_by_page"][page_num]["tables"].append(table_info)
        
        self._save_json_output(table_analysis, "table_content_analysis.json")
        
        # Print analysis summary
        print(f"📊 TABLE ANALYSIS SUMMARY:")
        print(f"   Total tables: {table_analysis['analysis']['total_tables']}")
        print(f"   Tables with captions: {len(table_analysis['analysis']['tables_with_captions'])}")
        
        if table_analysis['analysis']['table_sizes']:
            avg_rows = sum(t['rows'] for t in table_analysis['analysis']['table_sizes']) / len(table_analysis['analysis']['table_sizes'])
            avg_cols = sum(t['cols'] for t in table_analysis['analysis']['table_sizes']) / len(table_analysis['analysis']['table_sizes'])
            print(f"   Average table size: {avg_rows:.1f} rows × {avg_cols:.1f} cols")
        
        # Show sample table data
        for i, table in enumerate(table_analysis['analysis']['sample_table_data']):
            print(f"   Sample Table {i+1} (Page {table['page']}): {table['rows']}×{table['cols']}")
            if table['has_caption']:
                print(f"      Caption: {table['caption']}")
            if table['sample_data']:
                print(f"      Sample data: {table['sample_data'][0][:3] if table['sample_data'][0] else 'Empty'}")
    
    def test_export_readable_text_data(self):
        """Export dữ liệu ra file TXT dễ đọc."""
        print(f"\n🔍 EXPORTING READABLE TEXT DATA")
        
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_readable_data.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("PDF DATA INTEGRITY REPORT - READABLE FORMAT\n")
            f.write("=" * 80 + "\n")
            f.write(f"File: {self.TEST_PDF_PATH}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Pages: {document.num_pages}\n")
            f.write(f"Title: {document.meta.get('/Title', 'N/A')}\n")
            f.write(f"Author: {document.meta.get('/Author', 'N/A')}\n")
            f.write("\n")
            
            # Summary
            total_blocks = sum(len(page.blocks) if page.blocks else 0 for page in document.pages)
            total_tables = sum(len(page.tables) if page.tables else 0 for page in document.pages)
            
            f.write("-" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Pages:  {len(document.pages)}\n")
            f.write(f"Total Blocks: {total_blocks}\n")
            f.write(f"Total Tables: {total_tables}\n")
            f.write("\n")
            
            # Page by page analysis
            f.write("-" * 80 + "\n")
            f.write("PAGE-BY-PAGE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            
            for page in document.pages:
                f.write(f"\n=== PAGE {page.page_number} ===\n")
                f.write(f"Blocks: {len(page.blocks) if page.blocks else 0}\n")
                f.write(f"Tables: {len(page.tables) if page.tables else 0}\n")
                f.write(f"Page size: {page.source.get('page_size', {}).get('width', 'N/A')} x {page.source.get('page_size', {}).get('height', 'N/A')}\n")
                f.write("\n")
                
                # Show first 5 blocks
                if page.blocks:
                    f.write("BLOCKS (first 5):\n")
                    for i, block in enumerate(page.blocks[:5]):
                        if isinstance(block, (tuple, list)) and len(block) >= 5:
                            text = str(block[4])[:100] + "..." if len(str(block[4])) > 100 else str(block[4])
                            f.write(f"  Block {i+1}: {text.strip()}\n")
                            f.write(f"    Position: ({block[0]:.1f}, {block[1]:.1f}, {block[2]:.1f}, {block[3]:.1f})\n")
                            f.write(f"    Length: {len(str(block[4])) if len(block) > 4 else 0} chars\n")
                        f.write("\n")
                
                # Show tables
                if page.tables:
                    f.write("TABLES:\n")
                    for i, table in enumerate(page.tables):
                        if isinstance(table, dict):
                            matrix = table.get('matrix', [])
                            metadata = table.get('metadata', {})
                            caption = metadata.get('table_caption', 'No caption')
                            
                            f.write(f"  Table {i+1}: {caption}\n")
                            f.write(f"    Size: {len(matrix)} rows x {len(matrix[0]) if matrix and len(matrix) > 0 else 0} cols\n")
                            f.write(f"    Bbox: {table.get('bbox')}\n")
                            
                            # Show table content if available
                            if matrix and len(matrix) > 0:
                                f.write("    Content (first 3 rows):\n")
                                for row_idx, row in enumerate(matrix[:3]):
                                    if row:
                                        f.write(f"      Row {row_idx+1}: {row[:3]}\n")  # First 3 columns
                                    else:
                                        f.write(f"      Row {row_idx+1}: [Empty]\n")
                            else:
                                f.write("    Content: [Empty matrix]\n")
                        f.write("\n")
            
            # Block statistics
            f.write("-" * 80 + "\n")
            f.write("BLOCK STATISTICS\n")
            f.write("-" * 80 + "\n")
            
            all_block_lengths = []
            block_types = {}
            
            for page in document.pages:
                if not page.blocks:
                    continue
                for block in page.blocks:
                    if isinstance(block, (tuple, list)) and len(block) >= 5:
                        text_len = len(str(block[4])) if len(block) > 4 else 0
                        all_block_lengths.append(text_len)
                        
                        block_type = block[6] if len(block) > 6 else "unknown"
                        block_types[block_type] = block_types.get(block_type, 0) + 1
            
            if all_block_lengths:
                f.write(f"Average block length: {sum(all_block_lengths) / len(all_block_lengths):.1f} chars\n")
                f.write(f"Min block length: {min(all_block_lengths)} chars\n")
                f.write(f"Max block length: {max(all_block_lengths)} chars\n")
                f.write(f"Blocks with 0 chars: {all_block_lengths.count(0)}\n")
                f.write(f"Blocks with >100 chars: {sum(1 for l in all_block_lengths if l > 100)}\n")
            
            f.write(f"\nBlock types: {block_types}\n")
            
            # Table statistics
            f.write("\n" + "-" * 80 + "\n")
            f.write("TABLE STATISTICS\n")
            f.write("-" * 80 + "\n")
            
            all_captions = []
            table_sizes = []
            
            for page in document.pages:
                if not page.tables:
                    continue
                for table in page.tables:
                    if isinstance(table, dict):
                        matrix = table.get('matrix', [])
                        metadata = table.get('metadata', {})
                        caption = metadata.get('table_caption', '')
                        
                        if caption:
                            all_captions.append(caption)
                        
                        rows = len(matrix)
                        cols = len(matrix[0]) if matrix and len(matrix) > 0 else 0
                        table_sizes.append((rows, cols))
            
            f.write(f"Tables with captions: {len(all_captions)} / {total_tables}\n")
            f.write("\nAll table captions:\n")
            for i, caption in enumerate(all_captions, 1):
                f.write(f"  {i}. {caption}\n")
            
            if table_sizes:
                avg_rows = sum(s[0] for s in table_sizes) / len(table_sizes)
                avg_cols = sum(s[1] for s in table_sizes) / len(table_sizes)
                f.write(f"\nAverage table size: {avg_rows:.1f} rows x {avg_cols:.1f} cols\n")
                f.write(f"Table sizes: {table_sizes}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"📄 Readable data exported to: {output_file}")
        
        # Also create a simple blocks-only text file
        blocks_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_blocks_text.txt")
        with open(blocks_file, 'w', encoding='utf-8') as f:
            f.write("EXTRACTED TEXT BLOCKS\n")
            f.write("=" * 50 + "\n\n")
            
            for page in document.pages:
                f.write(f"=== PAGE {page.page_number} ===\n\n")
                
                if page.blocks:
                    for i, block in enumerate(page.blocks):
                        if isinstance(block, (tuple, list)) and len(block) >= 5:
                            text = str(block[4]).strip()
                            if text:  # Only write non-empty blocks
                                f.write(f"Block {i+1}:\n{text}\n\n")
                
                f.write("-" * 30 + "\n\n")
        
        print(f"📄 Block text exported to: {blocks_file}")
        
        return output_file, blocks_file
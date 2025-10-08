"""
Test để in dữ liệu blocks đã được merge ra file để xem xét.
"""

import pytest
import os
from datetime import datetime

from loaders.pdf_loader import PDFLoader


class TestPrintMergedData:
    """
    Test class để in dữ liệu blocks đã được merge.
    """
    
    TEST_PDF_PATH = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\data\pdf\Process_Risk Management.pdf"
    OUTPUT_DIR = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\test_outputs"
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup cho mỗi test method."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(self.TEST_PDF_PATH):
            pytest.skip(f"Test PDF file not found: {self.TEST_PDF_PATH}")
    
    def test_export_merged_blocks_readable(self):
        """In dữ liệu blocks đã merge ra file TXT để xem."""
        print(f"\n📄 EXPORTING MERGED BLOCKS DATA")
        
        # Load với block merging enabled
        loader = PDFLoader.create_default()  # Merging enabled by default
        document = loader.load(self.TEST_PDF_PATH)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_merged_blocks_readable.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("MERGED BLOCKS DATA - READABLE FORMAT\n")
            f.write("=" * 60 + "\n\n")
            
            total_blocks = 0
            for page_num, page in enumerate(document.pages, 1):
                f.write(f"=== PAGE {page_num} ===\n")
                f.write(f"Total blocks on this page: {len(page.blocks)}\n\n")
                
                for block_idx, block in enumerate(page.blocks):
                    if isinstance(block, (tuple, list)) and len(block) >= 5:
                        text = str(block[4]).strip()
                        
                        # Tính metadata
                        word_count = len(text.split()) if text else 0
                        char_count = len(text)
                        
                        f.write(f"Block {block_idx + 1}:\n")
                        f.write(f"  Length: {char_count} chars, {word_count} words\n")
                        f.write(f"  Content: {text}\n")
                        f.write("-" * 50 + "\n")
                        
                        total_blocks += 1
                
                f.write(f"\n")
            
            f.write(f"\nSUMMARY:\n")
            f.write(f"Total pages: {len(document.pages)}\n")
            f.write(f"Total blocks: {total_blocks}\n")
            f.write(f"Average blocks per page: {total_blocks/len(document.pages):.1f}\n")
        
        print(f"📄 Merged blocks data exported to: {output_file}")
        print(f"   Total blocks: {total_blocks}")
        print(f"   Total pages: {len(document.pages)}")
        
        return output_file
    
    def test_export_comparison_side_by_side(self):
        """In so sánh trước/sau merge side-by-side."""
        print(f"\n📊 EXPORTING SIDE-BY-SIDE COMPARISON")
        
        # Load without merging
        loader_no_merge = PDFLoader(
            extract_text=True,
            extract_tables=True,
            enable_block_merging=False
        )
        doc_no_merge = loader_no_merge.load(self.TEST_PDF_PATH)
        
        # Load with merging
        loader_with_merge = PDFLoader.create_default()
        doc_with_merge = loader_with_merge.load(self.TEST_PDF_PATH)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_side_by_side_comparison.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SIDE-BY-SIDE COMPARISON: BEFORE vs AFTER MERGING\n")
            f.write("=" * 80 + "\n\n")
            
            # Compare specific pages that have problematic sections
            target_pages = [15, 16]  # Pages with risk approaches section
            
            for page_num in target_pages:
                if page_num <= len(doc_no_merge.pages) and page_num <= len(doc_with_merge.pages):
                    page_no_merge = doc_no_merge.pages[page_num - 1]
                    page_with_merge = doc_with_merge.pages[page_num - 1]
                    
                    f.write(f"=== PAGE {page_num} ===\n")
                    f.write(f"BEFORE (No Merge): {len(page_no_merge.blocks)} blocks\n")
                    f.write(f"AFTER (With Merge): {len(page_with_merge.blocks)} blocks\n")
                    f.write(f"Improvement: {len(page_no_merge.blocks) - len(page_with_merge.blocks)} blocks reduced\n\n")
                    
                    # Show first 10 blocks from each
                    f.write("BEFORE MERGING:\n")
                    f.write("-" * 40 + "\n")
                    for i, block in enumerate(page_no_merge.blocks[:10]):
                        if isinstance(block, (tuple, list)) and len(block) >= 5:
                            text = str(block[4]).strip()[:100]  # First 100 chars
                            f.write(f"Block {i+1}: {text}...\n")
                    
                    f.write("\nAFTER MERGING:\n")
                    f.write("-" * 40 + "\n")
                    for i, block in enumerate(page_with_merge.blocks[:6]):  # Fewer blocks expected
                        if isinstance(block, (tuple, list)) and len(block) >= 5:
                            text = str(block[4]).strip()[:200]  # More chars since merged
                            f.write(f"Block {i+1}: {text}...\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"📊 Side-by-side comparison exported to: {output_file}")
        
        return output_file
    
    def test_export_specific_examples(self):
        """In các ví dụ cụ thể về cải thiện chunking."""
        print(f"\n🎯 EXPORTING SPECIFIC IMPROVEMENT EXAMPLES")
        
        # Load both versions
        loader_no_merge = PDFLoader(enable_block_merging=False, extract_tables=False)
        doc_no_merge = loader_no_merge.load(self.TEST_PDF_PATH)
        
        loader_with_merge = PDFLoader(enable_block_merging=True, extract_tables=False)
        doc_with_merge = loader_with_merge.load(self.TEST_PDF_PATH)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_specific_examples.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SPECIFIC CHUNKING IMPROVEMENT EXAMPLES\n")
            f.write("=" * 60 + "\n\n")
            
            # Example 1: Risk approaches section
            f.write("EXAMPLE 1: RISK APPROACHES SECTION\n")
            f.write("-" * 40 + "\n")
            
            # Find risk approaches in both versions
            risk_text_original = []
            risk_text_merged = []
            
            # Search original
            for page in doc_no_merge.pages:
                for block in page.blocks:
                    if isinstance(block, (tuple, list)) and len(block) >= 5:
                        text = str(block[4])
                        if "different approaches to risk management" in text.lower():
                            # Get next 8 blocks
                            page_blocks = page.blocks
                            start_idx = page_blocks.index(block)
                            risk_blocks = page_blocks[start_idx:start_idx + 8]
                            risk_text_original = [str(b[4]) for b in risk_blocks if isinstance(b, (tuple, list)) and len(b) >= 5]
                            break
                if risk_text_original:
                    break
            
            # Search merged
            for page in doc_with_merge.pages:
                for block in page.blocks:
                    if isinstance(block, (tuple, list)) and len(block) >= 5:
                        text = str(block[4])
                        if "different approaches to risk management" in text.lower():
                            # Get next 4 blocks (should be fewer)
                            page_blocks = page.blocks
                            start_idx = page_blocks.index(block)
                            risk_blocks = page_blocks[start_idx:start_idx + 4]
                            risk_text_merged = [str(b[4]) for b in risk_blocks if isinstance(b, (tuple, list)) and len(b) >= 5]
                            break
                if risk_text_merged:
                    break
            
            f.write("BEFORE (Fragmented):\n")
            for i, text in enumerate(risk_text_original):
                f.write(f"Block {i+1}: {text.strip()}\n\n")
            
            f.write("AFTER (Merged):\n")
            for i, text in enumerate(risk_text_merged):
                f.write(f"Block {i+1}: {text.strip()}\n\n")
            
            # Example 2: List items
            f.write("EXAMPLE 2: LIST ITEMS IMPROVEMENT\n")
            f.write("-" * 40 + "\n")
            
            # Find list items
            list_examples_original = []
            list_examples_merged = []
            
            for page in doc_no_merge.pages:
                for block in page.blocks:
                    if isinstance(block, (tuple, list)) and len(block) >= 5:
                        text = str(block[4]).strip()
                        if text.startswith("• Asset based:"):
                            # Get this and next block
                            page_blocks = page.blocks
                            start_idx = page_blocks.index(block)
                            if start_idx + 1 < len(page_blocks):
                                list_examples_original = [
                                    str(page_blocks[start_idx][4]),
                                    str(page_blocks[start_idx + 1][4])
                                ]
                            break
                if list_examples_original:
                    break
            
            for page in doc_with_merge.pages:
                for block in page.blocks:
                    if isinstance(block, (tuple, list)) and len(block) >= 5:
                        text = str(block[4]).strip()
                        if "Asset based:" in text and "relevant for operations" in text:
                            list_examples_merged = [str(block[4])]
                            break
                if list_examples_merged:
                    break
            
            f.write("BEFORE (List item fragmented):\n")
            for i, text in enumerate(list_examples_original):
                f.write(f"Block {i+1}: {text.strip()}\n\n")
            
            f.write("AFTER (List item consolidated):\n")
            for i, text in enumerate(list_examples_merged):
                f.write(f"Block {i+1}: {text.strip()}\n\n")
            
            # Stats
            total_original = sum(len(page.blocks) for page in doc_no_merge.pages)
            total_merged = sum(len(page.blocks) for page in doc_with_merge.pages)
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Original blocks: {total_original}\n")
            f.write(f"Merged blocks: {total_merged}\n")
            f.write(f"Reduction: {total_original - total_merged} ({(total_original - total_merged)/total_original*100:.1f}%)\n")
        
        print(f"🎯 Specific examples exported to: {output_file}")
        
        return output_file
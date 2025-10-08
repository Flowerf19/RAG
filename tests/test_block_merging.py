"""
Test tính năng Block Merging mới trong PDFLoader
"""

import pytest
import os
from datetime import datetime

from loaders.pdf_loader import PDFLoader
from loaders.normalizers.block_utils import analyze_block_improvement


class TestBlockMerging:
    """
    Test class để kiểm tra tính năng merge blocks trong PDFLoader.
    """
    
    TEST_PDF_PATH = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\data\pdf\Process_Risk Management.pdf"
    OUTPUT_DIR = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\test_outputs"
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup cho mỗi test method."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(self.TEST_PDF_PATH):
            pytest.skip(f"Test PDF file not found: {self.TEST_PDF_PATH}")
    
    def test_block_merging_enabled_vs_disabled(self):
        """Test so sánh kết quả với và không có block merging."""
        print(f"\n🔧 TESTING BLOCK MERGING: ENABLED vs DISABLED")
        
        # Test without block merging
        loader_no_merge = PDFLoader(
            extract_text=True,
            extract_tables=True,
            enable_block_merging=False  # DISABLED
        )
        doc_no_merge = loader_no_merge.load(self.TEST_PDF_PATH)
        
        # Test with block merging
        loader_with_merge = PDFLoader(
            extract_text=True,
            extract_tables=True,
            enable_block_merging=True,  # ENABLED
            min_block_length=50
        )
        doc_with_merge = loader_with_merge.load(self.TEST_PDF_PATH)
        
        # Collect all blocks
        blocks_no_merge = []
        blocks_with_merge = []
        
        for page in doc_no_merge.pages:
            blocks_no_merge.extend(page.blocks)
            
        for page in doc_with_merge.pages:
            blocks_with_merge.extend(page.blocks)
        
        # Analyze improvement
        improvement = analyze_block_improvement(
            blocks_no_merge, 
            blocks_with_merge,
            config={'min_block_length': 50, 'sentence_endings': ('.', '!', '?', ':', ';')}
        )
        
        print(f"📊 MERGING RESULTS:")
        print(f"   Original blocks: {improvement['original_blocks']}")
        print(f"   Merged blocks: {improvement['merged_blocks']}")
        print(f"   Blocks reduced: {improvement['blocks_reduced']} ({improvement['reduction_percentage']:.1f}%)")
        print(f"   Short blocks: {improvement['original_short_blocks']} → {improvement['merged_short_blocks']} (improved: {improvement['short_blocks_improved']})")
        print(f"   Incomplete sentences: {improvement['original_incomplete']} → {improvement['merged_incomplete']} (improved: {improvement['incomplete_improved']})")
        
        # Save detailed comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_block_merging_comparison.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("BLOCK MERGING COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Original blocks (no merge): {improvement['original_blocks']}\n")
            f.write(f"Merged blocks: {improvement['merged_blocks']}\n")
            f.write(f"Reduction: {improvement['blocks_reduced']} blocks ({improvement['reduction_percentage']:.1f}%)\n")
            f.write(f"Short blocks improved: {improvement['short_blocks_improved']}\n")
            f.write(f"Incomplete sentences improved: {improvement['incomplete_improved']}\n\n")
            
            # Show example of problem area (risk approaches section)
            f.write("EXAMPLE: RISK APPROACHES SECTION IMPROVEMENT\n")
            f.write("-" * 45 + "\n")
            
            # Find risk approaches section in both versions
            risk_blocks_original = []
            risk_blocks_merged = []
            
            # Search in original
            for i, block in enumerate(blocks_no_merge):
                if isinstance(block, (tuple, list)) and len(block) >= 5:
                    text = str(block[4])
                    if "different approaches to risk management" in text.lower():
                        # Get surrounding blocks
                        start_idx = max(0, i-1)
                        end_idx = min(len(blocks_no_merge), i+10)
                        risk_blocks_original = blocks_no_merge[start_idx:end_idx]
                        break
            
            # Search in merged
            for i, block in enumerate(blocks_with_merge):
                if isinstance(block, (tuple, list)) and len(block) >= 5:
                    text = str(block[4])
                    if "different approaches to risk management" in text.lower():
                        # Get surrounding blocks
                        start_idx = max(0, i-1)
                        end_idx = min(len(blocks_with_merge), i+6)  # Should be fewer blocks
                        risk_blocks_merged = blocks_with_merge[start_idx:end_idx]
                        break
            
            f.write("ORIGINAL (NO MERGE):\n")
            for i, block in enumerate(risk_blocks_original[:8]):  # Show first 8
                if isinstance(block, (tuple, list)) and len(block) >= 5:
                    text = str(block[4]).strip()[:100]  # First 100 chars
                    f.write(f"  Block {i}: {text}...\n")
            
            f.write(f"\nMERGED VERSION:\n")
            for i, block in enumerate(risk_blocks_merged[:4]):  # Show first 4
                if isinstance(block, (tuple, list)) and len(block) >= 5:
                    text = str(block[4]).strip()[:200]  # First 200 chars
                    f.write(f"  Block {i}: {text}...\n")
            
            f.write(f"\nIMPROVEMENT IN THIS SECTION:\n")
            f.write(f"  Original blocks: {len(risk_blocks_original)}\n")
            f.write(f"  Merged blocks: {len(risk_blocks_merged)}\n")
            f.write(f"  Reduction: {len(risk_blocks_original) - len(risk_blocks_merged)} blocks\n")
        
        print(f"📄 Detailed comparison saved to: {report_file}")
        
        # Assertions
        assert improvement['merged_blocks'] < improvement['original_blocks'], "Should reduce number of blocks"
        assert improvement['short_blocks_improved'] > 0, "Should improve short blocks"
        assert improvement['blocks_reduced'] > 50, "Should reduce significant number of blocks"
        
        return improvement
    
    def test_factory_methods_with_merging(self):
        """Test factory methods có enable block merging đúng cách."""
        print(f"\n🏭 TESTING FACTORY METHODS WITH BLOCK MERGING")
        
        # Test default factory
        loader_default = PDFLoader.create_default()
        assert loader_default.enable_block_merging == True, "Default should have merging enabled"
        assert loader_default.min_block_length == 50, "Default min block length should be 50"
        
        # Test text only factory
        loader_text = PDFLoader.create_text_only()
        assert loader_text.enable_block_merging == True, "Text only should have merging enabled"
        assert loader_text.extract_tables == False, "Text only should not extract tables"
        
        # Test tables only factory
        loader_tables = PDFLoader.create_tables_only()
        assert loader_tables.enable_block_merging == False, "Tables only should not have merging enabled"
        assert loader_tables.extract_text == False, "Tables only should not extract text"
        
        print(f"✅ All factory methods configured correctly")
    
    def test_merging_configuration_options(self):
        """Test các options khác nhau cho block merging."""
        print(f"\n⚙️ TESTING MERGING CONFIGURATION OPTIONS")
        
        # Test conservative merging (higher threshold)
        loader_conservative = PDFLoader(
            extract_text=True,
            extract_tables=False,
            enable_block_merging=True,
            min_block_length=80  # Higher threshold = less aggressive merging
        )
        
        # Test aggressive merging (lower threshold) 
        loader_aggressive = PDFLoader(
            extract_text=True,
            extract_tables=False,
            enable_block_merging=True,
            min_block_length=30  # Lower threshold = more aggressive merging
        )
        
        # Load with both configs
        doc_conservative = loader_conservative.load(self.TEST_PDF_PATH)
        doc_aggressive = loader_aggressive.load(self.TEST_PDF_PATH)
        
        # Count blocks
        blocks_conservative = sum(len(page.blocks) for page in doc_conservative.pages)
        blocks_aggressive = sum(len(page.blocks) for page in doc_aggressive.pages)
        
        print(f"📊 MERGING AGGRESSIVENESS:")
        print(f"   Conservative (min_length=80): {blocks_conservative} blocks")
        print(f"   Aggressive (min_length=30): {blocks_aggressive} blocks")
        
        # Aggressive should result in fewer blocks
        assert blocks_aggressive <= blocks_conservative, "Aggressive merging should result in fewer or equal blocks"
        
        return blocks_conservative, blocks_aggressive
    
    def test_merging_preserves_text_content(self):
        """Test rằng merging không làm mất nội dung text."""
        print(f"\n📝 TESTING TEXT CONTENT PRESERVATION")
        
        # Load without merging
        loader_no_merge = PDFLoader(enable_block_merging=False, extract_tables=False)
        doc_no_merge = loader_no_merge.load(self.TEST_PDF_PATH)
        
        # Load with merging
        loader_with_merge = PDFLoader(enable_block_merging=True, extract_tables=False)
        doc_with_merge = loader_with_merge.load(self.TEST_PDF_PATH)
        
        # Extract all text content
        text_no_merge = ""
        text_with_merge = ""
        
        for page in doc_no_merge.pages:
            for block in page.blocks:
                if isinstance(block, (tuple, list)) and len(block) >= 5:
                    text_no_merge += str(block[4]) + " "
        
        for page in doc_with_merge.pages:
            for block in page.blocks:
                if isinstance(block, (tuple, list)) and len(block) >= 5:
                    text_with_merge += str(block[4]) + " "
        
        # Normalize for comparison (remove extra spaces)
        import re
        text_no_merge = re.sub(r'\s+', ' ', text_no_merge.strip())
        text_with_merge = re.sub(r'\s+', ' ', text_with_merge.strip())
        
        # Calculate similarity
        text_similarity = len(set(text_no_merge.split()) & set(text_with_merge.split())) / len(set(text_no_merge.split()) | set(text_with_merge.split()))
        
        print(f"📊 TEXT PRESERVATION:")
        print(f"   Original text length: {len(text_no_merge)} chars")
        print(f"   Merged text length: {len(text_with_merge)} chars")
        print(f"   Word similarity: {text_similarity:.3f} (should be >0.95)")
        
        # Text should be very similar (>95% word overlap)
        assert text_similarity > 0.95, f"Text similarity too low: {text_similarity:.3f}"
        
        return text_similarity
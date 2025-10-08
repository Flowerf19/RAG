"""
Test BlockAwareChunker với merged blocks và source tracking.
"""

import pytest
import os
from datetime import datetime

from loaders.pdf_loader import PDFLoader
from chunkers.block_aware_chunker import BlockAwareChunker


class TestBlockAwareChunker:
    """
    Test class cho BlockAwareChunker với source tracking.
    """
    
    TEST_PDF_PATH = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\data\pdf\Process_Risk Management.pdf"
    OUTPUT_DIR = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\test_outputs"
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup cho mỗi test method."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(self.TEST_PDF_PATH):
            pytest.skip(f"Test PDF file not found: {self.TEST_PDF_PATH}")
    
    def test_chunk_with_source_tracking(self):
        """Test chunking với source tracking từ merged blocks."""
        print(f"\n🧩 TESTING BLOCK-AWARE CHUNKING WITH SOURCE TRACKING")
        
        # Load document với merged blocks
        loader = PDFLoader.create_default()  # Block merging enabled
        document = loader.load(self.TEST_PDF_PATH)
        
        # Create chunker
        chunker = BlockAwareChunker.create_from_merged_loader(chunk_size=800)
        
        # Chunk document
        chunks = chunker.chunk_document(document)
        
        print(f"📊 CHUNKING RESULTS:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Average chunk size: {sum(c.char_count for c in chunks) / len(chunks):.0f} chars")
        print(f"   Average words per chunk: {sum(c.word_count for c in chunks) / len(chunks):.0f} words")
        
        # Export detailed chunk info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_chunks_with_sources.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("BLOCK-AWARE CHUNKS WITH SOURCE TRACKING\n")
            f.write("=" * 60 + "\n\n")
            
            for i, chunk in enumerate(chunks[:10]):  # Show first 10 chunks
                source_info = chunker.get_chunk_source_info(chunk)
                
                f.write(f"=== CHUNK {i+1} ===\n")
                f.write(f"Chunk ID: {chunk.chunk_id}\n")
                f.write(f"Source: {source_info['citation']}\n")
                f.write(f"Size: {chunk.char_count} chars, {chunk.word_count} words\n")
                f.write(f"Block indices: {source_info['block_indices']}\n")
                f.write(f"Created: {source_info['created_at']}\n")
                
                if chunk.source.bbox:
                    f.write(f"Bbox: {chunk.source.bbox}\n")
                
                f.write(f"Content:\n{chunk.text[:300]}...\n")
                f.write("-" * 60 + "\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Pages covered: {len(set(c.source.page_number for c in chunks))}\n")
            f.write(f"Average blocks per chunk: {sum(len(c.source.block_indices) for c in chunks) / len(chunks):.1f}\n")
        
        print(f"📄 Detailed chunks exported to: {output_file}")
        
        # Assertions
        assert len(chunks) > 0, "Should generate chunks"
        assert all(c.source.file_path for c in chunks), "All chunks should have file path"
        assert all(c.source.chunk_id for c in chunks), "All chunks should have chunk ID"
        
        return chunks
    
    def test_source_citation_functionality(self):
        """Test tính năng citation và source tracking."""
        print(f"\n📚 TESTING CITATION & SOURCE TRACKING")
        
        # Load and chunk
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        chunker = BlockAwareChunker.create_citation_focused()
        chunks = chunker.chunk_document(document)
        
        # Test find chunks by source
        page_1_chunks = chunker.find_chunks_by_source(chunks, self.TEST_PDF_PATH, page_number=1)
        page_16_chunks = chunker.find_chunks_by_source(chunks, self.TEST_PDF_PATH, page_number=16)
        
        print(f"📊 CITATION RESULTS:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Page 1 chunks: {len(page_1_chunks)}")
        print(f"   Page 16 chunks: {len(page_16_chunks)}")
        
        # Show citation examples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        citation_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_citation_examples.txt")
        
        with open(citation_file, 'w', encoding='utf-8') as f:
            f.write("CITATION & SOURCE TRACKING EXAMPLES\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("=== PAGE 16 CHUNKS (Risk Approaches Section) ===\n")
            for chunk in page_16_chunks[:3]:  # First 3 chunks from page 16
                source_info = chunker.get_chunk_source_info(chunk)
                
                f.write(f"CHUNK: {chunk.chunk_id}\n")
                f.write(f"CITATION: {source_info['citation']}\n")
                f.write(f"CONTENT: {chunk.text[:200]}...\n")
                f.write(f"SOURCE DETAILS:\n")
                f.write(f"  - File: {source_info['file_path']}\n")
                f.write(f"  - Page: {source_info['page_number']}\n")
                f.write(f"  - Blocks: {source_info['block_indices']}\n")
                f.write(f"  - Bbox: {source_info['bbox']}\n")
                f.write("-" * 40 + "\n\n")
            
            # Show chunks containing specific text
            f.write("=== CHUNKS WITH 'RISK APPROACHES' TEXT ===\n")
            risk_chunks = [c for c in chunks if 'risk approaches' in c.text.lower()]
            
            for chunk in risk_chunks[:2]:
                source_info = chunker.get_chunk_source_info(chunk)
                f.write(f"FOUND IN: {source_info['citation']}\n")
                f.write(f"TEXT: {chunk.text}\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"📚 Citation examples exported to: {citation_file}")
        
        # Test assertions
        assert len(page_16_chunks) > 0, "Should find chunks on page 16"
        assert all('Process_Risk Management.pdf' in chunker.get_chunk_source_info(c)['citation'] for c in chunks[:5]), "Citations should contain filename"
        
        return page_16_chunks
    
    def test_chunker_vs_original_blocks_comparison(self):
        """So sánh chunker với original blocks để thấy improvement."""
        print(f"\n⚖️ TESTING CHUNKER vs ORIGINAL BLOCKS")
        
        # Load with merging disabled
        loader_no_merge = PDFLoader(enable_block_merging=False, extract_tables=False)
        doc_no_merge = loader_no_merge.load(self.TEST_PDF_PATH)
        
        # Load with merging enabled
        loader_merged = PDFLoader(enable_block_merging=True, extract_tables=False)
        doc_merged = loader_merged.load(self.TEST_PDF_PATH)
        
        # Chunk both versions
        chunker = BlockAwareChunker.create_from_merged_loader(chunk_size=1000)
        
        chunks_from_original = chunker.chunk_document(doc_no_merge)
        chunks_from_merged = chunker.chunk_document(doc_merged)
        
        print(f"📊 CHUNKER COMPARISON:")
        print(f"   Chunks from original blocks: {len(chunks_from_original)}")
        print(f"   Chunks from merged blocks: {len(chunks_from_merged)}")
        print(f"   Improvement: {len(chunks_from_original) - len(chunks_from_merged)} fewer chunks")
        
        # Quality comparison
        avg_size_original = sum(c.char_count for c in chunks_from_original) / len(chunks_from_original)
        avg_size_merged = sum(c.char_count for c in chunks_from_merged) / len(chunks_from_merged)
        
        print(f"   Average chunk size (original): {avg_size_original:.0f} chars")
        print(f"   Average chunk size (merged): {avg_size_merged:.0f} chars")
        
        # Export comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_chunker_comparison.txt")
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("CHUNKER: ORIGINAL vs MERGED BLOCKS COMPARISON\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Chunks from original blocks: {len(chunks_from_original)}\n")
            f.write(f"Chunks from merged blocks: {len(chunks_from_merged)}\n")
            f.write(f"Reduction: {len(chunks_from_original) - len(chunks_from_merged)} chunks\n")
            f.write(f"Average size (original): {avg_size_original:.0f} chars\n")
            f.write(f"Average size (merged): {avg_size_merged:.0f} chars\n")
            f.write(f"Size improvement: {avg_size_merged - avg_size_original:.0f} chars per chunk\n\n")
            
            # Show example chunks from both
            f.write("EXAMPLE: FIRST CHUNK FROM EACH\n")
            f.write("-" * 35 + "\n")
            
            if chunks_from_original:
                f.write("FROM ORIGINAL BLOCKS:\n")
                f.write(f"Size: {chunks_from_original[0].char_count} chars\n")
                f.write(f"Content: {chunks_from_original[0].text[:300]}...\n\n")
            
            if chunks_from_merged:
                f.write("FROM MERGED BLOCKS:\n")
                f.write(f"Size: {chunks_from_merged[0].char_count} chars\n")
                f.write(f"Content: {chunks_from_merged[0].text[:300]}...\n\n")
        
        print(f"⚖️ Comparison exported to: {comparison_file}")
        
        # Assertions
        assert len(chunks_from_merged) <= len(chunks_from_original), "Merged should result in fewer or equal chunks"
        assert avg_size_merged >= avg_size_original, "Merged chunks should be larger on average"
        
        return len(chunks_from_original), len(chunks_from_merged)
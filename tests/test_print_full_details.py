"""
Test để in chi tiết đầy đủ chunks và source tracking không có truncate.
"""

import pytest
import os
from datetime import datetime

from loaders.pdf_loader import PDFLoader
from chunkers.block_aware_chunker import BlockAwareChunker


class TestPrintFullChunkDetails:
    """
    Test class để in chi tiết đầy đủ chunks không có truncate.
    """
    
    TEST_PDF_PATH = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\data\pdf\Process_Risk Management.pdf"
    OUTPUT_DIR = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\test_outputs"
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup cho mỗi test method."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(self.TEST_PDF_PATH):
            pytest.skip(f"Test PDF file not found: {self.TEST_PDF_PATH}")
    
    def test_print_full_chunks_no_truncate(self):
        """In toàn bộ nội dung chunks chi tiết không truncate."""
        print(f"\n📄 PRINTING FULL CHUNK DETAILS (NO TRUNCATE)")
        
        # Load và chunk
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        chunker = BlockAwareChunker.create_citation_focused()
        chunks = chunker.chunk_document(document)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_full_chunks_details.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FULL CHUNK DETAILS - NO TRUNCATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Total pages: {len(set(c.source.page_number for c in chunks))}\n")
            f.write(f"Average chunk size: {sum(c.char_count for c in chunks) / len(chunks):.0f} chars\n")
            f.write(f"Average words per chunk: {sum(c.word_count for c in chunks) / len(chunks):.0f} words\n\n")
            
            for i, chunk in enumerate(chunks):
                source_info = chunker.get_chunk_source_info(chunk)
                
                f.write(f"{'='*80}\n")
                f.write(f"CHUNK {i+1} of {len(chunks)}\n")
                f.write(f"{'='*80}\n")
                
                f.write(f"CHUNK ID: {chunk.chunk_id}\n")
                f.write(f"CITATION: {source_info['citation']}\n")
                f.write(f"FILE: {source_info['file_path']}\n")
                f.write(f"PAGE: {source_info['page_number']}\n")
                f.write(f"BLOCK INDICES: {source_info['block_indices']}\n")
                f.write(f"CREATED: {source_info['created_at']}\n")
                f.write(f"SIZE: {chunk.char_count} characters, {chunk.word_count} words\n")
                
                if chunk.source.bbox:
                    bbox = chunk.source.bbox
                    f.write(f"BBOX: ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})\n")
                    f.write(f"BBOX AREA: {(bbox[2]-bbox[0]) * (bbox[3]-bbox[1]):.2f} square units\n")
                
                f.write(f"\nMETADATA:\n")
                for key, value in chunk.metadata.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write(f"\nFULL CONTENT:\n")
                f.write(f"{'-'*40}\n")
                f.write(f"{chunk.text}\n")
                f.write(f"{'-'*40}\n\n")
        
        print(f"📄 Full chunk details exported to: {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")
        print(f"   Total chunks: {len(chunks)}")
        
        return output_file
    
    def test_print_risk_approaches_chunks_full(self):
        """In chi tiết đầy đủ các chunks chứa 'risk approaches'."""
        print(f"\n🎯 PRINTING FULL RISK APPROACHES CHUNKS")
        
        # Load và chunk
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        chunker = BlockAwareChunker.create_citation_focused()
        chunks = chunker.chunk_document(document)
        
        # Tìm chunks chứa risk approaches
        risk_chunks = []
        for chunk in chunks:
            if 'risk approaches' in chunk.text.lower():
                risk_chunks.append(chunk)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_risk_approaches_full.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RISK APPROACHES CHUNKS - FULL DETAILS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Found {len(risk_chunks)} chunks containing 'risk approaches':\n\n")
            
            for i, chunk in enumerate(risk_chunks):
                source_info = chunker.get_chunk_source_info(chunk)
                
                f.write(f"{'='*60}\n")
                f.write(f"RISK APPROACHES CHUNK {i+1}\n")
                f.write(f"{'='*60}\n")
                
                f.write(f"CHUNK ID: {chunk.chunk_id}\n")
                f.write(f"EXACT CITATION: {source_info['citation']}\n")
                f.write(f"SOURCE FILE: {source_info['file_path'].split('/')[-1]}\n")
                f.write(f"PAGE NUMBER: {source_info['page_number']}\n")
                f.write(f"BLOCK INDICES: {source_info['block_indices']}\n")
                f.write(f"CHUNK SIZE: {chunk.char_count} chars, {chunk.word_count} words\n")
                
                if chunk.source.bbox:
                    bbox = chunk.source.bbox
                    f.write(f"POSITION (BBOX): x={bbox[0]:.1f}-{bbox[2]:.1f}, y={bbox[1]:.1f}-{bbox[3]:.1f}\n")
                
                f.write(f"CREATED AT: {source_info['created_at']}\n")
                
                f.write(f"\nCOMPLETE TEXT CONTENT:\n")
                f.write(f"{'='*40}\n")
                f.write(f"{chunk.text}\n")
                f.write(f"{'='*40}\n\n")
                
                # Highlight the risk approaches part
                if 'risk approaches' in chunk.text.lower():
                    text_lower = chunk.text.lower()
                    start_idx = text_lower.find('risk approaches')
                    if start_idx >= 0:
                        # Get context around "risk approaches"
                        context_start = max(0, start_idx - 100)
                        context_end = min(len(chunk.text), start_idx + 200)
                        context = chunk.text[context_start:context_end]
                        
                        f.write(f"HIGHLIGHTED CONTEXT (±100 chars around 'risk approaches'):\n")
                        f.write(f"{'~'*50}\n")
                        f.write(f"{context}\n")
                        f.write(f"{'~'*50}\n\n")
        
        print(f"🎯 Risk approaches chunks exported to: {output_file}")
        print(f"   Found: {len(risk_chunks)} chunks")
        print(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")
        
        return risk_chunks
    
    def test_print_page16_chunks_complete(self):
        """In chi tiết đầy đủ tất cả chunks từ page 16 (page có vấn đề chunking)."""
        print(f"\n📄 PRINTING COMPLETE PAGE 16 CHUNKS")
        
        # Load và chunk
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        chunker = BlockAwareChunker.create_citation_focused()
        chunks = chunker.chunk_document(document)
        
        # Tìm tất cả chunks từ page 16
        page16_chunks = chunker.find_chunks_by_source(chunks, self.TEST_PDF_PATH, page_number=16)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_page16_complete.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("PAGE 16 CHUNKS - COMPLETE DETAILS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Page 16 contains {len(page16_chunks)} chunks:\n\n")
            
            for i, chunk in enumerate(page16_chunks):
                source_info = chunker.get_chunk_source_info(chunk)
                
                f.write(f"{'='*60}\n")
                f.write(f"PAGE 16 - CHUNK {i+1} of {len(page16_chunks)}\n")
                f.write(f"{'='*60}\n")
                
                f.write(f"CHUNK ID: {chunk.chunk_id}\n")
                f.write(f"FULL CITATION: {source_info['citation']}\n")
                f.write(f"BLOCKS USED: {source_info['block_indices']}\n")
                f.write(f"TEXT LENGTH: {chunk.char_count} characters\n")
                f.write(f"WORD COUNT: {chunk.word_count} words\n")
                
                if chunk.source.bbox:
                    bbox = chunk.source.bbox
                    f.write(f"POSITION: Top-left ({bbox[0]:.1f}, {bbox[1]:.1f}), Bottom-right ({bbox[2]:.1f}, {bbox[3]:.1f})\n")
                    f.write(f"DIMENSIONS: {bbox[2]-bbox[0]:.1f} x {bbox[3]-bbox[1]:.1f} units\n")
                
                f.write(f"TIMESTAMP: {source_info['created_at']}\n")
                
                f.write(f"\nCOMPLETE CHUNK TEXT:\n")
                f.write(f"{'='*50}\n")
                f.write(f"{chunk.text}\n")
                f.write(f"{'='*50}\n\n")
                
                # Show word frequency for analysis
                words = chunk.text.lower().split()
                word_freq = {}
                for word in words:
                    clean_word = ''.join(c for c in word if c.isalnum())
                    if len(clean_word) > 3:  # Only words longer than 3 chars
                        word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
                
                # Top 5 most frequent words
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                if top_words:
                    f.write(f"TOP WORDS IN THIS CHUNK:\n")
                    for word, freq in top_words:
                        f.write(f"  '{word}': {freq} times\n")
                    f.write(f"\n")
        
        print(f"📄 Page 16 complete chunks exported to: {output_file}")
        print(f"   Chunks on page 16: {len(page16_chunks)}")
        print(f"   Total characters: {sum(c.char_count for c in page16_chunks)}")
        print(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")
        
        return page16_chunks
"""
Test để tìm và in toàn bộ chunk IDs chứa các blocks cụ thể.
"""

import pytest
import os
from datetime import datetime

from loaders.pdf_loader import PDFLoader
from chunkers.block_aware_chunker import BlockAwareChunker


class TestFindSpecificBlocks:
    """
    Test class để tìm chunks chứa các block indices cụ thể.
    """
    
    TEST_PDF_PATH = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\data\pdf\Process_Risk Management.pdf"
    OUTPUT_DIR = r"C:\Users\ENGUYEHWC\Downloads\RAG\RAG\test_outputs"
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup cho mỗi test method."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(self.TEST_PDF_PATH):
            pytest.skip(f"Test PDF file not found: {self.TEST_PDF_PATH}")
    
    def test_find_chunks_with_specific_blocks(self):
        """Tìm tất cả chunks chứa các block indices: 17,18,19,20,21,22,23,24,25,26."""
        print(f"\n🔍 FINDING CHUNKS WITH SPECIFIC BLOCK INDICES")
        
        # Target blocks
        target_blocks = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        print(f"Target blocks: {target_blocks}")
        
        # Load và chunk
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        chunker = BlockAwareChunker.create_citation_focused()
        chunks = chunker.chunk_document(document)
        
        # Tìm chunks chứa bất kỳ target blocks nào
        matching_chunks = []
        
        for chunk in chunks:
            # Check xem chunk có chứa bất kỳ target block nào không
            chunk_blocks = chunk.source.block_indices
            has_target_block = any(block_idx in target_blocks for block_idx in chunk_blocks)
            
            if has_target_block:
                # Tính số lượng target blocks trong chunk này
                matching_blocks = [b for b in chunk_blocks if b in target_blocks]
                matching_chunks.append({
                    'chunk': chunk,
                    'matching_blocks': matching_blocks,
                    'match_count': len(matching_blocks)
                })
        
        print(f"📊 SEARCH RESULTS:")
        print(f"   Found {len(matching_chunks)} chunks containing target blocks")
        
        # Sort by match count (descending)
        matching_chunks.sort(key=lambda x: x['match_count'], reverse=True)
        
        # Export detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_chunks_with_specific_blocks.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("CHUNKS CONTAINING SPECIFIC BLOCK INDICES\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"TARGET BLOCKS: {target_blocks}\n")
            f.write(f"TOTAL CHUNKS FOUND: {len(matching_chunks)}\n\n")
            
            for i, item in enumerate(matching_chunks):
                chunk = item['chunk']
                matching_blocks = item['matching_blocks']
                source_info = chunker.get_chunk_source_info(chunk)
                
                f.write(f"{'='*70}\n")
                f.write(f"RESULT {i+1} of {len(matching_chunks)}\n")
                f.write(f"{'='*70}\n")
                
                f.write(f"CHUNK ID: {chunk.chunk_id}\n")
                f.write(f"CITATION: {source_info['citation']}\n")
                f.write(f"PAGE: {source_info['page_number']}\n")
                f.write(f"ALL BLOCK INDICES: {source_info['block_indices']}\n")
                f.write(f"MATCHING TARGET BLOCKS: {matching_blocks}\n")
                f.write(f"MATCH COUNT: {len(matching_blocks)} out of {len(target_blocks)} target blocks\n")
                f.write(f"CHUNK SIZE: {chunk.char_count} chars, {chunk.word_count} words\n")
                
                if chunk.source.bbox:
                    bbox = chunk.source.bbox
                    f.write(f"POSITION: ({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})\n")
                
                f.write(f"CREATED: {source_info['created_at']}\n")
                
                f.write(f"\nFULL CHUNK CONTENT:\n")
                f.write(f"{'='*50}\n")
                f.write(f"{chunk.text}\n")
                f.write(f"{'='*50}\n\n")
                
                # Highlight specific parts if recognizable
                text_lower = chunk.text.lower()
                keywords = ['risk approaches', 'asset based', 'business based', 'operational']
                found_keywords = [kw for kw in keywords if kw in text_lower]
                if found_keywords:
                    f.write(f"CONTAINS KEYWORDS: {', '.join(found_keywords)}\n")
                
                f.write(f"\n")
            
            # Summary table
            f.write(f"{'='*70}\n")
            f.write(f"SUMMARY TABLE\n")
            f.write(f"{'='*70}\n")
            f.write(f"{'Chunk ID':<15} {'Page':<6} {'Blocks':<25} {'Matches':<10} {'Size':<8}\n")
            f.write(f"{'-'*70}\n")
            
            for item in matching_chunks:
                chunk = item['chunk']
                matching_blocks = item['matching_blocks']
                blocks_str = str(chunk.source.block_indices)
                if len(blocks_str) > 22:
                    blocks_str = blocks_str[:19] + "..."
                
                f.write(f"{chunk.chunk_id:<15} {chunk.source.page_number:<6} {blocks_str:<25} {len(matching_blocks):<10} {chunk.char_count:<8}\n")
            
            # Coverage analysis
            f.write(f"\n{'='*70}\n")
            f.write(f"COVERAGE ANALYSIS\n")
            f.write(f"{'='*70}\n")
            
            all_found_blocks = set()
            for item in matching_chunks:
                all_found_blocks.update(item['matching_blocks'])
            
            missing_blocks = set(target_blocks) - all_found_blocks
            
            f.write(f"Target blocks: {sorted(target_blocks)}\n")
            f.write(f"Found blocks: {sorted(all_found_blocks)}\n")
            f.write(f"Missing blocks: {sorted(missing_blocks) if missing_blocks else 'None'}\n")
            f.write(f"Coverage: {len(all_found_blocks)}/{len(target_blocks)} ({len(all_found_blocks)/len(target_blocks)*100:.1f}%)\n")
        
        print(f"📄 Detailed results exported to: {output_file}")
        print(f"   Matching chunks: {len(matching_chunks)}")
        
        # Print summary to console
        print(f"\n📋 CHUNK IDs SUMMARY:")
        for i, item in enumerate(matching_chunks):
            chunk = item['chunk']
            matching_blocks = item['matching_blocks']
            print(f"   {i+1}. {chunk.chunk_id} - Page {chunk.source.page_number} - Blocks {matching_blocks} - {chunk.char_count} chars")
        
        # Check coverage
        all_found_blocks = set()
        for item in matching_chunks:
            all_found_blocks.update(item['matching_blocks'])
        
        missing_blocks = set(target_blocks) - all_found_blocks
        print(f"\n📊 COVERAGE:")
        print(f"   Found blocks: {sorted(all_found_blocks)}")
        print(f"   Missing blocks: {sorted(missing_blocks) if missing_blocks else 'None'}")
        print(f"   Coverage: {len(all_found_blocks)}/{len(target_blocks)} ({len(all_found_blocks)/len(target_blocks)*100:.1f}%)")
        
        return matching_chunks
    
    def test_find_chunks_by_individual_blocks(self):
        """Tìm chunk cho từng block riêng lẻ để xem distribution."""
        print(f"\n🎯 FINDING CHUNKS FOR INDIVIDUAL BLOCKS")
        
        target_blocks = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        
        # Load và chunk
        loader = PDFLoader.create_default()
        document = loader.load(self.TEST_PDF_PATH)
        
        chunker = BlockAwareChunker.create_citation_focused()
        chunks = chunker.chunk_document(document)
        
        # Tìm chunk cho từng block
        block_to_chunk = {}
        
        for target_block in target_blocks:
            for chunk in chunks:
                if target_block in chunk.source.block_indices:
                    if target_block not in block_to_chunk:
                        block_to_chunk[target_block] = []
                    block_to_chunk[target_block].append(chunk.chunk_id)
        
        # Export mapping
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mapping_file = os.path.join(self.OUTPUT_DIR, f"{timestamp}_block_to_chunk_mapping.txt")
        
        with open(mapping_file, 'w', encoding='utf-8') as f:
            f.write("BLOCK TO CHUNK ID MAPPING\n")
            f.write("=" * 50 + "\n\n")
            
            for block_idx in sorted(target_blocks):
                chunk_ids = block_to_chunk.get(block_idx, [])
                f.write(f"Block {block_idx:2d}: {chunk_ids if chunk_ids else 'NOT FOUND'}\n")
            
            f.write(f"\nSUMMARY:\n")
            f.write(f"Target blocks: {len(target_blocks)}\n")
            f.write(f"Found blocks: {len(block_to_chunk)}\n")
            f.write(f"Missing blocks: {set(target_blocks) - set(block_to_chunk.keys())}\n")
        
        print(f"📄 Block mapping exported to: {mapping_file}")
        
        # Print mapping to console
        print(f"\n📋 BLOCK TO CHUNK MAPPING:")
        for block_idx in sorted(target_blocks):
            chunk_ids = block_to_chunk.get(block_idx, [])
            status = chunk_ids if chunk_ids else "NOT FOUND"
            print(f"   Block {block_idx:2d}: {status}")
        
        return block_to_chunk
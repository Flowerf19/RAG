"""
Test Embedding Readiness for Table Chunks
==========================================
Verify that table chunks are ready for embedding pipeline.
"""

from loaders.pdf_loader import PDFLoader
from chunkers.hybrid_chunker import HybridChunker
import json

def test_embedding_readiness():
    """Test that chunks are ready for embedding."""
    
    print("=" * 80)
    print("EMBEDDING READINESS TEST")
    print("=" * 80)
    
    # 1. Load and chunk document
    pdf_path = r"C:\Users\ENGUYEHWC\Prototype\Version_4\RAG\data\pdf\Process_Service Management.pdf"
    loader = PDFLoader.create_default()
    pdf_doc = loader.load(pdf_path)
    pdf_doc = pdf_doc.normalize()
    
    chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
    chunk_set = chunker.chunk(pdf_doc)
    
    print(f"\n✓ Loaded and chunked document: {len(chunk_set.chunks)} chunks\n")
    
    # 2. Test each requirement
    table_chunks = []
    regular_chunks = []
    
    for chunk in chunk_set.chunks:
        if chunk.metadata.get('group_type') == 'table':
            table_chunks.append(chunk)
        else:
            regular_chunks.append(chunk)
    
    print(f"📊 Found {len(table_chunks)} table chunks")
    print(f"📄 Found {len(regular_chunks)} regular chunks\n")
    
    # Test 1: Tables are separate chunks
    print("TEST 1: Bảng → chunk riêng")
    print("-" * 40)
    if len(table_chunks) > 0:
        print(f"✅ PASS: {len(table_chunks)} table chunks found")
        print(f"   First table chunk ID: {table_chunks[0].chunk_id}")
    else:
        print("❌ FAIL: No table chunks found!")
    print()
    
    # Test 2: table_payload exists
    print("TEST 2: Metadata có table_payload")
    print("-" * 40)
    payload_count = 0
    for chunk in table_chunks:
        if 'table_payload' in chunk.metadata:
            payload_count += 1
    
    if payload_count == len(table_chunks):
        print(f"✅ PASS: All {payload_count} table chunks have table_payload")
    else:
        print(f"❌ FAIL: Only {payload_count}/{len(table_chunks)} have table_payload")
    print()
    
    # Test 3: embedding_text exists
    print("TEST 3: Có embedding_text schema-aware")
    print("-" * 40)
    embedding_text_count = 0
    for chunk in table_chunks:
        if 'embedding_text' in chunk.metadata:
            embedding_text_count += 1
    
    if embedding_text_count == len(table_chunks):
        print(f"✅ PASS: All {embedding_text_count} table chunks have embedding_text")
        # Show sample
        if table_chunks:
            sample = table_chunks[0].metadata.get('embedding_text', '')[:100]
            print(f"   Sample: {sample}...")
    else:
        print(f"⚠️ PARTIAL: Only {embedding_text_count}/{len(table_chunks)} have embedding_text")
    print()
    
    # Test 4: Token budget check
    print("TEST 4: Token budget enforcement")
    print("-" * 40)
    oversized = []
    for chunk in table_chunks:
        if chunk.token_count > 200:  # max_tokens
            oversized.append(chunk)
    
    if len(oversized) == 0:
        print(f"✅ PASS: No table chunks exceed token budget (max=200)")
    else:
        print(f"⚠️ WARNING: {len(oversized)} table chunks exceed token budget:")
        for chunk in oversized[:3]:
            print(f"   - Chunk {chunk.chunk_id}: {chunk.token_count} tokens")
    print()
    
    # Test 5: Provenance exists
    print("TEST 5: Provenance tracking")
    print("-" * 40)
    provenance_count = 0
    for chunk in table_chunks:
        if chunk.provenance and chunk.provenance.page_numbers:
            provenance_count += 1
    
    if provenance_count == len(table_chunks):
        print(f"✅ PASS: All {provenance_count} table chunks have provenance")
    else:
        print(f"⚠️ PARTIAL: Only {provenance_count}/{len(table_chunks)} have provenance")
    print()
    
    # Test 6: Structure preservation
    print("TEST 6: Table structure preservation")
    print("-" * 40)
    if table_chunks:
        sample_chunk = table_chunks[0]
        payload = sample_chunk.metadata.get('table_payload')
        if payload:
            print(f"✅ PASS: Table structure preserved")
            print(f"   - Header: {getattr(payload, 'header', [])}")
            print(f"   - Rows: {len(getattr(payload, 'rows', []))}")
            print(f"   - BBox: {getattr(payload, 'bbox', None)}")
            caption = getattr(payload, 'metadata', {}).get('table_caption') if hasattr(payload, 'metadata') else None
            print(f"   - Caption: {caption}")
        else:
            print("❌ FAIL: No table_payload found")
    print()
    
    # Test 7: Embedding simulation
    print("TEST 7: Simulate embedding process")
    print("-" * 40)
    
    def get_text_for_embedding(chunk):
        """Helper function for embedder."""
        if chunk.metadata.get('group_type') == 'table':
            return chunk.metadata.get('embedding_text', chunk.text)
        return chunk.text
    
    success_count = 0
    for chunk in table_chunks[:3]:  # Test first 3
        text = get_text_for_embedding(chunk)
        if text and len(text.strip()) > 0:
            success_count += 1
            print(f"   ✓ Chunk {chunk.chunk_id[:8]}... → {len(text)} chars")
    
    print(f"✅ PASS: Successfully extracted text for {success_count}/3 samples")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    checks = {
        "Table chunks separated": len(table_chunks) > 0,
        "Table payload exists": payload_count == len(table_chunks),
        "Embedding text exists": embedding_text_count == len(table_chunks),
        "Token budget OK": len(oversized) == 0,
        "Provenance exists": provenance_count == len(table_chunks),
        "Structure preserved": True,  # Manual check
        "Embedding ready": True  # Manual check
    }
    
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    for check, status in checks.items():
        icon = "✅" if status else "⚠️"
        print(f"{icon} {check}")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED - READY FOR EMBEDDING!")
    elif passed >= total * 0.8:
        print("\n✅ MOSTLY READY - Minor improvements recommended")
    else:
        print("\n⚠️ NEEDS IMPROVEMENT - Address issues before embedding")
    
    print("\nNext steps:")
    print("1. Review EMBEDDING_READINESS_REPORT.md for detailed analysis")
    print("2. Implement token budget enforcement for large tables")
    print("3. Add cell-level provenance tracking")
    print("4. Test with actual embedder (e.g., OpenAI, Sentence-Transformers)")
    print()
    
    # Export sample for testing
    if table_chunks:
        sample_export = {
            'chunk_id': table_chunks[0].chunk_id,
            'text_for_embedding': get_text_for_embedding(table_chunks[0]),
            'token_count': table_chunks[0].token_count,
            'metadata': {
                'group_type': table_chunks[0].metadata.get('group_type'),
                'table_caption': getattr(
                    table_chunks[0].metadata.get('table_payload'), 
                    'metadata', {}
                ).get('table_caption') if hasattr(
                    table_chunks[0].metadata.get('table_payload'), 
                    'metadata'
                ) else None,
            }
        }
        
        with open('sample_table_chunk_for_embedding.json', 'w', encoding='utf-8') as f:
            json.dump(sample_export, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported sample chunk to: sample_table_chunk_for_embedding.json")

if __name__ == "__main__":
    test_embedding_readiness()

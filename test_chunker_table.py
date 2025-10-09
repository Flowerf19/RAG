"""
Test script để kiểm tra xem chunker có nhận được table_schema từ blocks không
"""

from loaders.pdf_loader import PDFLoader
from chunkers.hybrid_chunker import HybridChunker

pdf_path = r"C:\Users\ENGUYEHWC\Prototype\Version_4\RAG\data\pdf\Process_Service Management.pdf"

# 1. Load PDF
loader = PDFLoader.create_default()
pdf_doc = loader.load(pdf_path)

# 2. Chunk
chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
chunk_set = chunker.chunk(pdf_doc)

# 3. Kiểm tra metadata trong chunks
print("=== Kiểm tra table_schema trong chunks ===\n")

for i, chunk in enumerate(chunk_set.chunks):
    if chunk.metadata and (chunk.metadata.get("group_type") == "table" or chunk.metadata.get("block_type") == "table"):
        print(f"Chunk {i+1}:")
        print(f"  metadata keys: {list(chunk.metadata.keys())}")
        print(f"  group_type: {chunk.metadata.get('group_type')}")
        print(f"  block_type: {chunk.metadata.get('block_type')}")
        
        # Kiểm tra table_payload
        table_payload = chunk.metadata.get("table_payload")
        print(f"  Has table_payload: {table_payload is not None}")
        if table_payload:
            print(f"    table_payload type: {type(table_payload)}")
            if hasattr(table_payload, 'matrix'):
                print(f"    matrix rows: {len(table_payload.matrix) if table_payload.matrix else 0}")
        
        # Kiểm tra table_schema
        table_schema = chunk.metadata.get("table_schema")
        print(f"  Has table_schema: {table_schema is not None}")
        if table_schema:
            print(f"    table_schema type: {type(table_schema)}")
        
        # Kiểm tra embedding_text
        embedding_text = chunk.metadata.get("embedding_text")
        print(f"  Has embedding_text: {embedding_text is not None}")
        if embedding_text:
            print(f"    embedding_text length: {len(embedding_text)}")
        
        print()

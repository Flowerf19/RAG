from loaders import PDFLoader
from chunkers import HybridChunker
from chunkers.hybrid_chunker import ChunkerMode

loader = PDFLoader.create_default()
doc = loader.load('data/pdf/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy.pdf')

print(f'Total pages: {len(doc.pages)}')
print(f'Total blocks: {sum(len(p.blocks) for p in doc.pages)}')

for i in range(min(5, len(doc.pages))):
    print(f'Page {i+1}: {len(doc.pages[i].blocks)} blocks')

# Use SAME config as pipeline
chunker = HybridChunker(
    max_tokens=250,
    overlap_tokens=35,
    mode=ChunkerMode.SEMANTIC_FIRST  # Same as pipeline!
)
chunk_set = chunker.chunk(doc)

print(f'\nChunker created {len(chunk_set.chunks)} chunks')

# Check which pages are covered
all_pages = set()
for c in chunk_set.chunks:
    if hasattr(c, 'provenance') and c.provenance and hasattr(c.provenance, 'page_numbers'):
        all_pages.update(c.provenance.page_numbers)

print(f'Pages covered: {sorted(all_pages)}')
print(f'Missing pages: {set(range(1, 21)) - all_pages}')

# Check chunks by page - show first 10
print('\nFirst 10 chunks:')
for i in range(min(10, len(chunk_set.chunks))):
    c = chunk_set.chunks[i]
    pages = []
    if hasattr(c, 'provenance') and c.provenance and hasattr(c.provenance, 'page_numbers'):
        pages = sorted(c.provenance.page_numbers)
    text_preview = c.text[:60].replace('\n', ' ')
    print(f'Chunk {i+1} (pages {pages}): "{text_preview}..."')

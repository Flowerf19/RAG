from loaders import PDFLoader
from loaders.model.block import TableBlock

loader = PDFLoader.create_default()
doc = loader.load('data/pdf/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy.pdf')

# Check missing pages: 2, 3, 5, 6, 8
print("Checking pages that are missing from chunks:\n")

for page_num in [2, 3, 5, 6, 8]:
    page = doc.pages[page_num - 1]
    print(f'Page {page_num}: {len(page.blocks)} blocks')
    
    for i, b in enumerate(page.blocks[:3]):  # Check first 3 blocks
        has_metadata = b.metadata is not None
        page_in_metadata = b.metadata.get('page_number') if has_metadata else None
        text_preview = b.text[:50].replace('\n', ' ')
        print(f'  Block {i}: metadata={has_metadata}, page_number={page_in_metadata}, text="{text_preview}..."')
    print()

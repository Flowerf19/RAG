"""
Test script để kiểm tra xem loader có gán TableSchema vào block.metadata không
"""

from loaders.pdf_loader import PDFLoader

pdf_path = r"C:\Users\ENGUYEHWC\Prototype\Version_4\RAG\data\pdf\Process_Service Management.pdf"

# Load PDF
loader = PDFLoader.create_default()
pdf_doc = loader.load(pdf_path)

# Kiểm tra các blocks trong từng page
print("=== Kiểm tra TableSchema trong blocks ===\n")

for page in pdf_doc.pages:
    print(f"Page {page.page_number}:")
    for i, block in enumerate(page.blocks):
        if hasattr(block, 'block_type') and block.block_type == "table":
            print(f"  Block {i}: block_type={block.block_type}")
            print(f"    Has metadata: {block.metadata is not None}")
            if block.metadata:
                print(f"    Metadata keys: {list(block.metadata.keys())}")
                if 'table_schema' in block.metadata:
                    table_schema = block.metadata['table_schema']
                    print(f"    table_schema type: {type(table_schema)}")
                    print(f"    table_schema: {table_schema}")
                    if hasattr(table_schema, 'matrix'):
                        print(f"    Matrix rows: {len(table_schema.matrix) if table_schema.matrix else 0}")
                else:
                    print(f"    ❌ NO table_schema in metadata!")
            else:
                print(f"    ❌ NO metadata!")
            
            # Kiểm tra thuộc tính table của TableBlock
            if hasattr(block, 'table'):
                print(f"    Has table attribute: {block.table is not None}")
                if block.table:
                    print(f"    table type: {type(block.table)}")
                    if hasattr(block.table, 'matrix'):
                        print(f"    table.matrix rows: {len(block.table.matrix) if block.table.matrix else 0}")
            print()

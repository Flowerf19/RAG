import json
from loaders.pdf_loader import PDFLoader
from dataclasses import asdict

def export_blocks_and_tables(pdf_path, out_block, out_table):
    loader = PDFLoader()
    doc = loader.load_pdf(pdf_path)
    # Export all blocks from all pages
    all_blocks = []
    for page in doc.pages:
        for block in page.blocks:
            if hasattr(block, '__dict__'):
                all_blocks.append(block.__dict__)
            else:
                all_blocks.append(block)
    # Export all tables from all pages
    all_tables = []
    for page in doc.pages:
        for table in page.tables:
            if hasattr(table, '__dict__'):
                all_tables.append(table.__dict__)
            else:
                all_tables.append(table)
    with open(out_block, 'w', encoding='utf-8') as f:
        json.dump(all_blocks, f, ensure_ascii=False, indent=2)
    with open(out_table, 'w', encoding='utf-8') as f:
        json.dump(all_tables, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    export_blocks_and_tables(
        r"data/pdf/Process_Risk Management.pdf",
        r"data/pdf/Process_Risk_Management_blocks.json",
        r"data/pdf/Process_Risk_Management_tables.json"
    )

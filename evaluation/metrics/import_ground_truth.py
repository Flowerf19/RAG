"""
Import ground-truth Q&A from Excel into MetricsDB
Expected columns: STT, Câu hỏi, Câu trả lời, Nguồn
Run inside project venv.
"""
import sys
import os
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from evaluation.metrics.database import MetricsDB


def import_from_excel(path: str, db_path: str = "data/metrics.db"):
    df = pd.read_excel(path)

    # Normalize columns (Vietnamese headings)
    col_map = {}
    for c in df.columns:
        cn = c.strip().lower()
        if 'stt' in cn:
            col_map[c] = 'id'
        elif 'câu hỏi' in cn or 'cau hoi' in cn or 'question' in cn:
            col_map[c] = 'question'
        elif 'câu trả lời' in cn or 'cau tra loi' in cn or 'answer' in cn:
            col_map[c] = 'answer'
        elif 'nguồn' in cn or 'nguon' in cn or 'source' in cn:
            col_map[c] = 'source'

    df = df.rename(columns=col_map)

    required = ['question', 'answer']
    if not all(r in df.columns for r in required):
        raise ValueError(f"Excel must contain columns: {required}")

    db = MetricsDB(db_path)
    inserted = 0
    for _, row in df.iterrows():
        q = str(row.get('question', '')).strip()
        a = str(row.get('answer', '')).strip()
        s = row.get('source') if 'source' in row.index else None
        if not q or not a:
            continue
        db.insert_ground_truth(q, a, s)
        inserted += 1

    print(f"Inserted {inserted} ground-truth QA pairs into {db_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('excel_path', help='Path to Excel file')
    p.add_argument('--db', default='data/metrics.db', help='Path to metrics DB')
    args = p.parse_args()
    import_from_excel(args.excel_path, args.db)

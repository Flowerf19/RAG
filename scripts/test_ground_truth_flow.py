"""Test script for ground-truth import + optional evaluation

- Inserts a few sample ground-truth rows via BackendDashboard.insert_ground_truth_rows
- Verifies insertion via get_ground_truth_list
- Optionally runs a small evaluation (semantic similarity/recall/relevance) when --run-eval is passed

Usage:
    python scripts/test_ground_truth_flow.py            # insert and verify
    python scripts/test_ground_truth_flow.py --run-eval --limit 2   # also run evaluations on 2 rows

Note: Evaluations may attempt retrieval/embedding and can fail if environment lacks models. The script will catch and print any errors.
"""

import argparse
import sys
from pprint import pprint

from evaluation.backend_dashboard.api import BackendDashboard

SAMPLE_ROWS = [
    {
        'question': 'Chủ sở hữu của Quy trình Quản lý Rủi ro là ai?',
        'answer': '',
        'source': 'Process_Risk_Management.pdf'
    },
    {
        'question': 'Mục đích của Quy trình Quản lý Rủi ro là gì?',
        'answer': '',
        'source': 'Process_Risk_Management.pdf'
    }
]


def main(run_eval: bool = False, eval_limit: int = 2):
    backend = BackendDashboard()

    print("== Inserting sample ground-truth rows ==")
    try:
        inserted = backend.insert_ground_truth_rows(SAMPLE_ROWS)
        print(f"Inserted rows: {inserted}")
    except Exception as e:
        print(f"Failed to insert rows: {e}")
        sys.exit(1)

    print("\n== Verifying inserted rows (latest) ==")
    try:
        rows = backend.get_ground_truth_list(limit=10)
        # Print rows that match sample questions
        matched = [r for r in rows if any(s['question'] in (r.get('question') or '') for s in SAMPLE_ROWS)]
        if not matched:
            print("No matching rows found - check DB or insertion logic.")
        else:
            pprint(matched[:10])
    except Exception as e:
        print(f"Failed to fetch ground-truth list: {e}")

    if run_eval:
        print("\n== Running small evaluations (may be slow) ==")
        try:
            print("-> Semantic similarity (limit={})".format(eval_limit))
            sem = backend.evaluate_ground_truth_with_semantic_similarity(limit=eval_limit, save_to_db=False)
            pprint(sem.get('summary', {}))
        except Exception as e:
            print(f"Semantic similarity evaluation failed: {e}")

        try:
            print("-> Recall (limit={})".format(eval_limit))
            rec = backend.evaluate_recall(limit=eval_limit, save_to_db=False)
            pprint(rec.get('summary', {}))
        except Exception as e:
            print(f"Recall evaluation failed: {e}")

        try:
            print("-> Relevance (limit={})".format(eval_limit))
            rel = backend.evaluate_relevance(limit=eval_limit, save_to_db=False)
            pprint(rel.get('summary', {}))
        except Exception as e:
            print(f"Relevance evaluation failed: {e}")

    print('\nDone')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ground-truth import and optional evaluation')
    parser.add_argument('--run-eval', action='store_true', help='Run small evaluations after insert')
    parser.add_argument('--limit', type=int, default=2, help='Limit number of rows to evaluate (if --run-eval)')
    args = parser.parse_args()

    main(run_eval=args.run_eval, eval_limit=args.limit)

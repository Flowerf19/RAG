#!/usr/bin/env python3
"""
Debug script to check retrieved sources vs ground truth source.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from evaluation.metrics.database import MetricsDB
from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval

def debug_recall():
    """Debug why recall is 0."""
    # Get ground truth
    db = MetricsDB()
    rows = db.get_ground_truth_list(limit=5)

    for r in rows[:3]:  # Check first 3
        gt_id = r.get('id')
        question = r.get('question')
        true_source = r.get('source', '').strip()

        print(f"\n=== Ground Truth ID {gt_id} ===")
        print(f"Question: {question}")
        print(f"True source (first 200 chars): {repr(true_source[:200])}")

        # Get retrieval results
        result = fetch_retrieval(question, top_k=5)
        sources = result.get('sources', [])

        print(f"Retrieved {len(sources)} sources:")

        matched = 0
        for i, s in enumerate(sources):
            chunk_text = s.get('text', s.get('content', s.get('snippet', '')))
            print(f"  Chunk {i+1} (first 200 chars): {repr(chunk_text[:200])}")

            # Check substring matching
            if true_source and chunk_text:
                if true_source in chunk_text or chunk_text in true_source:
                    matched += 1
                    print("    *** MATCH! ***")

        recall = matched / len(sources) if sources else 0.0
        print(f"Matched chunks: {matched}/{len(sources)} = Recall: {recall}")

if __name__ == "__main__":
    debug_recall()
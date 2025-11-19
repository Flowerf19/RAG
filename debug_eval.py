#!/usr/bin/env python3
"""
Debug ground truth evaluation
"""

import logging
logging.basicConfig(level=logging.DEBUG)

from evaluation.backend_dashboard.api import BackendDashboard

backend = BackendDashboard()
rows = backend.get_ground_truth_list(limit=1)
print(f'Found {len(rows)} ground truth rows')
if rows:
    row = rows[0]
    question = row.get('question', '')
    answer = row.get('answer', '')
    source = row.get('source', '')
    print(f'Question: {question}')
    print(f'Answer: {answer}')
    print(f'Source length: {len(source)}')
    print(f'Source preview: {source[:200]}...')

# Test Ragas evaluation directly
if rows:
    print('\nTesting Ragas evaluation on first row...')
    try:
        result = backend.evaluate_ground_truth_with_ragas(
            limit=1,  # Just one question
            save_to_db=False
        )
        summary = result.get('summary', {})
        print(f'Ragas evaluation result: {summary}')
    except Exception as e:
        print(f'Ragas evaluation failed: {e}')
        import traceback
        traceback.print_exc()
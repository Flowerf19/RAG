#!/usr/bin/env python3
"""
Check ground truth data in database
"""

from evaluation.backend_dashboard.api import BackendDashboard
backend = BackendDashboard()
rows = backend.get_ground_truth_list(limit=10)
print(f'Ground truth rows in database: {len(rows)}')
for i, row in enumerate(rows[:3], 1):
    question = row.get('question', '')[:50]
    answer = row.get('answer', '')[:50]
    print(f'{i}. Q: {question}...')
    print(f'   A: {answer}...')
#!/usr/bin/env python3
"""
Check retrieved sources for first question
"""

from evaluation.backend_dashboard.api import BackendDashboard

backend = BackendDashboard()
rows = backend.get_ground_truth_list(limit=1)
if rows:
    row = rows[0]
    question = row.get('question', '')
    answer = row.get('answer', '')
    source = row.get('source', '')

    print(f'Question: {question}')
    print(f'Expected Answer: {answer}')
    print(f'Ground Truth Source: {source[:200]}...')

    # Get retrieval result
    from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval
    result = fetch_retrieval(
        query_text=question,
        top_k=5,
        embedder_type='ollama',
        reranker_type='none',
        use_query_enhancement=True,
        evaluate_response=False
    )

    print(f'\nRetrieved {len(result["sources"])} sources:')
    for i, src in enumerate(result['sources']):
        print(f'\nSource {i+1}:')
        print(f'File: {src.get("file_name", "N/A")}')
        print(f'Page: {src.get("page_number", "N/A")}')
        text = src.get('full_text', src.get('text', src.get('content', src.get('snippet', ''))))
        print(f'Text: {text[:500]}...')
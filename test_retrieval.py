#!/usr/bin/env python3
"""
Test retrieval orchestrator
"""

from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval

print('Testing retrieval orchestrator...')
try:
    result = fetch_retrieval(
        query_text='Mục đích của Quy trình Quản lý Dịch vụ là gì?',
        embedder_type='ollama',
        reranker_type='none',
        use_query_enhancement=True,
        top_k=3
    )

    sources = result.get('sources', [])
    context = result.get('context', '')

    print(f'Retrieval successful: {len(sources)} sources')
    print(f'Context length: {len(context)}')

    if sources:
        print('First source:', sources[0])

except Exception as e:
    print(f'Retrieval failed: {e}')
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f'Retrieval failed: {e}')
    import traceback
    traceback.print_exc()
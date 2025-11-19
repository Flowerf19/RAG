#!/usr/bin/env python3
"""
Test semantic similarity calculation directly
"""

from evaluation.backend_dashboard.api import BackendDashboard

backend = BackendDashboard()

# Get first ground truth
rows = backend.get_ground_truth_list(limit=1)
if rows:
    row = rows[0]
    question = row.get('question', '')
    true_source = row.get('source', '').strip()

    print(f'Question: {question}')
    print(f'True source: {true_source[:200]}...')

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

    sources = result.get('sources', [])
    print(f'Retrieved {len(sources)} sources')

    if sources:
        # Get embedder
        embedder = backend._get_or_create_embedder('ollama')

        # Embed true source
        true_embedding = embedder.embed(true_source)
        print(f'True source embedding shape: {len(true_embedding)}')

        # Test first source
        src = sources[0]
        chunk_text = src.get('full_text', src.get('text', src.get('content', src.get('snippet', ''))))
        print(f'Chunk text: {chunk_text[:200]}...')

        chunk_embedding = embedder.embed(chunk_text)
        print(f'Chunk embedding shape: {len(chunk_embedding)}')

        # Calculate similarity
        import numpy as np
        similarity = np.dot(true_embedding, chunk_embedding) / (
            np.linalg.norm(true_embedding) * np.linalg.norm(chunk_embedding)
        )
        print(f'Similarity: {similarity:.4f}')
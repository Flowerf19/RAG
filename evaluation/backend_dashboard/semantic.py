"""Semantic similarity evaluation logic extracted from `api.py`."""
from typing import Dict, Any, List

def run_semantic_similarity(db, fetch_retrieval, embedder_getter, embedder_type: str = "ollama",
                            reranker_type: str = "none", use_query_enhancement: bool = True,
                            top_k: int = 10, api_tokens: dict | None = None, llm_model: str | None = None, limit: int | None = None) -> Dict[str, Any]:
    """Run semantic similarity evaluation using the provided DB and retrieval function.

    This function is intentionally stateless: it uses `db` to read ground-truth rows
    and relies on a passed `fetch_retrieval` callable for retrieval. It returns the
    same result structure previously returned by the `BackendDashboard` method.
    
    Args:
        db: Database instance
        fetch_retrieval: Function to fetch retrieval results
        embedder_getter: Function to get cached embedder instance
        embedder_type: Type of embedder to use
        reranker_type: Type of reranker
        use_query_enhancement: Whether to use query enhancement
        top_k: Number of top results to retrieve
        api_tokens: API tokens if needed
        llm_model: LLM model to use
        limit: Limit number of ground truth items
    """
    rows = db.get_ground_truth_list(limit=limit or 10000)
    processed = 0
    errors = 0
    errors_list: List[str] = []
    semantic_results: List[Dict[str, Any]] = []

    for r in rows:
        gt_id = r.get('id')
        question = r.get('question')
        true_source = r.get('source', '').strip()

        try:
            try:
                result = fetch_retrieval(
                    query_text=question,
                    top_k=top_k,
                    embedder_type=embedder_type,
                    reranker_type=reranker_type,
                    use_query_enhancement=use_query_enhancement,
                    llm_model=llm_model,
                    evaluate_response=False,  # Skip LLM evaluation for semantic similarity
                )

                result.get('context', '')
                sources = result.get('sources', [])
            except ModuleNotFoundError as mnfe:
                # Heavy dependency (transformers / embedder) missing in this environment.
                # Fall back to placeholder result so callers can still run smoke tests.
                errors += 1
                errors_list.append(f"ID {gt_id}: dependency error: {mnfe}")
                semantic_results.append({
                    'ground_truth_id': gt_id,
                    'question': question,
                    'true_source': true_source[:500] + '...' if len(true_source) > 500 else true_source,
                    'retrieved_chunks': 0,
                    'semantic_similarity': 0.0,
                    'best_match_score': 0.0,
                    'chunks_above_threshold': 0,
                    'matched_chunks': [],
                    'embedder_used': embedder_type,
                    'error': f"dependency error: {mnfe}"
                })
                # Skip further processing for this row
                continue
            except Exception as e:
                errors += 1
                errors_list.append(f"ID {gt_id}: retrieval error: {e}")
                semantic_results.append({
                    'ground_truth_id': gt_id,
                    'question': question,
                    'error': str(e),
                    'semantic_similarity': 0.0,
                    'best_match_score': 0.0
                })
                continue

            # Evaluate semantic similarity with true source
            # Use the same logic as evaluate_semantic_similarity_with_source method
            try:
                # Get ground truth entry with true source
                gt_rows = db.get_ground_truth_list(limit=10000)
                true_source = None
                
                for row in gt_rows:
                    if row.get('id') == gt_id:
                        true_source = row.get('source', '').strip()
                        break
                
                if not true_source:
                    semantic_results.append({
                        'ground_truth_id': gt_id,
                        'question': question,
                        'true_source': true_source[:500] + '...' if len(true_source) > 500 else true_source,
                        'retrieved_chunks': len(sources),
                        'semantic_similarity': 0.0,
                        'best_match_score': 0.0,
                        'chunks_above_threshold': 0,
                        'matched_chunks': [],
                        'embedder_used': embedder_type,
                        'error': f'No source found for ground truth ID {gt_id}'
                    })
                    processed += 1
                    continue
                
                # Get cached embedder instance
                embedder = embedder_getter(embedder_type)
                
                # Get embeddings for true source
                true_source_embedding = embedder.embed(true_source)
                
                # Calculate semantic similarity for each retrieved chunk
                similarities = []
                matched_chunks = []
                
                for i, source in enumerate(sources):
                    chunk_text = source.get('text', source.get('content', source.get('snippet', '')))
                    if chunk_text.strip():
                        try:
                            chunk_embedding = embedder.embed(chunk_text)
                            
                            # Calculate cosine similarity
                            import numpy as np
                            similarity = np.dot(true_source_embedding, chunk_embedding) / (
                                np.linalg.norm(true_source_embedding) * np.linalg.norm(chunk_embedding)
                            )
                            
                            similarities.append(float(similarity))
                            
                            # Store match info if similarity > 0.5
                            if similarity > 0.5:
                                matched_chunks.append({
                                    'chunk_index': i,
                                    'similarity_score': round(float(similarity), 4),
                                    'chunk_text': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text,
                                    'file_name': source.get('file_name', ''),
                                    'page_number': source.get('page_number', 0)
                                })
                                
                        except Exception:
                            similarities.append(0.0)
                
                # Calculate overall metrics
                if similarities:
                    best_match_score = max(similarities)
                    avg_similarity = sum(similarities) / len(similarities)
                    
                    # Sort matched chunks by similarity
                    matched_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
                    
                    semantic_results.append({
                        'ground_truth_id': gt_id,
                        'question': question,
                        'true_source': true_source[:500] + '...' if len(true_source) > 500 else true_source,
                        'retrieved_chunks': len(sources),
                        'semantic_similarity': round(avg_similarity, 4),
                        'best_match_score': round(best_match_score, 4),
                        'chunks_above_threshold': len([s for s in similarities if s > 0.5]),
                        'matched_chunks': matched_chunks[:5],  # Top 5 matches
                        'embedder_used': embedder_type,
                        'error': None
                    })
                else:
                    semantic_results.append({
                        'ground_truth_id': gt_id,
                        'question': question,
                        'true_source': true_source[:500] + '...' if len(true_source) > 500 else true_source,
                        'retrieved_chunks': len(sources),
                        'semantic_similarity': 0.0,
                        'best_match_score': 0.0,
                        'chunks_above_threshold': 0,
                        'matched_chunks': [],
                        'embedder_used': embedder_type,
                        'error': 'No chunks to evaluate'
                    })
                    
            except Exception as e:
                semantic_results.append({
                    'ground_truth_id': gt_id,
                    'question': question,
                    'true_source': true_source[:500] + '...' if len(true_source) > 500 else true_source,
                    'retrieved_chunks': len(sources),
                    'semantic_similarity': 0.0,
                    'best_match_score': 0.0,
                    'chunks_above_threshold': 0,
                    'matched_chunks': [],
                    'embedder_used': embedder_type,
                    'error': str(e)
                })

            processed += 1

        except Exception as e:
            errors += 1
            errors_list.append(f"ID {gt_id}: {str(e)}")
            semantic_results.append({
                'ground_truth_id': gt_id,
                'question': question,
                'error': str(e),
                'semantic_similarity': 0.0,
                'best_match_score': 0.0
            })

    # Compute summary metrics based on available (possibly placeholder) values
    valid_results = [r for r in semantic_results if not r.get('error')]
    if valid_results:
        avg_semantic_similarity = sum(r.get('semantic_similarity', 0.0) for r in valid_results) / len(valid_results)
        avg_best_match = sum(r.get('best_match_score', 0.0) for r in valid_results) / len(valid_results)
        total_chunks_above_threshold = sum(r.get('chunks_above_threshold', 0) for r in valid_results)
    else:
        avg_semantic_similarity = 0.0
        avg_best_match = 0.0
        total_chunks_above_threshold = 0

    result = {
        'summary': {
            'total_questions': len(rows),
            'processed': processed,
            'errors': errors,
            'avg_semantic_similarity': round(avg_semantic_similarity, 4),
            'avg_best_match_score': round(avg_best_match, 4),
            'total_chunks_above_threshold': total_chunks_above_threshold,
            'embedder_used': embedder_type,
            'reranker_used': reranker_type,
            'query_enhancement_used': use_query_enhancement
        },
        'results': semantic_results,
        'errors_list': errors_list
    }

    return result

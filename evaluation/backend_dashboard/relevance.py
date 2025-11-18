"""Relevance evaluation logic extracted from `api.py`."""
from typing import Dict, Any, List
import time

from evaluation.metrics.logger import EvaluationLogger


def run_relevance(db, fetch_retrieval, evaluate_semantic_similarity_with_source, embedder_type: str = "ollama",
                  reranker_type: str = "none", use_query_enhancement: bool = True,
                  top_k: int = 10, limit: int | None = None) -> Dict[str, Any]:
    rows = db.get_ground_truth_list(limit=limit or 10000)
    processed = 0
    errors = 0
    errors_list: List[str] = []
    relevance_results: List[Dict[str, Any]] = []

    all_relevance_scores: List[float] = []
    relevance_distribution = {
        '0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0,
        '0.6-0.8': 0, '0.8-1.0': 0
    }

    eval_logger = EvaluationLogger()

    for r in rows:
        gt_id = r.get('id')
        question = r.get('question')
        true_source = r.get('source', '').strip()

        try:
            # Time retrieval + semantic evaluation per ground-truth question
            start_time = time.time()

            result = fetch_retrieval(
                query_text=question,
                top_k=top_k,
                embedder_type=embedder_type,
                reranker_type=reranker_type,
                use_query_enhancement=use_query_enhancement,
            )

            sources = result.get('sources', [])

            semantic_eval = evaluate_semantic_similarity_with_source(
                retrieved_sources=sources,
                ground_truth_id=gt_id,
                embedder_type=embedder_type
            )

            chunk_relevance_scores = []
            high_relevance_count = 0

            for chunk in semantic_eval.get('matched_chunks', []):
                score = chunk.get('similarity_score', 0.0)
                chunk_relevance_scores.append(score)
                if score > 0.8:
                    high_relevance_count += 1

                if score <= 0.2:
                    relevance_distribution['0-0.2'] += 1
                elif score <= 0.4:
                    relevance_distribution['0.2-0.4'] += 1
                elif score <= 0.6:
                    relevance_distribution['0.4-0.6'] += 1
                elif score <= 0.8:
                    relevance_distribution['0.6-0.8'] += 1
                else:
                    relevance_distribution['0.8-1.0'] += 1

            if chunk_relevance_scores:
                avg_chunk_relevance = sum(chunk_relevance_scores) / len(chunk_relevance_scores)
                max_chunk_relevance = max(chunk_relevance_scores)
                min_chunk_relevance = min(chunk_relevance_scores)
                relevant_chunks_ratio = sum(1 for s in chunk_relevance_scores if s > 0.5) / len(chunk_relevance_scores)
            else:
                avg_chunk_relevance = 0.0
                max_chunk_relevance = 0.0
                min_chunk_relevance = 0.0
                relevant_chunks_ratio = 0.0

            semantic_similarity = semantic_eval.get('semantic_similarity', 0.0)
            overall_relevance = (semantic_similarity + avg_chunk_relevance) / 2

            all_relevance_scores.extend(chunk_relevance_scores)

            relevance_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'true_source': true_source[:200] + '...' if len(true_source) > 200 else true_source,
                'retrieved_chunks': len(sources),
                'semantic_similarity': semantic_similarity,
                'avg_chunk_relevance': round(avg_chunk_relevance, 4),
                'max_chunk_relevance': round(max_chunk_relevance, 4),
                'min_chunk_relevance': round(min_chunk_relevance, 4),
                'overall_relevance': round(overall_relevance, 4),
                'relevant_chunks_ratio': round(relevant_chunks_ratio, 4),
                'high_relevance_chunks': high_relevance_count,
                'chunk_relevance_scores': [round(s, 4) for s in chunk_relevance_scores],
                'error': semantic_eval.get('error')
            })

            processed += 1

            # Log per-question relevance into metrics DB so dashboard can display it
            try:
                latency = time.time() - start_time
                eval_logger.log_evaluation(
                    query=question,
                    model=f"{embedder_type}_{reranker_type}",
                    latency=latency,
                    faithfulness=None,
                    relevance=overall_relevance,
                    recall=None,
                    error=False,
                    embedder_model=embedder_type,
                    llm_model=None,
                    reranker_model=reranker_type,
                    embedder_specific_model=None,
                    llm_specific_model=None,
                    reranker_specific_model=None,
                    query_enhanced=use_query_enhancement,
                    embedding_tokens=0,
                    reranking_tokens=0,
                    llm_tokens=0,
                    total_tokens=0,
                    retrieval_chunks=len(result.get('sources', [])),
                    metadata={'ground_truth_id': gt_id, 'evaluation_type': 'relevance'}
                )
            except Exception:
                # Don't let logging failures break the evaluation
                pass

        except Exception as e:
            errors += 1
            errors_list.append(f"ID {gt_id}: {str(e)}")
            relevance_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'error': str(e),
                'overall_relevance': 0.0,
                'avg_chunk_relevance': 0.0
            })

    if all_relevance_scores:
        global_avg_relevance = sum(all_relevance_scores) / len(all_relevance_scores)
        global_high_relevance_ratio = sum(1 for s in all_relevance_scores if s > 0.8) / len(all_relevance_scores)
        global_relevant_ratio = sum(1 for s in all_relevance_scores if s > 0.5) / len(all_relevance_scores)
    else:
        global_avg_relevance = 0.0
        global_high_relevance_ratio = 0.0
        global_relevant_ratio = 0.0

    valid_results = [r for r in relevance_results if not r.get('error')]
    if valid_results:
        avg_overall_relevance = sum(r['overall_relevance'] for r in valid_results) / len(valid_results)
        avg_chunk_relevance = sum(r['avg_chunk_relevance'] for r in valid_results) / len(valid_results)
        avg_semantic_similarity = sum(r['semantic_similarity'] for r in valid_results) / len(valid_results)
    else:
        avg_overall_relevance = 0.0
        avg_chunk_relevance = 0.0
        avg_semantic_similarity = 0.0

    result = {
        'summary': {
            'total_questions': len(rows),
            'processed': processed,
            'errors': errors,
            'avg_overall_relevance': round(avg_overall_relevance, 4),
            'avg_chunk_relevance': round(avg_chunk_relevance, 4),
            'avg_semantic_similarity': round(avg_semantic_similarity, 4),
            'global_avg_relevance': round(global_avg_relevance, 4),
            'global_high_relevance_ratio': round(global_high_relevance_ratio, 4),
            'global_relevant_ratio': round(global_relevant_ratio, 4),
            'total_chunks_evaluated': len(all_relevance_scores),
            'relevance_distribution': relevance_distribution,
            'embedder_used': embedder_type,
            'reranker_used': reranker_type,
            'query_enhancement_used': use_query_enhancement
        },
        'results': relevance_results,
        'errors_list': errors_list
    }

    return result

def run_relevance_batch(db, retrieval_results: Dict[int, Dict], ground_truth_rows: List[Dict], 
                       evaluate_semantic_similarity_with_source, embedder_type: str = "huggingface_local") -> Dict[str, Any]:
    """Run relevance evaluation using pre-fetched retrieval results."""
    processed = 0
    errors = 0
    errors_list: List[str] = []
    relevance_results: List[Dict[str, Any]] = []

    all_relevance_scores: List[float] = []
    relevance_distribution = {
        '0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0,
        '0.6-0.8': 0, '0.8-1.0': 0
    }

    eval_logger = EvaluationLogger()

    for r in ground_truth_rows:
        gt_id = r.get('id')
        question = r.get('question')
        true_source = r.get('source', '').strip()

        try:
            # Time retrieval + semantic evaluation per ground-truth question
            start_time = time.time()

            # Get cached retrieval result
            result = retrieval_results.get(gt_id, {})
            if result.get('error'):
                errors += 1
                errors_list.append(f"ID {gt_id}: {result['error']}")
                relevance_results.append({
                    'ground_truth_id': gt_id,
                    'question': question[:100] + '...' if len(question) > 100 else question,
                    'error': result['error'],
                    'overall_relevance': 0.0,
                    'avg_chunk_relevance': 0.0
                })
                continue

            sources = result.get('sources', [])

            semantic_eval = evaluate_semantic_similarity_with_source(
                retrieved_sources=sources,
                ground_truth_id=gt_id,
                embedder_type=embedder_type
            )

            chunk_relevance_scores = []
            high_relevance_count = 0

            for chunk in semantic_eval.get('matched_chunks', []):
                score = chunk.get('similarity_score', 0.0)
                chunk_relevance_scores.append(score)
                if score > 0.8:
                    high_relevance_count += 1

                if score <= 0.2:
                    relevance_distribution['0-0.2'] += 1
                elif score <= 0.4:
                    relevance_distribution['0.2-0.4'] += 1
                elif score <= 0.6:
                    relevance_distribution['0.4-0.6'] += 1
                elif score <= 0.8:
                    relevance_distribution['0.6-0.8'] += 1
                else:
                    relevance_distribution['0.8-1.0'] += 1

            if chunk_relevance_scores:
                avg_chunk_relevance = sum(chunk_relevance_scores) / len(chunk_relevance_scores)
                max_chunk_relevance = max(chunk_relevance_scores)
                min_chunk_relevance = min(chunk_relevance_scores)
                relevant_chunks_ratio = sum(1 for s in chunk_relevance_scores if s > 0.5) / len(chunk_relevance_scores)
            else:
                avg_chunk_relevance = 0.0
                max_chunk_relevance = 0.0
                min_chunk_relevance = 0.0
                relevant_chunks_ratio = 0.0

            semantic_similarity = semantic_eval.get('semantic_similarity', 0.0)
            overall_relevance = (semantic_similarity + avg_chunk_relevance) / 2

            all_relevance_scores.extend(chunk_relevance_scores)

            relevance_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'true_source': true_source[:200] + '...' if len(true_source) > 200 else true_source,
                'retrieved_chunks': len(sources),
                'semantic_similarity': semantic_similarity,
                'avg_chunk_relevance': round(avg_chunk_relevance, 4),
                'max_chunk_relevance': round(max_chunk_relevance, 4),
                'min_chunk_relevance': round(min_chunk_relevance, 4),
                'overall_relevance': round(overall_relevance, 4),
                'relevant_chunks_ratio': round(relevant_chunks_ratio, 4),
                'high_relevance_chunks': high_relevance_count,
                'chunk_relevance_scores': [round(s, 4) for s in chunk_relevance_scores],
                'error': semantic_eval.get('error')
            })

            processed += 1

            # Log per-question relevance into metrics DB so dashboard can display it
            try:
                latency = time.time() - start_time
                eval_logger.log_evaluation(
                    query=question,
                    model=f"batch_{embedder_type}",
                    latency=latency,
                    faithfulness=None,
                    relevance=overall_relevance,
                    recall=None,
                    error=False,
                    embedder_model=embedder_type,
                    llm_model=None,
                    reranker_model='batch',
                    embedder_specific_model=None,
                    llm_specific_model=None,
                    reranker_specific_model=None,
                    query_enhanced=True,
                    embedding_tokens=0,
                    reranking_tokens=0,
                    llm_tokens=0,
                    total_tokens=0,
                    retrieval_chunks=len(result.get('sources', [])),
                    metadata={'ground_truth_id': gt_id, 'evaluation_type': 'relevance_batch'}
                )
            except Exception:
                # Don't let logging failures break the evaluation
                pass

        except Exception as e:
            errors += 1
            errors_list.append(f"ID {gt_id}: {str(e)}")
            relevance_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'error': str(e),
                'overall_relevance': 0.0,
                'avg_chunk_relevance': 0.0
            })

    if all_relevance_scores:
        global_avg_relevance = sum(all_relevance_scores) / len(all_relevance_scores)
        global_high_relevance_ratio = sum(1 for s in all_relevance_scores if s > 0.8) / len(all_relevance_scores)
        global_relevant_ratio = sum(1 for s in all_relevance_scores if s > 0.5) / len(all_relevance_scores)
    else:
        global_avg_relevance = 0.0
        global_high_relevance_ratio = 0.0
        global_relevant_ratio = 0.0

    valid_results = [r for r in relevance_results if not r.get('error')]
    if valid_results:
        avg_overall_relevance = sum(r['overall_relevance'] for r in valid_results) / len(valid_results)
        avg_chunk_relevance = sum(r['avg_chunk_relevance'] for r in valid_results) / len(valid_results)
        avg_semantic_similarity = sum(r['semantic_similarity'] for r in valid_results) / len(valid_results)
    else:
        avg_overall_relevance = 0.0
        avg_chunk_relevance = 0.0
        avg_semantic_similarity = 0.0

    result = {
        'summary': {
            'total_questions': len(ground_truth_rows),
            'processed': processed,
            'errors': errors,
            'avg_overall_relevance': round(avg_overall_relevance, 4),
            'avg_chunk_relevance': round(avg_chunk_relevance, 4),
            'avg_semantic_similarity': round(avg_semantic_similarity, 4),
            'global_avg_relevance': round(global_avg_relevance, 4),
            'global_high_relevance_ratio': round(global_high_relevance_ratio, 4),
            'global_relevant_ratio': round(global_relevant_ratio, 4),
            'total_chunks_evaluated': len(all_relevance_scores),
            'relevance_distribution': relevance_distribution,
            'embedder_used': embedder_type,
            'reranker_used': 'batch',
            'query_enhancement_used': True
        },
        'results': relevance_results,
        'errors_list': errors_list
    }

    return result

"""Recall evaluation logic extracted from `api.py`."""
from typing import Dict, Any, List
import time

from evaluation.metrics.logger import EvaluationLogger


def run_recall(db, fetch_retrieval, evaluate_semantic_similarity_with_source, embedder_type: str = "ollama",
               reranker_type: str = "none", use_query_enhancement: bool = True,
               top_k: int = 10, similarity_threshold: float = 0.5, limit: int | None = None) -> Dict[str, Any]:
    rows = db.get_ground_truth_list(limit=limit or 10000)
    processed = 0
    errors = 0
    errors_list: List[str] = []
    recall_results: List[Dict[str, Any]] = []

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    eval_logger = EvaluationLogger()

    for r in rows:
        gt_id = r.get('id')
        question = r.get('question')
        true_source = r.get('source', '').strip()

        try:
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

            retrieved_chunks = len(sources)
            chunks_above_threshold = semantic_eval.get('chunks_above_threshold', 0)

            true_positives = chunks_above_threshold
            false_positives = retrieved_chunks - true_positives
            expected_relevant = 1
            false_negatives = max(0, expected_relevant - true_positives)

            if (true_positives + false_negatives) > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.0

            if retrieved_chunks > 0:
                precision = true_positives / retrieved_chunks
            else:
                precision = 0.0

            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

            recall_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'true_source': true_source[:200] + '...' if len(true_source) > 200 else true_source,
                'retrieved_chunks': retrieved_chunks,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'recall': round(recall, 4),
                'precision': round(precision, 4),
                'f1_score': round(f1_score, 4),
                'semantic_similarity': semantic_eval.get('semantic_similarity', 0.0),
                'best_match_score': semantic_eval.get('best_match_score', 0.0),
                'error': semantic_eval.get('error')
            })

            processed += 1

            # Log per-question recall metrics to metrics DB
            try:
                latency = time.time() - start_time
                eval_logger.log_evaluation(
                    query=question,
                    model=f"{embedder_type}_{reranker_type}",
                    latency=latency,
                    faithfulness=None,
                    relevance=None,
                    recall=recall,
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
                    retrieval_chunks=retrieved_chunks,
                    metadata={'ground_truth_id': gt_id, 'evaluation_type': 'recall'}
                )
            except Exception:
                pass

        except Exception as e:
            errors += 1
            errors_list.append(f"ID {gt_id}: {str(e)}")
            recall_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'error': str(e),
                'recall': 0.0,
                'precision': 0.0,
                'f1_score': 0.0
            })

    total_retrieved = total_true_positives + total_false_positives

    if (total_true_positives + total_false_negatives) > 0:
        overall_recall = total_true_positives / (total_true_positives + total_false_negatives)
    else:
        overall_recall = 0.0

    if total_retrieved > 0:
        overall_precision = total_true_positives / total_retrieved
    else:
        overall_precision = 0.0

    if (overall_precision + overall_recall) > 0:
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    else:
        overall_f1 = 0.0

    valid_results = [r for r in recall_results if not r.get('error')]
    if valid_results:
        avg_recall = sum(r['recall'] for r in valid_results) / len(valid_results)
        avg_precision = sum(r['precision'] for r in valid_results) / len(valid_results)
        avg_f1 = sum(r['f1_score'] for r in valid_results) / len(valid_results)
    else:
        avg_recall = 0.0
        avg_precision = 0.0
        avg_f1 = 0.0

    result = {
        'summary': {
            'total_questions': len(rows),
            'processed': processed,
            'errors': errors,
            'overall_recall': round(overall_recall, 4),
            'overall_precision': round(overall_precision, 4),
            'overall_f1_score': round(overall_f1, 4),
            'avg_recall': round(avg_recall, 4),
            'avg_precision': round(avg_precision, 4),
            'avg_f1_score': round(avg_f1, 4),
            'total_true_positives': total_true_positives,
            'total_false_positives': total_false_positives,
            'total_false_negatives': total_false_negatives,
            'total_retrieved_chunks': total_retrieved,
            'embedder_used': embedder_type,
            'reranker_used': reranker_type,
            'query_enhancement_used': use_query_enhancement,
            'similarity_threshold': similarity_threshold
        },
        'results': recall_results,
        'errors_list': errors_list
    }

    return result

def run_recall_batch(db, retrieval_results: Dict[int, Dict], ground_truth_rows: List[Dict], 
                    evaluate_semantic_similarity_with_source, embedder_type: str = "huggingface_local") -> Dict[str, Any]:
    """Run recall evaluation using pre-fetched retrieval results."""
    processed = 0
    errors = 0
    errors_list: List[str] = []
    recall_results: List[Dict[str, Any]] = []

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    eval_logger = EvaluationLogger()

    for r in ground_truth_rows:
        gt_id = r.get('id')
        question = r.get('question')
        true_source = r.get('source', '').strip()

        try:
            start_time = time.time()

            # Get cached retrieval result
            result = retrieval_results.get(gt_id, {})
            if result.get('error'):
                errors += 1
                errors_list.append(f"ID {gt_id}: {result['error']}")
                recall_results.append({
                    'ground_truth_id': gt_id,
                    'question': question[:100] + '...' if len(question) > 100 else question,
                    'error': result['error'],
                    'recall': 0.0,
                    'precision': 0.0,
                    'f1_score': 0.0
                })
                continue

            sources = result.get('sources', [])

            semantic_eval = evaluate_semantic_similarity_with_source(
                retrieved_sources=sources,
                ground_truth_id=gt_id,
                embedder_type=embedder_type
            )

            retrieved_chunks = len(sources)
            chunks_above_threshold = semantic_eval.get('chunks_above_threshold', 0)

            true_positives = chunks_above_threshold
            false_positives = retrieved_chunks - true_positives
            expected_relevant = 1
            false_negatives = max(0, expected_relevant - true_positives)

            if (true_positives + false_negatives) > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.0

            if retrieved_chunks > 0:
                precision = true_positives / retrieved_chunks
            else:
                precision = 0.0

            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

            recall_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'true_source': true_source[:200] + '...' if len(true_source) > 200 else true_source,
                'retrieved_chunks': retrieved_chunks,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'recall': round(recall, 4),
                'precision': round(precision, 4),
                'f1_score': round(f1_score, 4),
                'semantic_similarity': semantic_eval.get('semantic_similarity', 0.0),
                'best_match_score': semantic_eval.get('best_match_score', 0.0),
                'error': semantic_eval.get('error')
            })

            processed += 1

            # Log per-question recall metrics to metrics DB
            try:
                latency = time.time() - start_time
                eval_logger.log_evaluation(
                    query=question,
                    model=f"batch_{embedder_type}",
                    latency=latency,
                    faithfulness=None,
                    relevance=None,
                    recall=recall,
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
                    retrieval_chunks=retrieved_chunks,
                    metadata={'ground_truth_id': gt_id, 'evaluation_type': 'recall_batch'}
                )
            except Exception:
                pass

        except Exception as e:
            errors += 1
            errors_list.append(f"ID {gt_id}: {str(e)}")
            recall_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'error': str(e),
                'recall': 0.0,
                'precision': 0.0,
                'f1_score': 0.0
            })

    total_retrieved = total_true_positives + total_false_positives

    if (total_true_positives + total_false_negatives) > 0:
        overall_recall = total_true_positives / (total_true_positives + total_false_negatives)
    else:
        overall_recall = 0.0

    if total_retrieved > 0:
        overall_precision = total_true_positives / total_retrieved
    else:
        overall_precision = 0.0

    if (overall_precision + overall_recall) > 0:
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    else:
        overall_f1 = 0.0

    valid_results = [r for r in recall_results if not r.get('error')]
    if valid_results:
        avg_recall = sum(r['recall'] for r in valid_results) / len(valid_results)
        avg_precision = sum(r['precision'] for r in valid_results) / len(valid_results)
        avg_f1 = sum(r['f1_score'] for r in valid_results) / len(valid_results)
    else:
        avg_recall = 0.0
        avg_precision = 0.0
        avg_f1 = 0.0

    result = {
        'summary': {
            'total_questions': len(ground_truth_rows),
            'processed': processed,
            'errors': errors,
            'overall_recall': round(overall_recall, 4),
            'overall_precision': round(overall_precision, 4),
            'overall_f1_score': round(overall_f1, 4),
            'avg_recall': round(avg_recall, 4),
            'avg_precision': round(avg_precision, 4),
            'avg_f1_score': round(avg_f1, 4),
            'total_true_positives': total_true_positives,
            'total_false_positives': total_false_positives,
            'total_false_negatives': total_false_negatives,
            'total_retrieved_chunks': total_retrieved,
            'embedder_used': embedder_type,
            'reranker_used': 'batch',
            'query_enhancement_used': True,
            'similarity_threshold': 0.5
        },
        'results': recall_results,
        'errors_list': errors_list
    }

    return result

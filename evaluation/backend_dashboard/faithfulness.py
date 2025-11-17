"""Faithfulness evaluation logic extracted from `api.py`."""
from typing import Dict, Any, List

def run_faithfulness(db, fetch_retrieval, llm_client_factory, AutoEvaluatorClass, embedder_type: str = "ollama",
                     reranker_type: str = "none", llm_choice: str = "gemini",
                     use_query_enhancement: bool = True, top_k: int = 10, limit: int | None = None) -> Dict[str, Any]:
    rows = db.get_ground_truth_list(limit=limit or 10000)
    processed = 0
    errors = 0
    errors_list: List[str] = []
    faithfulness_results: List[Dict[str, Any]] = []

    all_faithfulness_scores: List[float] = []
    faithfulness_distribution = {
        '0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0,
        '0.6-0.8': 0, '0.8-1.0': 0
    }

    llm_client = None
    if llm_choice != "none":
        try:
            llm_client = llm_client_factory.create_from_string(llm_choice)
        except Exception:
            llm_client = None

    evaluator = AutoEvaluatorClass(llm_client=llm_client)

    for r in rows:
        gt_id = r.get('id')
        question = r.get('question')
        true_answer = r.get('answer', '').strip()

        try:
            result = fetch_retrieval(
                query_text=question,
                top_k=top_k,
                embedder_type=embedder_type,
                reranker_type=reranker_type,
                use_query_enhancement=use_query_enhancement,
            )

            context = result.get('context', '')

            generated_answer = ""
            if llm_client and context:
                try:
                    from llm.chat_handler import build_messages
                    messages = build_messages(
                        query=question,
                        context=context,
                        history=[]
                    )
                    generated_answer = llm_client.generate(messages, max_tokens=512)
                except Exception:
                    generated_answer = context[:1000]

            if generated_answer and context:
                faithfulness_score = evaluator._evaluate_faithfulness(generated_answer, context)
            else:
                faithfulness_score = 0.0

            if faithfulness_score <= 0.2:
                faithfulness_distribution['0-0.2'] += 1
            elif faithfulness_score <= 0.4:
                faithfulness_distribution['0.2-0.4'] += 1
            elif faithfulness_score <= 0.6:
                faithfulness_distribution['0.4-0.6'] += 1
            elif faithfulness_score <= 0.8:
                faithfulness_distribution['0.6-0.8'] += 1
            else:
                faithfulness_distribution['0.8-1.0'] += 1

            all_faithfulness_scores.append(faithfulness_score)

            faithfulness_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'true_answer': true_answer[:200] + '...' if len(true_answer) > 200 else true_answer,
                'generated_answer': generated_answer[:200] + '...' if len(generated_answer) > 200 else generated_answer,
                'context_length': len(context),
                'faithfulness_score': round(faithfulness_score, 4),
                'error': None
            })

            processed += 1

        except Exception as e:
            errors += 1
            errors_list.append(f"ID {gt_id}: {str(e)}")
            faithfulness_results.append({
                'ground_truth_id': gt_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'faithfulness_score': 0.0,
                'error': str(e)
            })

    if all_faithfulness_scores:
        global_avg_faithfulness = sum(all_faithfulness_scores) / len(all_faithfulness_scores)
        global_high_faithfulness_ratio = sum(1 for s in all_faithfulness_scores if s > 0.8) / len(all_faithfulness_scores)
        global_faithful_ratio = sum(1 for s in all_faithfulness_scores if s > 0.5) / len(all_faithfulness_scores)
    else:
        global_avg_faithfulness = 0.0
        global_high_faithfulness_ratio = 0.0
        global_faithful_ratio = 0.0

    valid_results = [r for r in faithfulness_results if not r.get('error')]
    if valid_results:
        avg_faithfulness = sum(r['faithfulness_score'] for r in valid_results) / len(valid_results)
    else:
        avg_faithfulness = 0.0

    result = {
        'summary': {
            'total_questions': len(rows),
            'processed': processed,
            'errors': errors,
            'avg_faithfulness': round(avg_faithfulness, 4),
            'global_avg_faithfulness': round(global_avg_faithfulness, 4),
            'global_high_faithfulness_ratio': round(global_high_faithfulness_ratio, 4),
            'global_faithful_ratio': round(global_faithful_ratio, 4),
            'total_answers_evaluated': len(all_faithfulness_scores),
            'faithfulness_distribution': faithfulness_distribution,
            'embedder_used': embedder_type,
            'reranker_used': reranker_type,
            'llm_used': llm_choice,
            'query_enhancement_used': use_query_enhancement
        },
        'results': faithfulness_results,
        'errors_list': errors_list
    }

    return result

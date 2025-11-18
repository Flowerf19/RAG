"""Faithfulness evaluation logic extracted from `api.py`."""
from typing import Dict, Any, List
import time
import logging
import os
from pathlib import Path

from evaluation.metrics.logger import EvaluationLogger
from evaluation.metrics.token_counter import token_counter

def _load_secrets_for_evaluation():
    """
    Load secrets from .streamlit/secret.toml and set environment variables
    for non-Streamlit evaluation contexts.
    """
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            logging.warning("tomllib/tomli not available, cannot load secrets")
            return

    # Find secrets file by searching up from current working directory
    current = Path.cwd()
    secrets_file = None

    # Search up to 5 levels up
    for _ in range(5):
        candidate = current / ".streamlit" / "secret.toml"
        if candidate.exists():
            secrets_file = candidate
            break
        if current.parent == current:
            break
        current = current.parent

    if not secrets_file:
        logging.warning("Secrets file not found in current directory tree")
        return

    try:
        with open(secrets_file, "rb") as f:
            secrets = tomllib.load(f)

        # Set environment variables for API keys
        if "gemini_api_key" in secrets:
            os.environ["GOOGLE_API_KEY"] = secrets["gemini_api_key"]
        if "HF_TOKEN" in secrets:
            os.environ["HF_TOKEN"] = secrets["HF_TOKEN"]

        logging.info("Successfully loaded secrets for evaluation")

    except Exception as e:
        logging.error(f"Failed to load secrets: {e}")

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
            # Load secrets for non-Streamlit contexts
            _load_secrets_for_evaluation()
            llm_client = llm_client_factory.create_from_string(llm_choice)
        except Exception:
            llm_client = None

    evaluator = AutoEvaluatorClass(llm_client=llm_client)

    eval_logger = EvaluationLogger()

    for r in rows:
        gt_id = r.get('id')
        question = r.get('question')
        true_answer = r.get('answer', '').strip()

        try:
            # Time this single GT retrieval+generation run
            start_time = time.time()

            result = fetch_retrieval(
                query_text=question,
                top_k=top_k,
                embedder_type=embedder_type,
                reranker_type=reranker_type,
                use_query_enhancement=use_query_enhancement,
                llm_model=llm_choice,
            )

            context = result.get('context', '')
            sources = result.get('sources', [])

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

            # Compute scores
            if generated_answer and context:
                faithfulness_score = evaluator._evaluate_faithfulness(generated_answer, context)
                relevance_score = evaluator._evaluate_relevance(generated_answer, question)
            else:
                faithfulness_score = 0.0
                relevance_score = 0.0

            # Approximate recall: if ground-truth 'source' exists, check substring matches
            recall_score = None
            try:
                true_source = r.get('source', '') or ''
                if true_source and sources:
                    matched = 0
                    total = len(sources)
                    for s in sources:
                        text = ''
                        if isinstance(s, dict):
                            text = (s.get('text') or s.get('content') or s.get('snippet') or '')
                        else:
                            text = str(s)
                        if not text:
                            continue
                        # crude substring check
                        if true_source.strip() and (true_source.strip() in text or text in true_source.strip()):
                            matched += 1
                    recall_score = matched / total if total > 0 else 0.0
                else:
                    recall_score = None
            except Exception:
                recall_score = None

            latency = time.time() - start_time

            # Estimate LLM tokens used for generated answer (best-effort)
            try:
                llm_tokens = token_counter.count_tokens(generated_answer, llm_choice) if generated_answer else 0
            except Exception:
                llm_tokens = 0

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
                'relevance_score': round(relevance_score, 4) if isinstance(relevance_score, (int, float)) else None,
                'recall_score': round(recall_score, 4) if isinstance(recall_score, (int, float)) else None,
                'error': None
            })

            # Log per-question metric so metrics table gets populated with scores
            try:
                eval_logger.log_evaluation(
                    query=question,
                    model=llm_choice or f"{embedder_type}_{reranker_type}",
                    latency=latency,
                    faithfulness=faithfulness_score,
                    relevance=relevance_score,
                    recall=recall_score,
                    error=False,
                    embedder_model=embedder_type,
                    llm_model=llm_choice,
                    reranker_model=reranker_type,
                    embedder_specific_model=None,
                    llm_specific_model=llm_choice,
                    reranker_specific_model=None,
                    query_enhanced=use_query_enhancement,
                    embedding_tokens=0,
                    reranking_tokens=0,
                    llm_tokens=llm_tokens,
                    total_tokens=llm_tokens,
                    retrieval_chunks=len(sources) if isinstance(sources, list) else 0,
                    metadata={'ground_truth_id': gt_id, 'evaluation_type': 'faithfulness'}
                )
            except Exception:
                pass

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

def run_faithfulness_batch(db, retrieval_results: Dict[int, Dict], ground_truth_rows: List[Dict], 
                          embedder_type: str = "huggingface_local", reranker_type: str = "none", 
                          llm_choice: str = "gemini", use_query_enhancement: bool = True, top_k: int = 10) -> Dict[str, Any]:
    """Run faithfulness evaluation using pre-fetched retrieval results."""
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
            # Load secrets for non-Streamlit contexts
            _load_secrets_for_evaluation()
            from llm.client_factory import LLMClientFactory
            llm_client = LLMClientFactory().create_from_string(llm_choice)
        except Exception:
            llm_client = None

    from evaluation.evaluators.auto_evaluator import AutoEvaluator
    evaluator = AutoEvaluator(llm_client=llm_client)

    eval_logger = EvaluationLogger()

    for r in ground_truth_rows:
        gt_id = r.get('id')
        question = r.get('question')
        true_answer = r.get('answer', '').strip()

        try:
            # Time this single GT retrieval+generation run
            start_time = time.time()

            # Get cached retrieval result
            result = retrieval_results.get(gt_id, {})
            if result.get('error'):
                errors += 1
                errors_list.append(f"ID {gt_id}: {result['error']}")
                faithfulness_results.append({
                    'ground_truth_id': gt_id,
                    'question': question[:100] + '...' if len(question) > 100 else question,
                    'faithfulness_score': 0.0,
                    'error': result['error']
                })
                continue

            context = result.get('context', '')
            sources = result.get('sources', [])

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

            # Compute scores
            if generated_answer and context:
                faithfulness_score = evaluator._evaluate_faithfulness(generated_answer, context)
                relevance_score = evaluator._evaluate_relevance(generated_answer, question)
            else:
                faithfulness_score = 0.0
                relevance_score = 0.0

            # Approximate recall: if ground-truth 'source' exists, check substring matches
            recall_score = None
            try:
                true_source = r.get('source', '') or ''
                if true_source and sources:
                    matched = 0
                    total = len(sources)
                    for s in sources:
                        text = ''
                        if isinstance(s, dict):
                            text = (s.get('text') or s.get('content') or s.get('snippet') or '')
                        else:
                            text = str(s)
                        if not text:
                            continue
                        # crude substring check
                        if true_source.strip() and (true_source.strip() in text or text in true_source.strip()):
                            matched += 1
                    recall_score = matched / total if total > 0 else 0.0
                else:
                    recall_score = None
            except Exception:
                recall_score = None

            latency = time.time() - start_time

            # Estimate LLM tokens used for generated answer (best-effort)
            try:
                llm_tokens = token_counter.count_tokens(generated_answer, llm_choice) if generated_answer else 0
            except Exception:
                llm_tokens = 0

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
                'relevance_score': round(relevance_score, 4) if isinstance(relevance_score, (int, float)) else None,
                'recall_score': round(recall_score, 4) if isinstance(recall_score, (int, float)) else None,
                'error': None
            })

            # Log per-question metric so metrics table gets populated with scores
            try:
                eval_logger.log_evaluation(
                    query=question,
                    model=llm_choice or f"batch_{embedder_type}_{reranker_type}",
                    latency=latency,
                    faithfulness=faithfulness_score,
                    relevance=relevance_score,
                    recall=recall_score,
                    error=False,
                    embedder_model=embedder_type,
                    llm_model=llm_choice,
                    reranker_model=reranker_type,
                    embedder_specific_model=None,
                    llm_specific_model=llm_choice,
                    reranker_specific_model=None,
                    query_enhanced=use_query_enhancement,
                    embedding_tokens=0,
                    reranking_tokens=0,
                    llm_tokens=llm_tokens,
                    total_tokens=llm_tokens,
                    retrieval_chunks=len(sources) if isinstance(sources, list) else 0,
                    metadata={'ground_truth_id': gt_id, 'evaluation_type': 'faithfulness_batch'}
                )
            except Exception:
                pass

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
            'total_questions': len(ground_truth_rows),
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

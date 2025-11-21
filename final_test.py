from evaluation.backend_dashboard.api import BackendDashboard

bd = BackendDashboard()
result = bd.evaluate_ground_truth_with_ragas(llm_provider='ollama', limit=2, save_to_db=True)
print('Evaluation with database save completed')
print(f'Total samples: {result.get("total_samples", 0)}')
print(f'Context Recall Mean: {result.get("context_recall", {}).get("mean", "N/A")}')
print(f'Answer Relevancy Mean: {result.get("answer_relevancy", {}).get("mean", "N/A")}')
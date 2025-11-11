from evaluation.metrics.database import MetricsDB

db = MetricsDB()
print('Recent metrics:')
for m in db.get_metrics(limit=5):
    print(f'{m["timestamp"]} {m["model"]} | Embedder: {m.get("embedder_model", "N/A")} | LLM: {m.get("llm_model", "N/A")} | Reranker: {m.get("reranker_model", "N/A")} | QE: {m.get("query_enhanced", False)} | {m["query"][:20]}... {m["latency"]:.2f}s {m["faithfulness"]} {m["relevance"]} {m["error"]}')
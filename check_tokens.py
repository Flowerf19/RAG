from evaluation.metrics.database import MetricsDB

db = MetricsDB()
results = db.get_metrics(limit=5)
print('Recent metrics:')
for r in results:
    print(f'{r["timestamp"]}: {r.get("total_tokens", 0)} tokens, embedding: {r.get("embedding_tokens", 0)}, llm: {r.get("llm_tokens", 0)}, reranking: {r.get("reranking_tokens", 0)}')

# Check token usage stats
token_stats = db.get_token_usage_stats()
print(f'\nToken usage overview: {token_stats}')
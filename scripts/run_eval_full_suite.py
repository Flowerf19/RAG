import os
import sys
import sqlite3
import json

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.backend_dashboard.api import BackendDashboard

print('Starting full-suite eval (relevance + recall + faithfulness)')
backend = BackendDashboard()

embedder = 'huggingface_local'
reranker = 'none'
llm_choice = 'gemini'

print('\n== Relevance ==')
rel = backend.evaluate_relevance(embedder_type=embedder, reranker_type=reranker, limit=1, save_to_db=True)
print('Summary:', rel.get('summary'))

print('\n== Recall ==')
rec = backend.evaluate_recall(embedder_type=embedder, reranker_type=reranker, limit=1, save_to_db=True)
print('Summary:', rec.get('summary'))

print('\n== Faithfulness ==')
faith = backend.evaluate_faithfulness(embedder_type=embedder, reranker_type=reranker, llm_choice=llm_choice, limit=1, save_to_db=True)
print('Summary:', faith.get('summary'))

print('\n== Latest metrics rows ==')
conn = sqlite3.connect('data/metrics.db')
cur = conn.cursor()
cur.execute("SELECT timestamp, query, faithfulness, relevance, recall, metadata FROM metrics ORDER BY timestamp DESC LIMIT 10")
rows = cur.fetchall()
for r in rows:
    ts, query, faith, rel, rec, meta = r
    try:
        meta_parsed = json.loads(meta) if isinstance(meta, str) else meta
    except Exception:
        meta_parsed = meta
    print('\n---')
    print('Time:', ts)
    print('Query:', (query or '')[:200])
    print('Faithfulness:', faith, 'Relevance:', rel, 'Recall:', rec)
    print('Metadata:', meta_parsed)

conn.close()
print('\nDone.')

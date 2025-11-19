## Demo RAG Configuration Comparison - Metrics Table

| Configuration                 |   Faithfulness |   Context_Recall |   Context_Relevance |   Answer_Relevancy |
|:------------------------------|---------------:|-----------------:|--------------------:|-------------------:|
| Local Ollama + No Re-ranking  |           0.95 |             1    |                0.97 |               0.96 |
| Gemini API + Query Rewrite    |           0.88 |             0.94 |                0.89 |               0.91 |
| Hybrid Local-API + Re-ranking |           0.92 |             0.98 |                0.95 |               0.93 |
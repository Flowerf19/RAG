# Evaluation (Ragas integration)

This folder provides evaluation tools and a backend dashboard for assessing RAG system quality using the Ragas framework. The evaluation modules support both light-weight similarity heuristics and full Ragas-based metrics (faithfulness, context recall, answer correctness, answer relevancy) using configurable LLM providers (Gemini, Ollama, etc.).

Key points:
- The project ships a `RagasEvaluator` wrapper (see `evaluation/backend_dashboard/ragas_evaluator.py`) that integrates with the `ragas` package to compute standardized RAG metrics.
- For accurate Ragas evaluations you should configure an LLM provider (Gemini via `GOOGLE_API_KEY` or Ollama local) before running evaluations.

## Requirements

- Python 3.10+
- Install project requirements (recommended virtualenv):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

- Install Ragas and related extras (if not already present):

```bash
pip install ragas datasets langchain
# If using Gemini (Google): install langchain Google GenAI wrapper
pip install git+https://github.com/google-research/langchain-google-generativeai.git  # optional
# If using Ollama locally: install langchain_ollama
pip install langchain-ollama
```

Note: some providers (Gemini) require an API key and provider-specific wrappers; Ollama works with a local Ollama daemon.

## Quick Start — Ragas evaluation

1. Set your environment variables (example for Gemini):

```powershell
setx GOOGLE_API_KEY "your_gemini_key"
# or on Linux/Mac:
# export GOOGLE_API_KEY="your_gemini_key"
```

2. Run the example usage to evaluate a sample response (uses `RagasEvaluator`):

```bash
python evaluation/example_usage.py
```

3. Run the Streamlit dashboard (UI lives in `ui/dashboard`):

```powershell
.venv\Scripts\Activate.ps1
streamlit run ui/dashboard/app.py
```

## Usage examples

### Single evaluation (programmatic)

```python
from evaluation.backend_dashboard.ragas_evaluator import RagasEvaluator

# Initialize (use 'gemini' or 'ollama')
e = RagasEvaluator(llm_provider='gemini', api_key='YOUR_GOOGLE_API_KEY')
res = e.evaluate(
    question='What is supervised learning?',
    answer='Supervised learning trains models on labeled data.',
    contexts=['Supervised learning uses labeled examples...', 'It trains models to map inputs to outputs.'],
    ground_truth='Supervised learning is a method that trains a model using labeled data.'
)
print(res)
```

### Batch evaluation

Use `RagasEvaluator.evaluate_batch()` to evaluate multiple samples; the wrapper adds a short delay between requests to avoid rate limits when using cloud LLMs.

## Integration with backend dashboard

`RagasBackendDashboard` extends the existing backend API with a `evaluate_with_ragas()` method that runs Ragas evaluation and optionally saves results to the metrics database. The backend dashboard class lives in `evaluation/backend_dashboard/api.py` and stores results in the configured metrics DB (`evaluation/metrics/database.py`).

## Configuration

- Environment variables:
  - `GOOGLE_API_KEY` — required for Gemini (if using Gemini)
  - `METRICS_DB_PATH` — path for metrics database (default: `data/metrics.db`)

- Database: the evaluation modules use SQLite by default; ensure the `data/` folder exists and is writable.

## Metrics (Ragas)

Ragas computes multiple RAG-specific metrics; this project exposes at least:
- `faithfulness` — how well the answer is supported by provided contexts
- `context_recall` — how much ground truth is covered by contexts
- `answer_correctness` — correctness compared to ground truth (requires embeddings)
- `answer_relevancy` — relevancy of answer to the query

## Troubleshooting & Tips

- If you see import errors for provider wrappers, install the corresponding langchain provider package (e.g., `langchain-google-generativeai` for Gemini).
- For local evaluation, Ollama is a practical choice — ensure the Ollama daemon is running and reachable.
- To speed up large-scale testing, use the similarity-based evaluators in `evaluation/evaluators` which avoid external API calls.
- When using cloud LLMs, be mindful of API costs and rate limits; use `RagasEvaluator.request_delay` to space requests.

---

If you want, I can also:
- add an example Jupyter notebook showing how to evaluate a set of queries with `RagasEvaluator`,
- update `evaluation/example_usage.py` to include both Ragas and similarity-based examples, or
- add instructions for configuring the Streamlit dashboard to show Ragas metrics.
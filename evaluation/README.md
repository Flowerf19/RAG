# RAG Evaluation System

A minimalist evaluation system for comparing RAG model performance and accuracy.

## Features

- **Automated Evaluation**: LLM-based scoring or similarity heuristics
- **Performance Metrics**: Latency, throughput, faithfulness, relevance
- **SQLite Storage**: Lightweight database for metrics
- **Streamlit Dashboard**: Real-time visualization and comparison
- **Multi-Model Support**: Compare GPT, Mistral, DeepSeek, and local models

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly scikit-learn
```

### 2. Run Example

```bash
python evaluation/example_usage.py
```

### 3. Launch Dashboard

```bash
streamlit run ui/dashboard/app.py
```

## Architecture

```
User Query
    ↓
RAG Pipeline (multiple models)
    ↓
Evaluation Logger → metrics.db
    ↓
Backend Dashboard API
    ↓
Streamlit Dashboard
```

## Usage

### Basic Evaluation

```python
from evaluation.metrics.logger import EvaluationLogger
from evaluation.evaluators.auto_evaluator import AutoEvaluator

# Initialize
logger = EvaluationLogger()
evaluator = AutoEvaluator()

# Evaluate response
faithfulness, relevance = evaluator.evaluate_response(query, answer, context)

# Log metrics
logger.log_evaluation(
    query=query,
    model="gpt-4",
    latency=1.25,
    faithfulness=faithfulness,
    relevance=relevance
)
```

### Pipeline Integration

```python
with logger.time_and_log(query, model_name) as timer:
    # Run your RAG pipeline
    result = rag_pipeline.run(query)

    # Evaluate and log
    faithfulness, relevance = evaluator.evaluate_response(query, result.answer, result.context)
    timer.set_scores(faithfulness=faithfulness, relevance=relevance)
```

## Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Latency | Response time | seconds |
| Faithfulness | Answer matches context | 0-1 |
| Relevance | Answer matches query | 0-1 |
| Error Rate | Pipeline failure rate | % |

## Dashboard Features

- **Overview Stats**: Total queries, average accuracy, latency, error rate
- **Model Comparison**: Tables for LLM, embedding, and reranking models
- **Performance Charts**: Latency trends and accuracy comparisons
- **Recent Activity**: Latest evaluation results

## Configuration

### Environment Variables

```bash
# For LLM-based evaluation
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key

# Database path
METRICS_DB_PATH=data/metrics.db
```

### Model Categories

The system automatically categorizes models:
- **LLM**: GPT, Claude, Mistral, DeepSeek
- **Embedding**: BGE-M3, E5-Large, embedding models
- **Reranking**: Cross-encoders, reranking models

## API Reference

### EvaluationLogger

- `log_evaluation()`: Log single evaluation
- `time_and_log()`: Context manager for pipeline timing

### AutoEvaluator

- `evaluate_response()`: Evaluate answer quality
- Supports both LLM and similarity-based evaluation

### BackendDashboard

- `get_overview_stats()`: System-wide statistics
- `get_model_comparison_data()`: Model comparison tables
- `get_latency_over_time()`: Time series data
- `get_accuracy_by_model()`: Accuracy metrics

## Troubleshooting

### Common Issues

- **No evaluation scores**: Check if embedder/LLM clients are properly initialized
- **Database errors**: Ensure data/ directory exists and is writable
- **Dashboard not loading**: Verify streamlit and plotly are installed

### Performance Tips

- Use similarity-based evaluation for faster processing
- Batch evaluations when possible
- Consider sampling for large datasets
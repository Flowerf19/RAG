"""
Data Preparation Utilities for RAG Metrics Visualization
========================================================

Functions to prepare and aggregate RAG evaluation metrics data for visualization.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def prepare_metrics_dataframe(evaluation_results: List[Dict[str, Any]],
                            config_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepare metrics data for visualization from evaluation results.

    Args:
        evaluation_results: List of evaluation result dictionaries
        config_names: Optional list of configuration names (if None, uses indices)

    Returns:
        DataFrame with columns: Configuration, Faithfulness, Context_Recall, Answer_Correctness, Answer_Relevancy
    """
    try:
        if not evaluation_results:
            logger.warning("No evaluation results provided")
            return pd.DataFrame()

        # If no config names provided, use indices
        if config_names is None:
            config_names = [f"Config_{i+1}" for i in range(len(evaluation_results))]

        # Extract metrics from each result
        rows = []
        for i, result in enumerate(evaluation_results):
            config_name = config_names[i] if i < len(config_names) else f"Config_{i+1}"

            # Handle different result formats
            if isinstance(result, dict):
                # Direct metrics dict
                faithfulness = result.get('faithfulness', result.get('Faithfulness', 0.0))
                context_recall = result.get('context_recall', result.get('Context_Recall', 0.0))
                answer_correctness = result.get('answer_correctness', result.get('Answer_Correctness', 0.0))
                answer_relevancy = result.get('answer_relevancy', result.get('Answer_Relevancy', 0.0))
            else:
                # Assume it's an object with attributes
                faithfulness = getattr(result, 'faithfulness', 0.0)
                context_recall = getattr(result, 'context_recall', 0.0)
                answer_correctness = getattr(result, 'answer_correctness', 0.0)
                answer_relevancy = getattr(result, 'answer_relevancy', 0.0)

            rows.append({
                'Configuration': config_name,
                'Faithfulness': float(faithfulness),
                'Context_Recall': float(context_recall),
                'Answer_Correctness': float(answer_correctness),
                'Answer_Relevancy': float(answer_relevancy)
            })

        df = pd.DataFrame(rows)
        logger.info(f"Prepared DataFrame with {len(df)} configurations")
        return df

    except Exception as e:
        logger.error(f"Error preparing metrics DataFrame: {e}")
        return pd.DataFrame()


def prepare_metrics_from_ragas_output(ragas_summary: Dict[str, Any],
                                    config_name: str = "Current_Config") -> pd.DataFrame:
    """
    Prepare DataFrame from Ragas evaluation summary output.

    Args:
        ragas_summary: Summary dict from Ragas evaluation
        config_name: Name for this configuration

    Returns:
        DataFrame ready for visualization
    """
    try:
        # Extract metrics from summary
        faithfulness = ragas_summary.get('faithfulness', {}).get('mean', 0.0)
        context_recall = ragas_summary.get('context_recall', {}).get('mean', 0.0)
        answer_correctness = ragas_summary.get('answer_correctness', {}).get('mean', 0.0)
        answer_relevancy = ragas_summary.get('answer_relevancy', {}).get('mean', 0.0)

        df = pd.DataFrame([{
            'Configuration': config_name,
            'Faithfulness': float(faithfulness),
            'Context_Recall': float(context_recall),
            'Answer_Correctness': float(answer_correctness),
            'Answer_Relevancy': float(answer_relevancy)
        }])

        logger.info(f"Prepared DataFrame from Ragas output for config: {config_name}")
        return df

    except Exception as e:
        logger.error(f"Error preparing DataFrame from Ragas output: {e}")
        return pd.DataFrame()
"""
Example: Using RAG Evaluation System
Demonstrates how to integrate evaluation into your RAG pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from evaluation.metrics.logger import EvaluationLogger
from evaluation.evaluators.auto_evaluator import AutoEvaluator


def example_usage():
    """Example of how to use the evaluation system."""

    # Initialize components
    logger = EvaluationLogger()
    evaluator = AutoEvaluator()  # Will use similarity-based evaluation if no LLM/embedder provided

    # Example query
    query = "What is machine learning?"
    context = "Machine learning is a subset of artificial intelligence..."
    answer = "Machine learning is a type of AI that allows computers to learn from data."

    # Evaluate response
    faithfulness, relevance = evaluator.evaluate_response(query, answer, context)

    # Log the evaluation
    logger.log_evaluation(
        query=query,
        model="example-model",
        latency=1.25,
        faithfulness=faithfulness,
        relevance=relevance,
        error=False
    )

    print(f"Logged evaluation: faithfulness={faithfulness:.3f}, relevance={relevance:.3f}")


def example_with_pipeline_integration():
    """Example of integrating evaluation into RAG pipeline using context manager."""

    logger = EvaluationLogger()

    # Simulate RAG pipeline execution
    query = "How does RAG work?"

    with logger.time_and_log(query, "gpt-4-turbo") as timer:
        # Simulate pipeline work
        time.sleep(0.5)  # Simulated processing time

        # Simulate getting results
        context = "RAG combines retrieval and generation..."
        answer = "RAG works by first retrieving relevant documents, then using them to generate answers."

        # Evaluate and set scores
        evaluator = AutoEvaluator()
        faithfulness, relevance = evaluator.evaluate_response(query, answer, context)
        timer.set_scores(faithfulness=faithfulness, relevance=relevance)

    print("Pipeline execution logged with evaluation scores")


if __name__ == "__main__":
    print("RAG Evaluation System Examples")
    print("=" * 40)

    print("\n1. Basic evaluation logging:")
    example_usage()

    print("\n2. Pipeline integration with context manager:")
    example_with_pipeline_integration()

    print("\n✅ Examples completed. Check data/metrics.db for logged data.")
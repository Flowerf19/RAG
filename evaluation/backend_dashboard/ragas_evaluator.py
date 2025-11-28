#!/usr/bin/env python3
"""
Ragas-based RAG Evaluation Module
==================================

This module provides standardized RAG evaluation using the Ragas framework
with Gemini LLM for evaluation metrics.

Supported Metrics:
- faithfulness: Measures how faithful the answer is to the context
- context_recall: Measures how much of the ground truth is covered by the context
- answer_correctness: Measures how correct the answer is compared to the ground truth

Usage:
    from evaluation.backend_dashboard.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator()
    results = evaluator.evaluate(
        question="What is machine learning?",
        answer="Machine learning is a subset of AI...",
        contexts=["ML is part of AI...", "Deep learning uses neural networks..."],
        ground_truth="Machine learning is a method of data analysis..."
    )
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, AnswerRelevancy
from ragas.metrics.collections import AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset

# Conditional imports for different LLM providers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: F401
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: F401
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from evaluation.backend_dashboard.api import BackendDashboard

logger = logging.getLogger(__name__)


@dataclass
class RagasEvaluationResult:
    """Container for Ragas evaluation results."""
    faithfulness: float
    context_recall: float
    answer_correctness: float
    answer_relevancy: float
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


class RagasEvaluator:
    """
    Ragas-based RAG evaluator supporting multiple LLM providers.

    This class provides standardized evaluation metrics for RAG systems
    using the Ragas framework with configurable LLM backends.
    """

    def __init__(self, llm_provider: str = 'gemini', model_name: Optional[str] = None, api_key: Optional[str] = None, request_delay: int = 2):
        """
        Initialize the Ragas evaluator with specified LLM provider.

        Args:
            llm_provider: LLM provider to use ('gemini' or 'ollama')
            model_name: Model name (optional, defaults based on provider)
            api_key: API key for Gemini (not needed for Ollama)
            request_delay: Delay in seconds between API requests to avoid rate limiting.
        """
        self.llm_provider = llm_provider
        self.model_name = model_name or ('gemma3:1b' if llm_provider == 'ollama' else 'gemini-2.0-flash-exp')
        self.request_delay = request_delay

        if llm_provider == 'gemini':
            self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY environment variable must be set or api_key must be provided for Gemini")
            
            # Import Gemini components
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            # Initialize Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.1
            )
            
            # Initialize Gemini embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
            
        elif llm_provider == 'ollama':
            # Initialize Ollama LLM
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0.1
            )
            
            # Initialize Ollama embeddings
            self.embeddings = OllamaEmbeddings(
                model="embeddinggemma:latest"
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Use 'gemini' or 'ollama'")

        # Wrap for Ragas compatibility
        self.ragas_llm = LangchainLLMWrapper(self.llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

        # Configure evaluation metrics
        self.answer_relevancy = AnswerRelevancy()
        
        if llm_provider == 'ollama':
            # For Ollama, skip AnswerCorrectness as it requires modern InstructorLLM
            self.answer_correctness = None
            self.metrics = [
                faithfulness,
                context_recall,
                self.answer_relevancy,
            ]
        else:
            # For Gemini and other providers, include AnswerCorrectness
            self.answer_correctness = AnswerCorrectness(embeddings=self.ragas_embeddings)
            self.metrics = [
                faithfulness,
                context_recall,
                self.answer_correctness,
                self.answer_relevancy,
            ]

        # Set LLM for metrics that need it
        faithfulness.llm = self.ragas_llm
        context_recall.llm = self.ragas_llm
        self.answer_relevancy.llm = self.ragas_llm
        if self.answer_correctness:
            self.answer_correctness.llm = self.ragas_llm

        logger.info(f"RagasEvaluator initialized with {llm_provider.upper()} LLM ({self.model_name})")

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> RagasEvaluationResult:
        """
        Evaluate a single RAG response using Ragas metrics.

        Args:
            question: The user's question
            answer: The RAG system's answer
            contexts: List of retrieved context chunks
            ground_truth: The expected correct answer

        Returns:
            RagasEvaluationResult with all metric scores
        """
        try:
            # Prepare data for Ragas
            data = {
                'question': [question],
                'answer': [answer],
                'contexts': [contexts],
                'ground_truth': [ground_truth]
            }

            # Create dataset
            dataset = Dataset.from_dict(data)

            # Run evaluation with custom timeout for Ollama
            run_config = RunConfig(
                timeout=600,  # 10 minutes timeout for Ollama
                max_retries=5,  # Reduce retries to avoid long waits
                max_workers=4  # Reduce workers to be more conservative
            )
            
            logger.info(f"Evaluating question: {question[:50]}... (timeout: {run_config.timeout}s, workers: {run_config.max_workers})")
            try:
                results = evaluate(
                    dataset=dataset,
                    metrics=self.metrics,
                    llm=self.ragas_llm,
                    embeddings=self.ragas_embeddings,
                    run_config=run_config,
                    raise_exceptions=False  # Don't raise on individual metric failures
                )
            except Exception as eval_error:
                logger.error(f"Ragas evaluation failed for question: {question[:50]}... Error: {str(eval_error)}")
                raise

            # Extract scores
            faithfulness_score = float(results['faithfulness'][0])
            context_recall_score = float(results['context_recall'][0])

            # Robust lookup for AnswerCorrectness under variant keys (only if metric is available)
            answer_correctness_score = 0.0
            if self.answer_correctness:
                possible_ac_keys = ['answer_correctness', 'nv_answer_correctness', 'AnswerCorrectness', 'nv_answer_correctness_mean']
                for k in possible_ac_keys:
                    if k in results:
                        try:
                            answer_correctness_score = float(results[k][0])
                            break
                        except Exception:
                            continue

            answer_relevancy_score = float(results['answer_relevancy'][0])

            result = RagasEvaluationResult(
                faithfulness=faithfulness_score,
                context_recall=context_recall_score,
                answer_correctness=answer_correctness_score,
                answer_relevancy=answer_relevancy_score,
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth
            )

            logger.info(f"Evaluation complete - Faithfulness: {faithfulness_score:.3f}, "
                       f"Context Recall: {context_recall_score:.3f}, "
                       f"{'Answer Correctness: {:.3f}, '.format(answer_correctness_score) if self.answer_correctness else ''}"
                       f"Answer Relevancy: {answer_relevancy_score:.3f}")

            return result

        except Exception as e:
            logger.error(f"Error during Ragas evaluation: {str(e)}")
            raise

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: List[str]
    ) -> List[RagasEvaluationResult]:
        """
        Evaluate multiple RAG responses in batch.

        Args:
            questions: List of user questions
            answers: List of RAG system answers
            contexts_list: List of retrieved context chunks for each question
            ground_truths: List of expected correct answers

        Returns:
            List of RagasEvaluationResult objects
        """
        if not (len(questions) == len(answers) == len(contexts_list) == len(ground_truths)):
            raise ValueError("All input lists must have the same length")

        results = []
        for i, (q, a, ctx, gt) in enumerate(zip(questions, answers, contexts_list, ground_truths)):
            logger.info(f"Evaluating sample {i+1}/{len(questions)}")
            result = self.evaluate(q, a, ctx, gt)
            results.append(result)
            
            # Add delay between requests to avoid rate limiting
            if i < len(questions) - 1:  # Don't delay after the last request
                logger.info(f"Waiting {self.request_delay} seconds before next evaluation...")
                time.sleep(self.request_delay)

        return results


class RagasBackendDashboard(BackendDashboard):
    """
    Extended BackendDashboard with Ragas evaluation capabilities.

    This class integrates Ragas evaluation into the existing dashboard API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ragas_evaluator = RagasEvaluator()

    def evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate using Ragas framework and optionally save to database.

        Args:
            question: User's question
            answer: RAG system answer
            contexts: Retrieved context chunks
            ground_truth: Expected answer
            save_to_db: Whether to save results to metrics database

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Run Ragas evaluation
            result = self.ragas_evaluator.evaluate(question, answer, contexts, ground_truth)

            # Prepare result dictionary
            evaluation_data = {
                'faithfulness': result.faithfulness,
                'context_recall': result.context_recall,
                'answer_correctness': result.answer_correctness,
                'answer_relevancy': result.answer_relevancy,
                'question': result.question,
                'answer': result.answer,
                'contexts': result.contexts,
                'ground_truth': result.ground_truth,
                'evaluation_type': 'ragas'
            }

            if save_to_db:
                # Save to database
                try:
                    meta = {'evaluation_type': 'ragas'}
                    self.db.insert_metric(
                        query=result.question,
                        model='ragas',
                        faithfulness=result.faithfulness,
                        relevance=result.answer_relevancy,
                        answer_correctness=result.answer_correctness,
                        recall=result.context_recall,
                        metadata=json.dumps(meta),
                    )
                except Exception:
                    logger.exception('Failed to save ragas evaluation to DB')

            logger.info("Ragas evaluation completed and saved to database")
            return evaluation_data

        except Exception as e:
            logger.error(f"Error in Ragas evaluation: {str(e)}")
            raise


# Convenience function for quick evaluation
def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    api_key: Optional[str] = None
) -> RagasEvaluationResult:
    """
    Quick evaluation function for single RAG responses.

    Args:
        question: User's question
        answer: RAG system answer
        contexts: Retrieved context chunks
        ground_truth: Expected answer
        api_key: Optional Google API key

    Returns:
        RagasEvaluationResult object
    """
    evaluator = RagasEvaluator(api_key=api_key)
    return evaluator.evaluate(question, answer, contexts, ground_truth)


if __name__ == "__main__":
    # Example usage
    import os

    # Set your API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        exit(1)

    # Create evaluator
    evaluator = RagasEvaluator(api_key=api_key)

    # Example evaluation
    result = evaluator.evaluate(
        question="What is machine learning?",
        answer="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        contexts=[
            "Machine learning is a method of data analysis that automates analytical model building.",
            "It is a branch of artificial intelligence based on the idea that systems can learn from data."
        ],
        ground_truth="Machine learning is a subset of AI that allows systems to automatically learn and improve from experience."
    )

    print("Evaluation Results:")
    print(f"Faithfulness: {result.faithfulness:.3f}")
    print(f"Context Recall: {result.context_recall:.3f}")
    print(f"Answer Correctness: {result.answer_correctness:.3f}")
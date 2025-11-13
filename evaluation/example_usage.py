"""
RAG Evaluation System Examples
Demonstrates how to integrate evaluation into your RAG pipeline.
Refactored into a class-based structure for better organization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from evaluation.metrics.logger import EvaluationLogger
from evaluation.evaluators.auto_evaluator import AutoEvaluator

# Optional imports for models
try:
    from embedders.embedder_factory import EmbedderFactory
    from embedders.embedder_type import EmbedderType
    from embedders.model.embedding_profile import EmbeddingProfile
    _embedders_available = True
except ImportError:
    _embedders_available = False

try:
    from llm.client_factory import LLMClientFactory
    _llm_available = True
except ImportError:
    _llm_available = False


class EvaluationExamples:
    """Collection of examples demonstrating RAG evaluation system usage."""

    def __init__(self,
                 embedder_type: str = None,
                 llm_type: str = None):
        """Initialize example components with optional model configuration."""
        self.logger = EvaluationLogger()

        # Configure evaluator with models if provided
        embedder = None
        llm_client = None

        if embedder_type and _embedders_available:
            try:
                factory = EmbedderFactory()
                if embedder_type == "ollama":
                    embedder = factory.create_bge_m3()
                elif embedder_type == "huggingface_local":
                    embedder = factory.create_huggingface_local(device="cpu")
                elif embedder_type == "huggingface_api":
                    embedder = factory.create_huggingface_api()
                print(f"Using embedder: {embedder_type}")
            except Exception as e:
                print(f"Failed to create embedder {embedder_type}: {e}")

        if llm_type and _llm_available:
            try:
                factory = LLMClientFactory()
                if llm_type == "gemini":
                    llm_client = factory.create_gemini()
                elif llm_type == "lmstudio":
                    llm_client = factory.create_lmstudio()
                elif llm_type == "ollama":
                    llm_client = factory.create_ollama()
                print(f"Using LLM: {llm_type}")
            except Exception as e:
                print(f"Failed to create LLM client {llm_type}: {e}")

        self.evaluator = AutoEvaluator(embedder=embedder, llm_client=llm_client)
        self.embedder_type = embedder_type
        self.llm_type = llm_type

    def basic_evaluation_example(self):
        """Example of basic evaluation logging."""
        print("1. Basic evaluation logging:")

        # Example query
        query = "What is machine learning?"
        context = "Machine learning is a subset of artificial intelligence..."
        answer = "Machine learning is a type of AI that allows computers to learn from data."
        
        # Example retrieved and relevant docs for recall calculation
        retrieved_docs = ["Machine learning is a subset of AI", "ML allows computers to learn", "Deep learning is part of ML"]
        relevant_docs = ["Machine learning is a subset of AI", "ML allows computers to learn", "AI encompasses ML"]

        # Evaluate response
        faithfulness, relevance, recall = self.evaluator.evaluate_response(
            query, answer, context, retrieved_docs, relevant_docs
        )
        print(f"Evaluation result: faithfulness={faithfulness:.3f}, relevance={relevance:.3f}, recall={recall:.3f}")

        # Log the evaluation
        self.logger.log_evaluation(
            query=query,
            model="example-model",
            latency=1.25,
            faithfulness=faithfulness,
            relevance=relevance,
            recall=recall,
            error=False,
            embedder_model=self.embedder_type or "none",
            llm_model=self.llm_type or "none",
            reranker_model="none",
            query_enhanced=False,
            embedding_tokens=150,
            reranking_tokens=0,
            llm_tokens=50,
            total_tokens=200,
            retrieval_chunks=5
        )

        print(f"Logged evaluation: faithfulness={faithfulness:.3f}, relevance={relevance:.3f}, recall={recall:.3f}")

    def pipeline_integration_example(self):
        """Example of integrating evaluation into RAG pipeline using context manager."""
        print("2. Pipeline integration with context manager:")

        # Simulate RAG pipeline execution
        query = "How does RAG work?"

        with self.logger.time_and_log(query, "gpt-4-turbo") as timer:
            # Set model configuration
            timer.set_model_config(
                embedder_model=self.embedder_type or "none",
                llm_model=self.llm_type or "none",
                reranker_model="none",
                query_enhanced=False,
                retrieval_chunks=3
            )

            # Simulate pipeline work
            time.sleep(0.5)  # Simulated processing time

            # Simulate getting results
            context = "RAG combines retrieval and generation..."
            answer = "RAG works by first retrieving relevant documents, then using them to generate answers."
            
            # Simulate retrieved and relevant docs for recall
            retrieved_docs = ["RAG combines retrieval and generation", "Retrieval finds relevant docs", "Generation creates answers"]
            relevant_docs = ["RAG combines retrieval and generation", "Retrieval finds relevant docs", "Generation uses retrieved docs"]

            # Evaluate and set scores
            faithfulness, relevance, recall = self.evaluator.evaluate_response(
                query, answer, context, retrieved_docs, relevant_docs
            )
            timer.set_scores(faithfulness=faithfulness, relevance=relevance, recall=recall)

        print("Pipeline execution logged with evaluation scores")

    def run_all_examples(self):
        """Run all evaluation examples."""
        print("RAG Evaluation System Examples")
        print("=" * 40)

        if self.embedder_type or self.llm_type:
            print(f"Using models: Embedder={self.embedder_type or 'None'}, LLM={self.llm_type or 'None'}")
        else:
            print("Using default evaluation (similarity-based, returns 0.5 scores)")

        print()

        self.basic_evaluation_example()
        print()
        self.pipeline_integration_example()
        print()

        print("✅ Examples completed. Check data/metrics.db for logged data.")


def main():
    """Main entry point for running examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation examples")
    parser.add_argument("--embedder", type=str, help="Embedder type (e.g., huggingface_local, ollama)")
    parser.add_argument("--llm", type=str, help="LLM type (e.g., gemini, lmstudio, ollama)")

    args = parser.parse_args()

    examples = EvaluationExamples(
        embedder_type=args.embedder,
        llm_type=args.llm
    )
    examples.run_all_examples()


if __name__ == "__main__":
    main()
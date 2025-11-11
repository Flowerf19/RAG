"""
Auto-Evaluation Functions
Automatic evaluation of RAG responses using LLM scoring or similarity heuristics.
"""

from typing import Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from embedders.i_embedder import IEmbedder
from llm.base_client import BaseLLMClient


class AutoEvaluator:
    """Automatic evaluation of RAG response quality."""

    def __init__(self,
                 embedder: Optional[IEmbedder] = None,
                 llm_client: Optional[BaseLLMClient] = None):
        """
        Initialize evaluator with optional embedder and LLM client.

        Args:
            embedder: For similarity-based evaluation
            llm_client: For LLM-based scoring
        """
        self.embedder = embedder
        self.llm_client = llm_client

    def evaluate_response(self,
                         query: str,
                         answer: str,
                         context: str) -> Tuple[float, float]:
        """
        Evaluate response quality using available methods.

        Returns:
            Tuple of (faithfulness, relevance) scores (0-1)
        """
        faithfulness = self._evaluate_faithfulness(answer, context)
        relevance = self._evaluate_relevance(answer, query)

        return faithfulness, relevance

    def _evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Evaluate how faithful the answer is to the provided context."""
        if self.llm_client:
            return self._llm_faithfulness_score(answer, context)
        elif self.embedder:
            return self._similarity_faithfulness_score(answer, context)
        else:
            return 0.5  # Neutral score if no evaluation method available

    def _evaluate_relevance(self, answer: str, query: str) -> float:
        """Evaluate how relevant the answer is to the query."""
        if self.llm_client:
            return self._llm_relevance_score(answer, query)
        elif self.embedder:
            return self._similarity_relevance_score(answer, query)
        else:
            return 0.5  # Neutral score if no evaluation method available

    def _llm_faithfulness_score(self, answer: str, context: str) -> float:
        """Use LLM to score faithfulness."""
        prompt = f"""
        Rate how faithful this answer is to the provided context on a scale of 0-1.
        A score of 1.0 means the answer is completely faithful to the context.
        A score of 0.0 means the answer contradicts or ignores the context.

        Context: {context}
        Answer: {answer}

        Return only a number between 0 and 1.
        """

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # Clamp to 0-1
        except Exception:
            return 0.5

    def _llm_relevance_score(self, answer: str, query: str) -> float:
        """Use LLM to score relevance."""
        prompt = f"""
        Rate how relevant this answer is to the query on a scale of 0-1.
        A score of 1.0 means the answer directly addresses the query.
        A score of 0.0 means the answer is completely unrelated.

        Query: {query}
        Answer: {answer}

        Return only a number between 0 and 1.
        """

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # Clamp to 0-1
        except Exception:
            return 0.5

    def _similarity_faithfulness_score(self, answer: str, context: str) -> float:
        """Use embedding similarity for faithfulness."""
        try:
            answer_emb = self.embedder.encode([answer])[0]
            context_emb = self.embedder.encode([context])[0]

            similarity = cosine_similarity([answer_emb], [context_emb])[0][0]
            return float(max(0.0, min(1.0, similarity)))
        except Exception:
            return 0.5

    def _similarity_relevance_score(self, answer: str, query: str) -> float:
        """Use embedding similarity for relevance."""
        try:
            answer_emb = self.embedder.encode([answer])[0]
            query_emb = self.embedder.encode([query])[0]

            similarity = cosine_similarity([answer_emb], [query_emb])[0][0]
            return float(max(0.0, min(1.0, similarity)))
        except Exception:
            return 0.5
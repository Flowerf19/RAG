"""
Auto-Evaluation Functions
Automatic evaluation of RAG responses using LLM scoring or similarity heuristics.
"""

from typing import Tuple, Optional, List
import re
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
                         context: str,
                         retrieved_docs: Optional[List[str]] = None,
                         relevant_docs: Optional[List[str]] = None) -> Tuple[float, float, Optional[float]]:
        """
        Evaluate response quality using available methods.

        Args:
            query: The user's query
            answer: The generated answer
            context: The context provided to generate the answer
            retrieved_docs: Optional list of retrieved documents for recall calculation
            relevant_docs: Optional list of ground truth relevant documents for recall calculation

        Returns:
            Tuple of (faithfulness, relevance, recall) scores (0-1), recall is None if not provided
        """
        faithfulness = self._evaluate_faithfulness(answer, context)
        relevance = self._evaluate_relevance(answer, query)
        
        recall = None
        if retrieved_docs is not None and relevant_docs is not None:
            recall = self.calculate_recall(retrieved_docs, relevant_docs)

        return faithfulness, relevance, recall

    def calculate_recall(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate recall for retrieval evaluation.
        
        Recall = (Number of relevant docs retrieved) / (Total number of relevant docs)
        
        Args:
            retrieved_docs: List of documents retrieved by the system
            relevant_docs: List of ground truth relevant documents
            
        Returns:
            Recall score between 0 and 1
        """
        if not relevant_docs:
            return 1.0  # Perfect recall if no relevant docs expected
        
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        # Number of relevant docs that were retrieved
        relevant_retrieved = len(retrieved_set.intersection(relevant_set))
        
        # Total number of relevant docs
        total_relevant = len(relevant_set)
        
        return relevant_retrieved / total_relevant

    def _extract_score_from_response(self, response: str) -> float:
        """Extract numerical score from LLM response."""
        # Find first floating point number in response
        match = re.search(r'(\d+\.?\d*)', response.strip())
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))  # Clamp to 0-1
            except ValueError:
                pass
        return 0.5  # Default if no valid score found

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
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.generate(messages, max_tokens=10)
            return self._extract_score_from_response(response)
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
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.generate(messages, max_tokens=10)
            return self._extract_score_from_response(response)
        except Exception:
            return 0.5

    def _similarity_faithfulness_score(self, answer: str, context: str) -> float:
        """Use embedding similarity for faithfulness."""
        try:
            answer_emb = self.embedder.embed(answer)
            context_emb = self.embedder.embed(context)

            similarity = cosine_similarity([answer_emb], [context_emb])[0][0]
            return float(max(0.0, min(1.0, similarity)))
        except Exception:
            return 0.5

    def _similarity_relevance_score(self, answer: str, query: str) -> float:
        """Use embedding similarity for relevance."""
        try:
            answer_emb = self.embedder.embed(answer)
            query_emb = self.embedder.embed(query)

            similarity = cosine_similarity([answer_emb], [query_emb])[0][0]
            return float(max(0.0, min(1.0, similarity)))
        except Exception:
            return 0.5
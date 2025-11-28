"""
Base Local Reranker
===================
Base class cho local rerankers (BGE, etc.)
"""

from typing import List
import logging
from reranking.i_reranker import IReranker, RerankResult, RerankerProfile

logger = logging.getLogger(__name__)


class BaseLocalReranker(IReranker):
    """Base class for local reranker implementations"""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize local reranker.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._profile = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer - to be implemented by subclasses"""
        raise NotImplementedError("Subclass must implement _load_model")
    
    @property
    def profile(self) -> RerankerProfile:
        """Get reranker profile"""
        if self._profile is None:
            raise RuntimeError("Profile not initialized")
        return self._profile
    
    def get_model_name(self) -> str:
        """Get specific reranker model name"""
        return self.profile.model_id if self.profile else self.model_name
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[RerankResult]:
        """
        Rerank documents using local model.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of results to return
            
        Returns:
            List of RerankResult sorted by score
        """
        if not documents:
            return []
        
        # Compute scores - to be implemented by subclasses
        scores = self._compute_scores(query, documents)
        
        # Create results with original indices
        results = [
            RerankResult(index=idx, score=score, document=doc)
            for idx, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k
        return results[:top_k]
    
    def _compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """Compute relevance scores - to be implemented by subclasses"""
        raise NotImplementedError("Subclass must implement _compute_scores")
    
    def test_connection(self) -> bool:
        """Test if model is loaded"""
        return self._model is not None and self._tokenizer is not None

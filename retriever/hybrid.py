from typing import List
from .embedding import GemmaEmbedder
from .bm25 import BM25Retriever
import numpy as np

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

class HybridRetriever:
    def __init__(self, embedder: GemmaEmbedder, bm25: BM25Retriever, alpha: float = 0.5):
        """
        Kết hợp điểm embedding và BM25.
        alpha: trọng số cho embedding (0.0~1.0)
        """
        self.embedder = embedder
        self.bm25 = bm25
        self.alpha = alpha

    def score(self, query: str, chunks: List[str]) -> List[float]:
        # Tính embedding score
        query_emb = self.embedder.embed([query])[0]
        chunk_embs = self.embedder.embed(chunks)
        emb_scores = [cosine_similarity(query_emb, emb) for emb in chunk_embs]
        # Tính BM25 score
        bm25_scores = self.bm25.get_scores(query)
        # Chuẩn hóa điểm BM25 về [0,1] nếu cần
        if max(bm25_scores) > 0:
            bm25_scores = [b / max(bm25_scores) for b in bm25_scores]
        # Kết hợp
        return [self.alpha * e + (1 - self.alpha) * b for e, b in zip(emb_scores, bm25_scores)]

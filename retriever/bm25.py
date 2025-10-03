from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    def __init__(self, documents: List[str]):
        """
        Khởi tạo BM25 với danh sách văn bản (mỗi văn bản là một chunk).
        """
        self.documents = documents
        self.bm25 = BM25Okapi([doc.split() for doc in documents])

    def get_scores(self, query: str) -> List[float]:
        """
        Tính điểm BM25 cho query với tất cả các chunk.
        """
        return self.bm25.get_scores(query.split()).tolist()

    def get_cosine_scores(self, query: str) -> List[float]:
        """
        Tính điểm cosine similarity giữa vector BM25 của query và các chunk.
        """
        # Vector hóa query và các chunk
        all_tokens = list(set(token for doc in self.documents for token in doc.split()))
        def vectorize(text):
            tokens = text.split()
            return np.array([tokens.count(tok) for tok in all_tokens])
        query_vec = vectorize(query)
        chunk_vecs = [vectorize(doc) for doc in self.documents]
        # Tính cosine similarity
        def cosine(a, b):
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                return 0.0
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return [cosine(query_vec, chunk_vec) for chunk_vec in chunk_vecs]

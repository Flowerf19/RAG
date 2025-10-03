from typing import List, Optional, Dict, Any
import numpy as np

class SimpleReranker:
    """
    Reranker đơn giản dựa trên cosine similarity với embedding.
    Không cần Hugging Face hoặc Ollama rerank API.
    """
    
    def __init__(self, embedder):
        """
        Args:
            embedder: Object có method embed() (ví dụ: GemmaEmbedder)
        """
        self.embedder = embedder
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = True
    ) -> Dict[str, Any]:
        """
        Rerank documents dựa trên cosine similarity với query.
        
        Args:
            query: Query string
            documents: Danh sách documents cần rerank
            top_n: Số lượng kết quả trả về (None = tất cả)
            return_documents: Có trả về nội dung document không
        
        Returns:
            Dict với format:
            {
                "results": [
                    {
                        "index": int,
                        "relevance_score": float,
                        "document": str (nếu return_documents=True)
                    }
                ]
            }
        """
        # Tính embedding cho query và documents
        query_emb = self.embedder.embed([query])[0]
        doc_embs = self.embedder.embed(documents)
        
        # Tính cosine similarity
        results = []
        for idx, doc_emb in enumerate(doc_embs):
            score = self._cosine_similarity(query_emb, doc_emb)
            result = {
                "index": idx,
                "relevance_score": float(score)
            }
            if return_documents:
                result["document"] = documents[idx]
            results.append(result)
        
        # Sắp xếp theo điểm giảm dần
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Lấy top_n nếu được chỉ định
        if top_n is not None:
            results = results[:top_n]
        
        return {"results": results}
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Tính cosine similarity giữa 2 vector."""
        vec_a = np.array(a)
        vec_b = np.array(b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# Ví dụ sử dụng:
if __name__ == "__main__":
    from embedding import GemmaEmbedder
    
    # Khởi tạo embedder và reranker
    embedder = GemmaEmbedder()
    reranker = SimpleReranker(embedder)
    
    # Test reranking
    query = "What is machine learning?"
    docs = [
        "Angela Merkel was the Chancellor of Germany",
        "Machine learning is a subset of artificial intelligence",
        "Pizza is made with tomatoes and cheese",
        "Deep learning uses neural networks for pattern recognition",
        "The weather today is sunny and warm"
    ]
    
    print("=== Simple Reranking Results ===")
    result = reranker.rerank(query, docs, top_n=3)
    
    for i, r in enumerate(result["results"], 1):
        print(f"{i}. [Index {r['index']}] Score: {r['relevance_score']:.4f}")
        print(f"   {r['document']}")
        print()

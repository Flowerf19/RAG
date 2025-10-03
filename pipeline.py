"""
Pipeline RAG hoàn chỉnh: Loaders → Chunkers → Retriever
Tích hợp 3 module để xây dựng hệ thống RAG end-to-end
"""

import sys
import os
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loaders.pdf_loader import PDFLoader
from loaders.model import PDFDocument
from chunkers.chunker import SemanticChunker
from chunkers.model import ChunkDocument, Chunk
from retriever.embedding import GemmaEmbedder
from retriever.bm25 import BM25Retriever
from retriever.hybrid import HybridRetriever
from retriever.reranker import SimpleReranker


class RAGPipeline:
    """
    Pipeline RAG hoàn chỉnh từ PDF đến retrieval.
    """
    
    def __init__(
        self,
        embedding_config_path: Optional[str] = None,
        hybrid_alpha: float = 0.5
    ):
        """
        Khởi tạo RAG Pipeline.
        
        Args:
            embedding_config_path: Đường dẫn config embedding (None = dùng mặc định)
            hybrid_alpha: Trọng số cho embedding trong hybrid (0.0~1.0)
        """
        # 1. Initialize components
        self.pdf_loader = PDFLoader()
        self.chunker = SemanticChunker()
        self.embedder = GemmaEmbedder(config_path=embedding_config_path)
        
        # 2. Storage for indexed data
        self.pdf_document: Optional[PDFDocument] = None
        self.chunk_document: Optional[ChunkDocument] = None
        self.chunk_texts: List[str] = []
        self.chunk_embeddings: List[List[float]] = []
        
        # 3. Retrieval components (initialized after indexing)
        self.bm25: Optional[BM25Retriever] = None
        self.hybrid: Optional[HybridRetriever] = None
        self.reranker: Optional[SimpleReranker] = None
        self.hybrid_alpha = hybrid_alpha
    
    def load_and_index(self, pdf_path: str) -> Dict[str, Any]:
        """
        Load PDF, chunk, và tạo index.
        
        Args:
            pdf_path: Đường dẫn đến file PDF
        
        Returns:
            Dict chứa thông tin về quá trình indexing
        """
        print(f"[1/4] Loading PDF: {pdf_path}")
        # Step 1: Load PDF
        self.pdf_document = self.pdf_loader.load(pdf_path)
        
        print(f"[2/4] Chunking document...")
        print(f"      PDF has {len(self.pdf_document.pages)} pages")
        # Step 2: Chunk document
        self.chunk_document = self.chunker.chunk(self.pdf_document)
        self.chunk_texts = [chunk.content for chunk in self.chunk_document.chunks]
        print(f"      Generated {len(self.chunk_texts)} chunks")
        
        print(f"[3/4] Computing embeddings for {len(self.chunk_texts)} chunks...")
        # Step 3: Compute embeddings
        self.chunk_embeddings = self.embedder.embed(self.chunk_texts)
        
        print(f"[4/4] Building BM25 index...")
        # Step 4: Build BM25 and hybrid retriever
        self.bm25 = BM25Retriever(self.chunk_texts)
        self.hybrid = HybridRetriever(self.embedder, self.bm25, alpha=self.hybrid_alpha)
        self.reranker = SimpleReranker(self.embedder)
        
        print("✅ Indexing complete!")
        
        return {
            "total_pages": len(self.pdf_document.pages) if self.pdf_document else 0,
            "total_chunks": len(self.chunk_texts),
            "embedding_dimension": len(self.chunk_embeddings[0]) if self.chunk_embeddings else 0
        }
    
    def search(
        self,
        query: str,
        top_k_hybrid: int = 50,
        top_k_final: int = 5,
        use_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm chunks liên quan đến query.
        
        Args:
            query: Câu truy vấn
            top_k_hybrid: Số lượng chunks lấy từ hybrid search
            top_k_final: Số lượng chunks trả về cuối cùng
            use_rerank: Có sử dụng reranker không
        
        Returns:
            List các chunks với điểm và metadata
        """
        if not self.chunk_texts:
            raise ValueError("Pipeline chưa được index. Gọi load_and_index() trước.")
        
        print(f"\n[Search] Query: {query}")
        print(f"[1/3] Hybrid search (top {top_k_hybrid})...")
        
        # Step 1: Hybrid search
        hybrid_scores = self.hybrid.score(query, self.chunk_texts)
        
        # Get top-k indices from hybrid
        top_indices = sorted(
            range(len(hybrid_scores)),
            key=lambda i: hybrid_scores[i],
            reverse=True
        )[:top_k_hybrid]
        
        top_chunks = [self.chunk_texts[i] for i in top_indices]
        
        # Step 2: Rerank if enabled
        if use_rerank and self.reranker:
            print(f"[2/3] Reranking top {top_k_hybrid} chunks...")
            rerank_result = self.reranker.rerank(query, top_chunks, top_n=top_k_final)
            
            # Map back to original indices and add metadata
            results = []
            for r in rerank_result["results"]:
                original_idx = top_indices[r["index"]]
                chunk = self.chunk_document.chunks[original_idx]
                results.append({
                    "chunk_index": original_idx,
                    "content": r["document"],
                    "score": r["relevance_score"],
                    "metadata": chunk.metadata.__dict__
                })
        else:
            print(f"[2/3] Skipping rerank, using hybrid scores...")
            # Just use hybrid scores
            results = []
            for idx in top_indices[:top_k_final]:
                chunk = self.chunk_document.chunks[idx]
                results.append({
                    "chunk_index": idx,
                    "content": chunk.content,
                    "score": hybrid_scores[idx],
                    "metadata": chunk.metadata.__dict__
                })
        
        print(f"[3/3] Returning top {len(results)} results ✅")
        return results
    
    def print_results(self, results: List[Dict[str, Any]]):
        """
        In kết quả tìm kiếm theo định dạng đẹp.
        
        Args:
            results: Kết quả từ search()
        """
        print("\n" + "="*80)
        print("SEARCH RESULTS")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result['score']:.4f} | Chunk Index: {result['chunk_index']}")
            print(f"Metadata: Page {result['metadata'].get('page', 'N/A')}, "
                  f"Section: {result['metadata'].get('section', 'N/A')}")
            print(f"\nContent:\n{result['content'][:300]}...")
            print("-" * 80)


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline(hybrid_alpha=0.5)
    
    # Load and index a PDF
    pdf_path = "data/pdf/Process_Risk Management.pdf"
    
    if os.path.exists(pdf_path):
        # Index the document
        info = pipeline.load_and_index(pdf_path)
        print(f"\n📊 Index Info:")
        print(f"  - Total pages: {info['total_pages']}")
        print(f"  - Total chunks: {info['total_chunks']}")
        print(f"  - Embedding dimension: {info['embedding_dimension']}")
        
        # Perform search
        query = "What is risk management?"
        results = pipeline.search(
            query=query,
            top_k_hybrid=50,
            top_k_final=5,
            use_rerank=True
        )
        
        # Print results
        pipeline.print_results(results)
    else:
        print(f"❌ File not found: {pdf_path}")
        print("Please update the pdf_path variable to point to a valid PDF file.")

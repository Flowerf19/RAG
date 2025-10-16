"""
Query Test Script - Test RAG query pipeline with vector store and LLM
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from RAG_system.EMBEDDERS.embedder_factory import EmbedderFactory
from RAG_system.EMBEDDERS.i_embedder import IEmbedder
from RAG_system.EMBEDDERS.embedder_type import EmbedderType
from RAG_system.EMBEDDERS.embedding_profile import EmbeddingProfile
from RAG_system.RETRIEVAL.VECTOR.vector_store import VectorStore
from RAG_system.LLM.llm_client import LLMClient

class RAGQueryTester:
    """RAG Query Testing System"""
    
    def __init__(self):
        """Initialize RAG components"""
        self.vector_dir = Path("prepare_data/VECTOR")
        self.embedder: Optional[IEmbedder] = None
        self.vector_store: Optional[VectorStore] = None
        self.llm_client: Optional[LLMClient] = None
        
    def load_components(self):
        """Load embedder, vector store, and LLM"""
        print("\n" + "="*70)
        print("🔄 LOADING RAG COMPONENTS")
        print("="*70)
        
        load_start = time.time()
        
        # 1. Initialize Embedder (Gemma)
        print("\n1️⃣ Initializing Embedder...")
        embed_start = time.time()
        
        profile = EmbeddingProfile(
            model_id="embeddinggemma",
            dimension=768,
            max_tokens=8192,
            normalize=True
        )
        
        factory = EmbedderFactory()
        self.embedder = factory.create(
            embedder_type=EmbedderType.GEMMA,
            profile=profile
        )
        
        embed_time = time.time() - embed_start
        print(f"✅ Embedder loaded in {embed_time:.2f}s")
        print(f"   Model: {self.embedder.model_name}")
        print(f"   Dimensions: {self.embedder.get_dimensions()}")
        
        # 2. Load Vector Store
        print("\n2️⃣ Loading Vector Store...")
        store_start = time.time()
        
        # Load all .npy embedding files
        embedding_files = list(self.vector_dir.glob("*_embeddings.npy"))
        metadata_files = list(self.vector_dir.glob("*_metadata.json"))
        
        if not embedding_files:
            print(f"❌ No embedding files found in: {self.vector_dir}")
            print("Please run embedding generation first.")
            sys.exit(1)
        
        print(f"   Found {len(embedding_files)} document embedding files")
        
        # Load all embeddings and chunks
        all_embeddings = []
        all_chunks = []
        
        for emb_file in embedding_files:
            doc_name = emb_file.stem.replace("_embeddings", "")
            meta_file = self.vector_dir / f"{doc_name}_metadata.json"
            
            # Load embeddings
            embeddings = np.load(emb_file)
            all_embeddings.append(embeddings)
            
            # Load metadata to get chunks
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    chunks = [chunk['text_preview'] for chunk in metadata['chunks']]
                    all_chunks.extend(chunks)
                    print(f"   ✓ Loaded {len(chunks)} chunks from {doc_name}")
        
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        # Create vector store and add embeddings
        embedding_dim = combined_embeddings.shape[1]
        self.vector_store = VectorStore(dimension=embedding_dim)
        self.vector_store.add_embeddings(combined_embeddings, all_chunks)
        
        store_time = time.time() - store_start
        print(f"✅ Vector store loaded in {store_time:.2f}s")
        print(f"   Total vectors: {len(all_chunks)}")
        print(f"   Embedding dimension: {embedding_dim}")
        
        # 3. Initialize LLM Client
        print("\n3️⃣ Initializing LLM...")
        llm_start = time.time()
        
        self.llm_client = LLMClient(model="gemma3:1b")
        
        llm_time = time.time() - llm_start
        print(f"✅ LLM client initialized in {llm_time:.2f}s")
        
        total_load_time = time.time() - load_start
        print(f"\n⏱️  Total loading time: {total_load_time:.2f}s")
        print("="*70)
        
    def query(self, query_text: str, k: int = 3, verbose: bool = True) -> dict:
        """
        Execute a single query through the RAG pipeline
        
        Args:
            query_text: User query
            k: Number of chunks to retrieve
            verbose: Print detailed information
            
        Returns:
            Dictionary with query results and metadata
        """
        if verbose:
            print("\n" + "="*70)
            print(f"🔍 QUERY: {query_text}")
            print("="*70)
        
        query_start = time.time()
        
        # Step 1: Embed query
        if verbose:
            print("\n1️⃣ Embedding query...")
        embed_start = time.time()
        
        if self.embedder is None:
            raise RuntimeError("Embedder not loaded. Call load_components() first.")
        
        query_embedding = self.embedder.embed(query_text)
        query_embedding_array = np.array(query_embedding, dtype='float32')
        
        embed_time = time.time() - embed_start
        if verbose:
            print(f"✅ Query embedded in {embed_time:.3f}s")
            print(f"   Embedding dimension: {len(query_embedding)}")
        
        # Step 2: Retrieve similar chunks
        if verbose:
            print("\n2️⃣ Retrieving relevant chunks...")
        retrieval_start = time.time()
        
        if self.vector_store is None:
            raise RuntimeError("Vector store not loaded. Call load_components() first.")
        
        results = self.vector_store.search(query_embedding_array, k=k)
        
        retrieval_time = time.time() - retrieval_start
        if verbose:
            print(f"✅ Retrieved {len(results)} chunks in {retrieval_time:.3f}s")
            print(f"\n   📄 Top {k} chunks by similarity:")
            for i, (chunk, score) in enumerate(results, 1):
                preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                print(f"   {i}. Score: {score:.4f} | {preview}")
        
        # Prepare context
        context = "\n\n".join([chunk for chunk, _ in results])
        
        # Step 3: Generate LLM response
        if verbose:
            print(f"\n3️⃣ Generating response with LLM...")
        llm_start = time.time()
        
        if self.llm_client is None:
            raise RuntimeError("LLM client not loaded. Call load_components() first.")
        
        llm_result = self.llm_client.generate(query_text, context)
        
        llm_time = time.time() - llm_start
        if verbose:
            print(f"✅ Response generated in {llm_time:.3f}s")
        
        # Calculate total time
        total_time = time.time() - query_start
        
        # Display results
        if verbose:
            print("\n" + "="*70)
            print("🤖 ASSISTANT RESPONSE")
            print("="*70)
            print(llm_result["response"])
            print("="*70)
            
            print(f"\n⏱️  TIMING BREAKDOWN:")
            print(f"   • Query embedding:  {embed_time:.3f}s")
            print(f"   • Retrieval:        {retrieval_time:.3f}s")
            print(f"   • LLM generation:   {llm_time:.3f}s")
            print(f"   • Total:            {total_time:.3f}s")
            
            print(f"\n📊 TOKEN USAGE:")
            print(f"   • Prompt tokens:    {llm_result['prompt_tokens']}")
            print(f"   • Response tokens:  {llm_result['response_tokens']}")
            print(f"   • Total tokens:     {llm_result['tokens_used']}")
            print(f"   • Session total:    {llm_result['total_tokens']}")
            print("="*70 + "\n")
        
        return {
            "query": query_text,
            "response": llm_result["response"],
            "retrieved_chunks": [(chunk, float(score)) for chunk, score in results],
            "timing": {
                "embedding": embed_time,
                "retrieval": retrieval_time,
                "llm": llm_time,
                "total": total_time
            },
            "tokens": {
                "prompt": llm_result["prompt_tokens"],
                "response": llm_result["response_tokens"],
                "total": llm_result["tokens_used"]
            }
        }
    
    def batch_query(self, queries: List[str], k: int = 3) -> List[dict]:
        """
        Execute multiple queries in batch
        
        Args:
            queries: List of query strings
            k: Number of chunks to retrieve per query
            
        Returns:
            List of query results
        """
        print("\n" + "="*70)
        print(f"🔄 BATCH QUERY MODE - {len(queries)} queries")
        print("="*70)
        
        results = []
        total_start = time.time()
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing: {query[:50]}...")
            result = self.query(query, k=k, verbose=False)
            results.append(result)
        
        total_time = time.time() - total_start
        
        # Summary
        print("\n" + "="*70)
        print("📊 BATCH QUERY SUMMARY")
        print("="*70)
        
        avg_embed_time = np.mean([r["timing"]["embedding"] for r in results])
        avg_retrieval_time = np.mean([r["timing"]["retrieval"] for r in results])
        avg_llm_time = np.mean([r["timing"]["llm"] for r in results])
        avg_total_time = np.mean([r["timing"]["total"] for r in results])
        
        total_tokens = sum([r["tokens"]["total"] for r in results])
        
        print(f"\n⏱️  Average Timing per Query:")
        print(f"   • Embedding:   {avg_embed_time:.3f}s")
        print(f"   • Retrieval:   {avg_retrieval_time:.3f}s")
        print(f"   • LLM:         {avg_llm_time:.3f}s")
        print(f"   • Total:       {avg_total_time:.3f}s")
        
        print(f"\n⏱️  Total Processing Time: {total_time:.2f}s")
        print(f"📊 Total Tokens Used: {total_tokens}")
        print(f"📊 Average Tokens per Query: {total_tokens/len(queries):.0f}")
        print("="*70 + "\n")
        
        return results
    
    def interactive_mode(self):
        """Interactive query mode"""
        print("\n" + "="*70)
        print("💬 INTERACTIVE QUERY MODE")
        print("="*70)
        print("Commands:")
        print("  • Type your question to query the RAG system")
        print("  • 'quit' or 'exit' - Exit the program")
        print("  • 'stats' - Show token statistics")
        print("  • 'clear' - Clear screen")
        print("="*70 + "\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    if self.llm_client is None:
                        print("\n❌ LLM client not loaded")
                    else:
                        print(f"\n📊 Session Statistics:")
                        print(f"   Total tokens used: {self.llm_client.total_tokens}")
                    continue
                
                if query.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                self.query(query)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                continue
    
    def save_results(self, results: List[dict], output_file: str = "query_results.json"):
        """Save query results to file"""
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")


def test_sample_queries():
    """Test with sample queries"""
    tester = RAGQueryTester()
    tester.load_components()
    
    # Sample queries - customize these based on your documents
    sample_queries = [
        "What is the main topic of the document?",
        "Summarize the key points",
        "What are the important findings?",
    ]
    
    print("\n" + "="*70)
    print("🧪 TESTING WITH SAMPLE QUERIES")
    print("="*70)
    print(f"Running {len(sample_queries)} test queries...")
    
    results = tester.batch_query(sample_queries)
    
    # Save results
    tester.save_results(results, "test_query_results.json")
    
    return results


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "test":
            # Test mode with sample queries
            test_sample_queries()
            
        elif mode == "batch":
            # Batch mode - expects queries as additional arguments
            tester = RAGQueryTester()
            tester.load_components()
            
            queries = sys.argv[2:]
            if not queries:
                print("❌ No queries provided")
                print("Usage: python test_query.py batch 'query1' 'query2' ...")
                sys.exit(1)
            
            results = tester.batch_query(queries)
            tester.save_results(results)
            
        elif mode == "interactive" or mode == "i":
            # Interactive mode
            tester = RAGQueryTester()
            tester.load_components()
            tester.interactive_mode()
            
        else:
            print(f"❌ Unknown mode: {mode}")
            print("\nUsage: python test_query.py [mode]")
            print("Modes:")
            print("  test        - Run sample test queries")
            print("  interactive - Interactive query mode (or: i)")
            print("  batch       - Batch query mode")
            print("  (no args)   - Interactive mode (default)")
            sys.exit(1)
    else:
        # Default: interactive mode
        tester = RAGQueryTester()
        tester.load_components()
        tester.interactive_mode()


if __name__ == "__main__":
    main()

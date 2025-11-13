"""
Test Ground Truth Evaluation
"""
from evaluation.backend_dashboard.api import BackendDashboard
import os

def test_ground_truth_evaluation():
    """Test ground truth evaluation with current setup."""

    # Initialize backend
    backend = BackendDashboard()

    # Get ground truth data
    rows = backend.get_ground_truth_list(limit=5)
    print(f"Found {len(rows)} ground truth rows")

    # Check vector store
    vector_files = [f for f in os.listdir('data/vectors') if f.endswith('.faiss') or f.endswith('.pkl')]
    has_vector_store = len(vector_files) > 0
    print(f"Vector store exists: {has_vector_store}")

    # Test parameters
    embedder_choice = "ollama"
    reranker_choice = "none"
    use_qem = True

    # Run evaluation for each question
    for idx, row in enumerate(rows, start=1):
        gt_id = row.get('id')
        question = row.get('question')
        expected_answer = row.get('answer')

        print(f"\n=== Running {idx}/{len(rows)} | id={gt_id} ===")
        print(f"Question: {question}")

        try:
            if not has_vector_store:
                print("⚠️ No vector store found - skipping retrieval")
                print("Retrieved chunks: 0")
                print("Context: (empty)")
                print("Sources: []")
                continue

            # Call the same retrieval method used by RAG pipeline
            from pipeline.rag_pipeline import RAGPipeline
            from pipeline.retrieval.retrieval_service import RAGRetrievalService
            from query_enhancement.query_processor import create_query_processor

            # Parse embedder type (same logic as retrieval_orchestrator)
            def _parse_embedder_type(embedder_type: str):
                from embedders.embedder_type import EmbedderType
                if embedder_type.lower() in ["e5_large_instruct", "e5_base", "gte_multilingual_base",
                                           "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"]:
                    return EmbedderType.HUGGINGFACE, False
                elif embedder_type == "huggingface_api":
                    return EmbedderType.HUGGINGFACE, True
                elif embedder_type == "huggingface_local":
                    return EmbedderType.HUGGINGFACE, False
                else:  # ollama and others
                    return EmbedderType.OLLAMA, False

            embedder_enum, use_api = _parse_embedder_type(embedder_choice)

            # Initialize pipeline (reuse cached if available)
            pipeline = RAGPipeline(embedder_type=embedder_enum, hf_use_api=use_api)

            # Override embedder for specific multilingual models
            if embedder_choice.lower() in ["e5_large_instruct", "e5_base", "gte_multilingual_base",
                                         "paraphrase_mpnet_base_v2", "paraphrase_minilm_l12_v2"]:
                from embedders.embedder_factory import EmbedderFactory
                factory = EmbedderFactory()

                if embedder_choice.lower() == "e5_large_instruct":
                    pipeline.embedder = factory.create_e5_large_instruct(device="cpu")
                elif embedder_choice.lower() == "e5_base":
                    pipeline.embedder = factory.create_e5_base(device="cpu")
                elif embedder_choice.lower() == "gte_multilingual_base":
                    pipeline.embedder = factory.create_gte_multilingual_base(device="cpu")
                elif embedder_choice.lower() == "paraphrase_mpnet_base_v2":
                    pipeline.embedder = factory.create_paraphrase_mpnet_base_v2(device="cpu")
                elif embedder_choice.lower() == "paraphrase_minilm_l12_v2":
                    pipeline.embedder = factory.create_paraphrase_minilm_l12_v2(device="cpu")

            # Create retriever service
            retriever = RAGRetrievalService(pipeline)

            # Query Enhancement (same as RAG pipeline)
            query_processor = create_query_processor(use_qem, pipeline.embedder)
            expanded_queries = query_processor.enhance_query(question, use_qem)

            # Create fused embedding
            fused_embedding = query_processor.fuse_query_embeddings(expanded_queries)
            bm25_query = " ".join(expanded_queries).strip()

            # Hybrid retrieval
            results = retriever.retrieve_hybrid(
                query_text=question,
                top_k=10,
                query_embedding=fused_embedding,
                bm25_query=bm25_query,
            )

            context = retriever.build_context(results, max_chars=2000)
            sources = retriever.to_ui_items(results)

            total_retrieved = len(results)

            print(f"Retrieved chunks: {total_retrieved}")
            print(f"Context (truncated 200 chars): {context[:200]}")
            print(f"Sources: {sources}")

        except Exception as e:
            print(f"Error processing id={gt_id}: {e}")

    print("\n✅ Ground truth evaluation test completed")

if __name__ == "__main__":
    test_ground_truth_evaluation()
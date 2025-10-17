# Repository Structure

Generated directory tree for the current project workspace:

```text
D:.
|   .gitattributes
|   .gitignore
|   pyproject.toml
|   RAG_QUICKSTART.md
|   README.md
|   README_STRUCTURE.md
|   requirements.txt
|   run_pipeline.py
|   
+---.github
|       copilot-instructions.md
|       
+---.streamlit
|       config.toml
|       config.toml.example
|       secrets.toml
|       
+---chunkers
|   |   base_chunker.py
|   |   block_aware_chunker.py
|   |   chunk_pdf_demo.py
|   |   fixed_size_chunker.py
|   |   hybrid_chunker.py
|   |   rule_based_chunker.py
|   |   semantic_chunker.py
|   |   test_fixed_size_chunker.py
|   |   __init__.py
|   |   
|   +---model
|   |   |   block_span.py
|   |   |   chunk.py
|   |   |   chunk_set.py
|   |   |   chunk_stats.py
|   |   |   enums.py
|   |   |   provenance_agg.py
|   |   |   score.py
|   |   |   __init__.py
|   |   |   
|   |   \---__pycache__
|   |           block_span.cpython-313.pyc
|   |           chunk.cpython-313.pyc
|   |           chunk_set.cpython-313.pyc
|   |           chunk_stats.cpython-313.pyc
|   |           enums.cpython-313.pyc
|   |           provenance_agg.cpython-313.pyc
|   |           score.cpython-313.pyc
|   |           __init__.cpython-313.pyc
|   |           
|   \---__pycache__
|           base_chunker.cpython-313.pyc
|           fixed_size_chunker.cpython-313.pyc
|           hybrid_chunker.cpython-313.pyc
|           rule_based_chunker.cpython-313.pyc
|           semantic_chunker.cpython-313.pyc
|           __init__.cpython-313.pyc
|           
+---config
|       app.yaml
|       
+---data
|   |   batch_summary_20251017_095950.json
|   |   
|   +---cache
|   |       processed_chunks.json
|   |       
|   +---chunks
|   |       medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_chunks_20251017_095750.txt
|   |       
|   +---embeddings
|   |       medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_embeddings_20251017_095750.json
|   |       
|   +---metadata
|   |       medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_summary_20251017_095750.json
|   |       
|   +---pdf
|   |       medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy.pdf
|   |       
|   \---vectors
|           medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_metadata_map_20251017_095750.pkl
|           medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_vectors_20251017_095750.faiss
|           
+---embedders
|   |   embedder_factory.py
|   |   embedder_type.py
|   |   i_embedder.py
|   |   __init__.py
|   |   
|   +---model
|   |   |   embedding_profile.py
|   |   |   embed_request.py
|   |   |   
|   |   \---__pycache__
|   |           embedding_profile.cpython-313.pyc
|   |           
|   +---providers
|   |   |   base_embedder.py
|   |   |   ollama_embedder.py
|   |   |   __init__.py
|   |   |   
|   |   +---ollama
|   |   |   |   base_ollama_embedder.py
|   |   |   |   bge3_embedder.py
|   |   |   |   gemma_embedder.py
|   |   |   |   model_switcher.py
|   |   |   |   __init__.py
|   |   |   |   
|   |   |   \---__pycache__
|   |   |           base_ollama_embedder.cpython-313.pyc
|   |   |           bge3_embedder.cpython-313.pyc
|   |   |           gemma_embedder.cpython-313.pyc
|   |   |           model_switcher.cpython-313.pyc
|   |   |           __init__.cpython-313.pyc
|   |   |           
|   |   \---__pycache__
|   |           base_embedder.cpython-313.pyc
|   |           ollama_embedder.cpython-313.pyc
|   |           __init__.cpython-313.pyc
|   |           
|   \---__pycache__
|           embedder_factory.cpython-313.pyc
|           embedder_type.cpython-313.pyc
|           i_embedder.cpython-313.pyc
|           __init__.cpython-313.pyc
|           
+---llm
|       chat_handler.py
|       chat_styles.css
|       config_loader.py
|       LLM_API.py
|       LLM_FE.py
|       LLM_LOCAL.py
|       readme
|       
+---loaders
|   |   config.py
|   |   ids.py
|   |   pdf_loader.py
|   |   __init__.py
|   |   
|   +---model
|   |   |   base.py
|   |   |   block.py
|   |   |   document.py
|   |   |   page.py
|   |   |   table.py
|   |   |   text.py
|   |   |   __init__.py
|   |   |   
|   |   \---__pycache__
|   |           base.cpython-313.pyc
|   |           block.cpython-313.pyc
|   |           document.cpython-313.pyc
|   |           page.cpython-313.pyc
|   |           table.cpython-313.pyc
|   |           text.cpython-313.pyc
|   |           __init__.cpython-313.pyc
|   |           
|   +---normalizers
|   |   |   analyze_duplicates.py
|   |   |   block_utils.py
|   |   |   check_block_duplicates.py
|   |   |   check_dots.py
|   |   |   review_lost_blocks.py
|   |   |   spacy_utils.py
|   |   |   table_utils.py
|   |   |   text_utils.py
|   |   |   
|   |   \---__pycache__
|   |           block_utils.cpython-313.pyc
|   |           spacy_utils.cpython-313.pyc
|   |           table_utils.cpython-313.pyc
|   |           text_utils.cpython-313.pyc
|   |           
|   \---__pycache__
|           ids.cpython-313.pyc
|           pdf_loader.cpython-313.pyc
|           __init__.cpython-313.pyc
|           
\---pipeline
    |   chunk_cache_manager.py
    |   data_integrity_checker.py
    |   data_quality_analyzer.py
    |   pipeline_qa.py
    |   query_expander.py
    |   rag_pipeline.py
    |   retriever.py
    |   summary_generator.py
    |   vector_store.py
    |   __init__.py
    |   
    \---__pycache__
            chunk_cache_manager.cpython-313.pyc
            data_integrity_checker.cpython-313.pyc
            data_quality_analyzer.cpython-313.pyc
            rag_pipeline.cpython-313.pyc
            retriever.cpython-313.pyc
            summary_generator.cpython-313.pyc
            vector_store.cpython-313.pyc
            __init__.cpython-313.pyc
```

"""
RAG Pipeline - Refactored Implementation
=========================================
Orchestrates: PDF -> Chunks -> Embeddings -> Vector Storage -> Retrieval
Single Responsibility: Pipeline coordination (delegates to specialized modules).

Output: All data saved to data folder
"""
 
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from embedders.embedder_factory import EmbedderFactory
from embedders.embedder_type import EmbedderType
from embedders.providers.ollama import OllamaModelSwitcher, OllamaModelType
from pipeline.storage.vector_store import VectorStore
from pipeline.storage.summary_generator import SummaryGenerator
from pipeline.retrieval.retriever import Retriever
from BM25.bm25_manager import BM25Manager
from pipeline.processing.pdf_processor import create_pdf_processor, PDFProcessor
from pipeline.processing.embedding_processor import EmbeddingProcessor
from pipeline.storage.file_manager import FileManager, BatchSummaryManager

# Configure logging FIRST before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



logger = logging.getLogger(__name__)
 
 
class RAGPipeline:
    """
    Complete RAG Pipeline implementation.
    Single Responsibility: Orchestrate full PDF → Vector Storage workflow.
    """
   
    def __init__(self,
                 output_dir: str = "data",
                 pdf_dir: Optional[str | Path] = None,
                 embedder_type: EmbedderType = EmbedderType.OLLAMA,
                 model_type: OllamaModelType = OllamaModelType.GEMMA,
                 hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hf_use_api: Optional[bool] = None,
                 hf_api_token: Optional[str] = None):
        """
        Initialize RAG Pipeline.
       
        Args:
            output_dir: Directory để lưu output files
            pdf_dir: Directory chứa PDF files (default: output_dir/pdf)
            embedder_type: Type of embedder to use (OLLAMA or HUGGINGFACE)
            model_type: Ollama model type (GEMMA hoặc BGE_M3) - only used for OLLAMA
            hf_model_name: HF model name - only used for HUGGINGFACE
            hf_use_api: Whether to use HF API - only used for HUGGINGFACE
            hf_api_token: HF API token - only used for HUGGINGFACE
        """
        self.output_dir = Path(output_dir)
        self.pdf_dir = Path(pdf_dir) if pdf_dir else self.output_dir / "pdf"
        self.embedder_type = embedder_type
        self.model_type = model_type
        self.hf_model_name = hf_model_name
        self.hf_use_api = hf_use_api
        self.hf_api_token = hf_api_token
       
        # Create output subdirectories
        self.chunks_dir = self.output_dir / "chunks"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.vectors_dir = self.output_dir / "vectors"
        self.metadata_dir = self.output_dir / "metadata"
        self.cache_dir = self.output_dir / "cache"
       
        # Create all directories
        for directory in [self.chunks_dir, self.embeddings_dir,
                         self.vectors_dir, self.metadata_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
       
        # Initialize cache for processed chunks
        self.processed_chunks_cache = self.cache_dir / "processed_chunks.json"
       
        # Initialize components
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize PDF processor (handles loading + chunking)
        self.pdf_processor = create_pdf_processor()
        self.loader = self.pdf_processor.loader
        self.chunker = self.pdf_processor.chunker
        self.chunker_provider = self.pdf_processor.chunker_provider

        # Initialize embedder based on type
        if embedder_type == EmbedderType.OLLAMA:
            # Initialize embedder with model switcher
            self.model_switcher = OllamaModelSwitcher()
            if model_type == OllamaModelType.GEMMA:
                self.embedder = self.model_switcher.switch_to_gemma()
            else:
                self.embedder = self.model_switcher.switch_to_bge_m3()
        elif embedder_type == EmbedderType.HUGGINGFACE:
            # Initialize HuggingFace embedder
            factory = EmbedderFactory()
            
            # Determine which embedder to use
            if hf_use_api is True:
                # Use API embedder
                self.embedder = factory.create_huggingface_api(
                    model_name=hf_model_name,
                    api_token=hf_api_token
                )
            elif hf_use_api is False:
                # Use local embedder
                self.embedder = factory.create_huggingface_local(
                    model_name=hf_model_name,
                    device="cpu"
                )
            else:
                # Auto-detect: prefer local, fallback to API
                try:
                    self.embedder = factory.create_huggingface_local(
                        model_name=hf_model_name,
                        device="cpu"
                    )
                except ImportError:
                    logger.info("Transformers not available, using HF API")
                    self.embedder = factory.create_huggingface_api(
                        model_name=hf_model_name,
                        api_token=hf_api_token
                    )
            
            self.model_switcher = None  # Not used for HF
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")
       
        # Initialize supporting components
        self.vector_store = VectorStore(self.vectors_dir)
        self.summary_generator = SummaryGenerator(self.metadata_dir, self.output_dir)
        self.retriever = Retriever(self.embedder)
        
        # Initialize BM25 manager
        self.bm25_manager = BM25Manager(self.output_dir, self.cache_dir)
        
        # Initialize file manager
        self.file_manager = FileManager(
            self.chunks_dir,
            self.embeddings_dir,
            self.vectors_dir,
            self.metadata_dir,
        )

        logger.info("=== RAG Pipeline Configuration ===")
        logger.info("Loader: PDFProvider (OCR=auto, multilingual)")
        logger.info(f"Chunker: SemanticChunker ({self.chunker.max_tokens} tokens, {self.chunker.overlap_tokens} overlap)")
        logger.info("Provider: SemanticChunkerProvider (normalize+entities+language)")
        logger.info(f"Embedder: {self.embedder.profile.model_id}")
        logger.info(f"Dimension: {self.embedder.dimension}")
        logger.info(f"BM25: {'Available' if self.bm25_manager.is_available() else 'Unavailable'}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("==================================")

    def switch_model(self, model_type: OllamaModelType) -> None:
        """
        Switch the embedding model.
       
        Args:
            model_type: New model type to switch to
        """
        if self.embedder_type != EmbedderType.OLLAMA:
            raise ValueError("Model switching only supported for OLLAMA embedders")
            
        if self.model_switcher is None:
            raise RuntimeError("Model switcher not initialized")
            
        if model_type == OllamaModelType.GEMMA:
            self.embedder = self.model_switcher.switch_to_gemma()
        else:
            self.embedder = self.model_switcher.switch_to_bge_m3()
       
        self.model_type = model_type
        logger.info(f"Switched to model: {self.embedder.profile.model_id}")
   
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the current pipeline configuration.
       
        Returns:
            Dict with pipeline information
        """
        return {
            "output_dir": str(self.output_dir),
            "pdf_dir": str(self.pdf_dir),
            "model_type": self.model_type.value,
            "embedder_model": self.embedder.profile.model_id,
            "embedder_dimension": self.embedder.dimension,
            "loader": {
                "type": "PDFProvider",
                "features": ["OCR (auto)", "Layout detection", "Table extraction", "Figure extraction"],
                "ocr_lang": "multilingual"
            },
            "chunker": {
                "type": "SemanticChunker",
                "max_tokens": 500,
                "overlap_tokens": 50
            },
            "provider": {
                "type": "SemanticChunkerProvider",
                "features": ["Text normalization", "Entity extraction", "Language detection", "POS tagging"]
            },
            "vector_store": "FAISS (cosine similarity)",
            "cache_enabled": True
        }

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 5,
        normalize_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute BM25 search (delegates to BM25Manager).
        
        Args:
            query: Search query string
            top_k: Number of results to return
            normalize_scores: Whether to normalize scores
            
        Returns:
            List of result dictionaries with metadata
        """
        return self.bm25_manager.search(
            query,
            top_k=top_k,
            normalize_scores=normalize_scores,
        )

    def process_pdf(self, pdf_path: str | Path, chunk_callback=None) -> Dict[str, Any]:
        """
        Process single PDF through complete pipeline (orchestrates all steps).
       
        Args:
            pdf_path: Path to PDF file (str or Path)
            chunk_callback: Optional callback function(current, total) for progress

        Returns:
            Dict with processing results and file paths
        """
        # Step 1: Validate PDF path
        pdf_path = PDFProcessor.validate_pdf_path(pdf_path)
        file_name = pdf_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Step 2: Load PDF and create chunks (delegates to PDFProcessor)
        pdf_doc, chunk_set = self.pdf_processor.process(pdf_path)

        # Step 3: Index chunks with BM25 (delegates to BM25Manager)
        bm25_indexed = self.bm25_manager.ingest_chunk_set(chunk_set)

        # Step 4: Generate embeddings (delegates to EmbeddingProcessor)
        logger.info("Generating embeddings...")
        embedding_processor = EmbeddingProcessor(self.embedder, timestamp)
        embeddings_data, skipped_chunks = embedding_processor.process_chunks(
            chunk_set, pdf_path, chunk_callback
        )
       
        # Step 5: Save chunks and embeddings (delegates to FileManager)
        logger.info("Saving chunks and embeddings...")
        chunks_file = self.file_manager.save_chunks(
            chunk_set, embeddings_data, file_name, timestamp, pdf_path.name, skipped_chunks
        )
        embeddings_file = self.file_manager.save_embeddings(
            embeddings_data, file_name, timestamp
        )
       
        # Step 6: Create FAISS index (only if we have embeddings)
        if embeddings_data:
            logger.info("Creating FAISS vector index...")
            faiss_file, metadata_map_file = self.vector_store.create_index(
                embeddings_data, file_name, timestamp
            )
        else:
            logger.info("No new embeddings - skipping FAISS index creation")
            faiss_file, metadata_map_file = self.file_manager.create_placeholder_files(
                file_name, timestamp
            )
       
        # Step 5: Save document summary (lightweight)
        logger.info("Creating document summary...")
        summary = self.summary_generator.create_document_summary(
            pdf_doc, chunk_set, embeddings_data, faiss_file, metadata_map_file
        )
       
        summary_file = self.summary_generator.save_document_summary(
            summary, file_name, timestamp
        )

        # Summary
        logger.info(
            "Pipeline completed - Pages: %d, Chunks: %d, Embeddings: %d, Skipped: %d, BM25 indexed: %d",
            len(pdf_doc.pages),
            len(chunk_set.chunks),
            len(embeddings_data),
            skipped_chunks,
            bm25_indexed,
        )

        return {
            "success": True,
            "file_name": pdf_path.name,
            "pages": len(pdf_doc.pages),
            "chunks": len(chunk_set.chunks),
            "embeddings": len(embeddings_data),
            "skipped_chunks": skipped_chunks,
            "bm25_indexed": bm25_indexed,
            "dimension": self.embedder.dimension,
            "files": {
                "chunks": str(chunks_file),
                "embeddings": str(embeddings_file),
                "faiss_index": str(faiss_file),
                "metadata_map": str(metadata_map_file),
                "summary": str(summary_file)
            }
        }
   
    def process_directory(self, pdf_dir: Optional[str | Path] = None) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory.
       
        Args:
            pdf_dir: Directory containing PDFs (default: self.pdf_dir)
           
        Returns:
            List of processing results
        """
        if pdf_dir is None:
            pdf_dir = self.pdf_dir
        else:
            pdf_dir = Path(pdf_dir)
       
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
       
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        # Convert to absolute paths
        pdf_files = [f.resolve() for f in pdf_files]
       
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []
       
        logger.info(f"Found {len(pdf_files)} PDF file(s)")
       
        results = []
        for idx, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"Processing {idx}/{len(pdf_files)}: {pdf_file.name}")
           
            try:
                result = self.process_pdf(str(pdf_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                results.append({
                    "success": False,
                    "file_name": pdf_file.name,
                    "error": str(e)
                })
       
        # Create and save batch summary (delegates to BatchSummaryManager)
        batch_summary_manager = BatchSummaryManager(self.output_dir)
        batch_summary = batch_summary_manager.create_summary(results)
        batch_summary_manager.save_summary(batch_summary)
       
        return results
   
    def load_index(self, faiss_file: Path, metadata_map_file: Path) -> tuple:
        """
        Load existing FAISS index and metadata map.
       
        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file
           
        Returns:
            Tuple of (faiss_index, metadata_map)
        """
        return self.vector_store.load_index(faiss_file, metadata_map_file)
   
    def search_similar(
        self,
        faiss_file: Path,
        metadata_map_file: Path,
        query_text: str | None,
        top_k: int = 10,
        *,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using FAISS index.
       
        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file
            query_text: Query text to search (optional when query_embedding provided)
            top_k: Number of results to return
            query_embedding: Optional precomputed embedding to reuse
       
        Returns:
            List of similar chunks with metadata and distances
        """
        return self.retriever.search_similar(
            faiss_file=faiss_file,
            metadata_map_file=metadata_map_file,
            query_text=query_text,
            top_k=top_k,
            query_embedding=query_embedding,
        )

def main():
    """Main entry point for RAG Pipeline."""
    logger.info("Starting RAG Pipeline")
    

    # Initialize pipeline với Gemma embedder
    pipeline = RAGPipeline(
        output_dir="data",
        model_type=OllamaModelType.GEMMA
    )

    logger.info("RAG Pipeline initialized and ready to use")
    
    # Process all PDFs in data/pdf directory
    try:
        pipeline.process_directory()
        
        logger.info("All processing completed successfully")
        logger.info(f"Output files saved to: {pipeline.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


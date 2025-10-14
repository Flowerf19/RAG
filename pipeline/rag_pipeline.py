"""
RAG Pipeline - Complete Implementation
========================================
Pipeline hoàn chỉnh: PDF -> Chunks -> Embeddings -> Vector Storage -> Retrieval

Output: Tất cả dữ liệu được lưu vào data folder
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from loaders.pdf_loader import PDFLoader
from chunkers.hybrid_chunker import HybridChunker
from embedders.embedder_factory import EmbedderFactory
from embedders.providers.ollama import OllamaModelSwitcher, OllamaModelType
from pipeline.vector_store import VectorStore
from pipeline.vector_store import VectorStore
from pipeline.summary_generator import SummaryGenerator
from pipeline.retriever import Retriever

# Configure logging
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
                 output_dir: str = r"C:\Users\ENGUYEHWC\Prototype\Version_4\data",
                 model_type: OllamaModelType = OllamaModelType.GEMMA):
        """
        Initialize RAG Pipeline.
        
        Args:
            output_dir: Directory để lưu output files
            model_type: Ollama model type (GEMMA hoặc BGE_M3)
        """
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        
        # Create output subdirectories
        self.chunks_dir = self.output_dir / "chunks"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.vectors_dir = self.output_dir / "vectors"
        self.metadata_dir = self.output_dir / "metadata"
        
        # Create all directories
        for directory in [self.chunks_dir, self.embeddings_dir, 
                         self.vectors_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing RAG Pipeline...")
        self.loader = PDFLoader.create_default()
        self.chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
        
        # Initialize embedder with model switcher
        self.model_switcher = OllamaModelSwitcher()
        if model_type == OllamaModelType.GEMMA:
            self.embedder = self.model_switcher.switch_to_gemma()
        else:
            self.embedder = self.model_switcher.switch_to_bge_m3()
        
        # Initialize supporting components
        self.vector_store = VectorStore(self.vectors_dir)
        self.summary_generator = SummaryGenerator(self.metadata_dir, self.output_dir)
        self.retriever = Retriever(self.embedder)
        
        logger.info(f"Loader: PDFLoader")
        logger.info(f"Chunker: HybridChunker (max_tokens=200)")
        logger.info(f"Embedder: {self.embedder.profile.model_id}")
        logger.info(f"Dimension: {self.embedder.dimension}")
        logger.info(f"Output: {self.output_dir}")
    
    def process_pdf(self, pdf_path: str | Path) -> Dict[str, Any]:
        """
        Process single PDF through complete pipeline.
        
        Args:
            pdf_path: Path to PDF file (str or Path)
            
        Returns:
            Dict with processing results and file paths
        """
        pdf_path = Path(pdf_path)
        file_name = pdf_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Step 1: Load PDF
        logger.info("Loading PDF...")
        pdf_doc = self.loader.load(str(pdf_path))
        pdf_doc = pdf_doc.normalize()
        logger.info(f"Loaded {len(pdf_doc.pages)} pages, {sum(len(p.blocks) for p in pdf_doc.pages)} blocks")
        
        # Step 2: Chunk document
        logger.info("Chunking document...")
        chunk_set = self.chunker.chunk(pdf_doc)
        logger.info(f"Created {len(chunk_set.chunks)} chunks, strategy: {chunk_set.chunk_strategy}, tokens: {chunk_set.total_tokens}")
        
        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")
        embeddings_data = []
        
        for idx, chunk in enumerate(chunk_set.chunks, 1):
            # Test connection on first chunk
            if idx == 1 and not self.embedder.test_connection():
                raise ConnectionError("Cannot connect to Ollama server!")
            
            try:
                embedding = self.embedder.embed(chunk.text)
            except Exception as e:
                logger.warning(f"Error embedding chunk {idx}: {e}")
                embedding = [0.0] * self.embedder.dimension
            
            # Prepare embedding data với full metadata
            chunk_embedding = {
                "chunk_id": chunk.chunk_id,
                "chunk_index": idx - 1,
                "text": chunk.text,
                "text_length": len(chunk.text),
                "token_count": chunk.token_count,
                
                # Embedding vector
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "embedding_model": self.embedder.profile.model_id,
                
                # Source metadata
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "page_number": list(chunk.provenance.page_numbers)[0] if chunk.provenance and chunk.provenance.page_numbers else None,
                "page_numbers": sorted(list(chunk.provenance.page_numbers)) if chunk.provenance else [],
                
                # Block tracing
                "block_type": chunk.metadata.get("block_type") or chunk.metadata.get("type"),
                "block_ids": chunk.provenance.source_blocks if chunk.provenance else [],
                
                # Table detection
                "is_table": chunk.metadata.get("block_type") == "table",
                
                # Provenance
                "provenance": {
                    "source_file": str(pdf_path),
                    "extraction_method": "PDFLoader",
                    "chunking_strategy": chunk_set.chunk_strategy or "unknown",
                    "embedding_model": self.embedder.profile.model_id,
                    "timestamp": timestamp
                }
            }
            
            # Add table data if applicable
            if chunk_embedding["is_table"]:
                table_payload = chunk.metadata.get("table_payload")
                if table_payload:
                    chunk_embedding["table_data"] = {
                        "table_id": getattr(table_payload, "id", None),
                        "header": getattr(table_payload, "header", []),
                        "num_rows": len(getattr(table_payload, "rows", [])),
                        "page_number": getattr(table_payload, "page_number", None)
                    }
            
            embeddings_data.append(chunk_embedding)
            
            if idx % 10 == 0:
                logger.info(f"Processed {idx}/{len(chunk_set.chunks)} chunks...")
        
        logger.info(f"Generated {len(embeddings_data)} embeddings")
        
        # Step 4: Create FAISS index and save
        logger.info("Creating FAISS vector index...")
        faiss_file, metadata_map_file = self.vector_store.create_index(
            embeddings_data, file_name, timestamp
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
        logger.info(f"Pipeline completed - Pages: {len(pdf_doc.pages)}, Chunks: {len(chunk_set.chunks)}, Embeddings: {len(embeddings_data)}")
        
        return {
            "success": True,
            "file_name": pdf_path.name,
            "pages": len(pdf_doc.pages),
            "chunks": len(chunk_set.chunks),
            "embeddings": len(embeddings_data),
            "dimension": self.embedder.dimension,
            "files": {
                "faiss_index": str(faiss_file),
                "metadata_map": str(metadata_map_file),
                "summary": str(summary_file)
            }
        }
    
    def process_directory(self, pdf_dir: Optional[str | Path] = None) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDFs (default: data/pdf)
            
        Returns:
            List of processing results
        """
        if pdf_dir is None:
            pdf_dir = self.output_dir / "pdf"
        else:
            pdf_dir = Path(pdf_dir)
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
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
        
        # Create summary
        batch_summary = self.summary_generator.create_batch_summary(results)
        self.summary_generator.save_batch_summary(batch_summary)
        
        return results
    
    def search_similar(self, faiss_file: Path, metadata_map_file: Path,
                      query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using FAISS index.
        
        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file
            query_text: Query text to search
            top_k: Number of results to return
            
        Returns:
            List of similar chunks with metadata and distances
        """
        return self.retriever.search_similar(faiss_file, metadata_map_file, query_text, top_k)


def main():
    """Main entry point for RAG Pipeline."""
    logger.info("Starting RAG Pipeline")
    
    # Initialize pipeline với Gemma embedder
    pipeline = RAGPipeline(
        output_dir=r"C:\Users\ENGUYEHWC\Prototype\Version_4\data",
        model_type=OllamaModelType.GEMMA
    )
    
    # Process all PDFs in data/pdf directory
    try:
        results = pipeline.process_directory()
        
        logger.info("All processing completed successfully")
        logger.info(f"Output files saved to: {pipeline.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

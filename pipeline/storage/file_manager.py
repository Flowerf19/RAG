"""
File Manager Module
===================
Handles file I/O operations for chunks, embeddings, and summaries.
Single Responsibility: Save and manage RAG pipeline output files.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class FileManager:
    """
    Manages file I/O operations for RAG pipeline outputs.
    """

    def __init__(
        self,
        chunks_dir: Path,
        embeddings_dir: Path,
        vectors_dir: Path,
        metadata_dir: Path,
    ):
        """
        Initialize file manager.
        
        Args:
            chunks_dir: Directory for chunk text files
            embeddings_dir: Directory for embedding JSON files
            vectors_dir: Directory for FAISS index files
            metadata_dir: Directory for metadata files
        """
        self.chunks_dir = chunks_dir
        self.embeddings_dir = embeddings_dir
        self.vectors_dir = vectors_dir
        self.metadata_dir = metadata_dir

    def save_chunks(
        self,
        chunk_set: Any,
        embeddings_data: List[Dict[str, Any]],
        file_name: str,
        timestamp: str,
        pdf_name: str,
        skipped_chunks: int = 0,
    ) -> Path:
        """
        Save chunks to text file for debugging.
        
        Args:
            chunk_set: ChunkSet object
            embeddings_data: List of embedding data dicts
            file_name: Base file name (stem)
            timestamp: Timestamp string
            pdf_name: Original PDF filename
            skipped_chunks: Number of skipped chunks
            
        Returns:
            Path to saved chunks file
        """
        # Delete old chunk files for this PDF (overwrite mode)
        for old_file in self.chunks_dir.glob(f"{file_name}_chunks_*.txt"):
            try:
                old_file.unlink()
                logger.debug(f"Deleted old chunk file: {old_file.name}")
            except Exception:
                pass

        chunks_file = self.chunks_dir / f"{file_name}_chunks_{timestamp}.txt"

        with open(chunks_file, 'w', encoding='utf-8') as f:
            f.write(f'Document: {pdf_name}\n')
            f.write(f'Total chunks: {len(chunk_set.chunks)}\n')
            f.write(f'New chunks processed: {len(embeddings_data)}\n')
            f.write(f'Skipped chunks: {skipped_chunks}\n')
            f.write(f'Timestamp: {timestamp}\n')
            f.write('=' * 80 + '\n\n')

            for i, chunk in enumerate(chunk_set.chunks, 1):
                # Only write chunks that were actually processed (have embeddings)
                if any(e['chunk_id'] == chunk.chunk_id for e in embeddings_data):
                    f.write(f'CHUNK {i}: {chunk.chunk_id}\n')
                    page = (
                        list(chunk.provenance.page_numbers)[0]
                        if chunk.provenance and chunk.provenance.page_numbers
                        else 'N/A'
                    )
                    f.write(
                        f'Page: {page} | Tokens: {chunk.token_count} | Type: {chunk.chunk_type.value}\n'
                    )
                    f.write('-' * 40 + '\n')
                    f.write(chunk.text.strip())
                    f.write('\n\n' + '=' * 80 + '\n\n')

        logger.info(f"Saved chunks to: {chunks_file}")
        return chunks_file

    def save_embeddings(
        self,
        embeddings_data: List[Dict[str, Any]],
        file_name: str,
        timestamp: str,
    ) -> Path:
        """
        Save embeddings to JSON file.
        
        Args:
            embeddings_data: List of embedding data dicts
            file_name: Base file name (stem)
            timestamp: Timestamp string
            
        Returns:
            Path to saved embeddings file
        """
        embeddings_file = self.embeddings_dir / f"{file_name}_embeddings_{timestamp}.json"

        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved embeddings to: {embeddings_file}")
        return embeddings_file

    def create_placeholder_files(self, file_name: str, timestamp: str) -> tuple[Path, Path]:
        """
        Create placeholder FAISS and metadata files for skipped processing.
        
        Args:
            file_name: Base file name (stem)
            timestamp: Timestamp string
            
        Returns:
            Tuple of (faiss_file path, metadata_map_file path)
        """
        faiss_file = self.vectors_dir / f"{file_name}_vectors_{timestamp}.faiss"
        metadata_map_file = self.vectors_dir / f"{file_name}_metadata_map_{timestamp}.pkl"

        # Create empty files to indicate processing was attempted
        faiss_file.touch()
        metadata_map_file.touch()

        return faiss_file, metadata_map_file


class BatchSummaryManager:
    """
    Manages batch processing summaries.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize batch summary manager.
        
        Args:
            output_dir: Root output directory
        """
        self.output_dir = output_dir

    def create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create batch processing summary from individual results.
        
        Args:
            results: List of processing result dicts
            
        Returns:
            Batch summary dictionary
        """
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        return {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_files": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_pages": sum(r.get("pages", 0) for r in successful),
            "total_chunks": sum(r.get("chunks", 0) for r in successful),
            "total_embeddings": sum(r.get("embeddings", 0) for r in successful),
            "results": results,
        }

    def save_summary(self, summary: Dict[str, Any]) -> Path:
        """
        Save batch summary to JSON file.
        
        Args:
            summary: Batch summary dictionary
            
        Returns:
            Path to saved summary file
        """
        timestamp = summary.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        summary_file = self.output_dir / f"batch_summary_{timestamp}.json"

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved batch summary to: {summary_file}")
        return summary_file

"""
Clean Corrupted FAISS Indexes
==============================
Script để kiểm tra và làm sạch các FAISS indexes bị corrupt hoặc có invalid entries
"""

import logging
from pathlib import Path
import faiss
import pickle

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_faiss_index(faiss_file: Path, metadata_file: Path) -> dict:
    """
    Kiểm tra FAISS index và metadata để phát hiện vấn đề
    
    Returns:
        Dict chứa thông tin về index
    """
    try:
        # Load FAISS index
        index = faiss.read_index(str(faiss_file))
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata_map = pickle.load(f)
        
        # Get index info
        index_size = index.ntotal
        metadata_size = len(metadata_map)
        
        # Check dimension
        dimension = index.d
        
        info = {
            "file": faiss_file.name,
            "index_vectors": index_size,
            "metadata_entries": metadata_size,
            "dimension": dimension,
            "match": index_size == metadata_size,
            "valid": True
        }
        
        # Check if sizes match
        if index_size != metadata_size:
            logger.warning(f"❌ Size mismatch: {faiss_file.name}")
            logger.warning(f"   Index vectors: {index_size}, Metadata entries: {metadata_size}")
            info["valid"] = False
        else:
            logger.info(f"✅ Valid: {faiss_file.name} ({index_size} vectors, {dimension}D)")
        
        return info
        
    except Exception as e:
        logger.error(f"❌ Error checking {faiss_file.name}: {e}")
        return {
            "file": faiss_file.name,
            "error": str(e),
            "valid": False
        }


def clean_corrupted_indexes(data_dir: Path = Path("data")):
    """
    Kiểm tra tất cả FAISS indexes và report status
    """
    vectors_dir = data_dir / "vectors"
    
    if not vectors_dir.exists():
        logger.error(f"Vectors directory not found: {vectors_dir}")
        return
    
    # Find all FAISS files
    faiss_files = list(vectors_dir.glob("*_vectors_*.faiss"))
    
    if not faiss_files:
        logger.warning("No FAISS index files found")
        return
    
    logger.info(f"Found {len(faiss_files)} FAISS index files")
    logger.info("="*80)
    
    valid_indexes = []
    invalid_indexes = []
    
    for faiss_file in faiss_files:
        # Find corresponding metadata file
        metadata_file = Path(str(faiss_file).replace("_vectors_", "_metadata_map_").replace(".faiss", ".pkl"))
        
        if not metadata_file.exists():
            logger.error(f"❌ Missing metadata: {metadata_file.name}")
            invalid_indexes.append(faiss_file)
            continue
        
        # Check index
        info = check_faiss_index(faiss_file, metadata_file)
        
        if info.get("valid", False):
            valid_indexes.append(info)
        else:
            invalid_indexes.append(info)
    
    # Summary
    logger.info("="*80)
    logger.info("\n📊 Summary:")
    logger.info(f"   Valid indexes: {len(valid_indexes)}")
    logger.info(f"   Invalid indexes: {len(invalid_indexes)}")
    
    if invalid_indexes:
        logger.info("\n⚠️  Invalid indexes found:")
        for idx_info in invalid_indexes:
            if isinstance(idx_info, dict):
                logger.info(f"   - {idx_info.get('file', 'Unknown')}")
                if 'error' in idx_info:
                    logger.info(f"     Error: {idx_info['error']}")
            else:
                logger.info(f"   - {idx_info}")
        
        logger.info("\n💡 Recommendation:")
        logger.info("   Run 'python run_pipeline.py' to rebuild all indexes")
        logger.info("   Or delete corrupted files manually and re-process PDFs")
    
    return {
        "valid": valid_indexes,
        "invalid": invalid_indexes
    }


def rebuild_specific_index(pdf_file: Path):
    """
    Rebuild FAISS index cho một PDF cụ thể
    """
    logger.info(f"Rebuilding index for: {pdf_file}")
    
    from pipeline.rag_pipeline import RAGPipeline
    
    pipeline = RAGPipeline()
    
    # Process single PDF
    try:
        pipeline.process_pdf(pdf_file)
        logger.info(f"✅ Successfully rebuilt index for {pdf_file.name}")
    except Exception as e:
        logger.error(f"❌ Failed to rebuild index: {e}")


if __name__ == "__main__":
    print("\n🔍 Checking FAISS Indexes...")
    print("="*80)
    
    result = clean_corrupted_indexes()
    
    if result and len(result.get("invalid", [])) > 0:
        print("\n" + "="*80)
        print("⚠️  Some indexes are invalid or corrupted")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✅ All FAISS indexes are valid!")
        print("="*80)

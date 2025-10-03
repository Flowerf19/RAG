import hashlib
from typing import Optional, Dict, Any


def generate_chunk_id(
    content: str,
    source: str,
    page: Optional[int] = None,
    chunk_index: Optional[int] = None,
    **metadata: Any
) -> str:
    """
    Generate deterministic ID cho chunk.
    
    Args:
        content: Chunk content
        source: Source document
        page: Page number (optional)
        chunk_index: Index của chunk trong document (optional)
        **metadata: Additional metadata
    
    Returns:
        Stable chunk ID (SHA256 hash)
    """
    # Tạo string để hash
    id_parts = [
        f"content:{content}",
        f"source:{source}",
    ]
    
    if page is not None:
        id_parts.append(f"page:{page}")
    
    if chunk_index is not None:
        id_parts.append(f"index:{chunk_index}")
    
    # Add metadata (sorted để deterministic)
    for key, value in sorted(metadata.items()):
        id_parts.append(f"{key}:{value}")
    
    id_string = "|".join(id_parts)
    
    # Generate SHA256 hash
    hash_obj = hashlib.sha256(id_string.encode('utf-8'))
    return f"chunk_{hash_obj.hexdigest()[:16]}"


def generate_short_id(content: str, length: int = 8) -> str:
    """
    Generate short ID cho chunk (dùng cho display).
    
    Args:
        content: Chunk content
        length: Length of short ID
    
    Returns:
        Short ID
    """
    hash_obj = hashlib.md5(content.encode('utf-8'))
    return hash_obj.hexdigest()[:length]


def generate_citation(
    document_title: str,
    page: Optional[int] = None,
    section: Optional[str] = None,
    chunk_index: Optional[int] = None
) -> str:
    """
    Generate human-readable citation cho chunk.
    
    Args:
        document_title: Title of document
        page: Page number
        section: Section name
        chunk_index: Chunk index
    
    Returns:
        Citation string (e.g., "Document Title, p.12, Section 3.1")
    """
    parts = [document_title]
    
    if page is not None:
        parts.append(f"p.{page}")
    
    if section:
        parts.append(section)
    
    if chunk_index is not None:
        parts.append(f"chunk #{chunk_index}")
    
    return ", ".join(parts)


def generate_chunk_metadata_id(metadata: Dict[str, Any]) -> str:
    """
    Generate ID dựa trên metadata.
    
    Args:
        metadata: Chunk metadata dict
    
    Returns:
        Metadata-based ID
    """
    # Serialize metadata (sorted để deterministic)
    metadata_str = "|".join(f"{k}:{v}" for k, v in sorted(metadata.items()))
    hash_obj = hashlib.sha256(metadata_str.encode('utf-8'))
    return f"meta_{hash_obj.hexdigest()[:16]}"

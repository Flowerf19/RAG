"""
Semantic Chunker Provider
=========================
Provider cho SemanticChunker với spaCy preprocessing.
Xử lý chunks trước khi gửi vào embedder:
- Text normalization
- Entity extraction
- Language detection
- Metadata enrichment
"""

from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Import models
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import Chunk, ChunkSet


class SemanticChunkerProvider:
    """
    Provider cho SemanticChunker với enhanced preprocessing:
    - spaCy-based entity extraction
    - Language detection
    - POS tagging metadata
    - Text normalization
    """
    
    def __init__(self, 
                 normalize_text: bool = True,
                 extract_entities: bool = True,
                 detect_language: bool = True,
                 remove_stopwords: bool = False,
                 min_chunk_length: int = 10,
                 spacy_model: str = "en_core_web_sm",
                 nlp=None):
        """
        Initialize semantic provider.
        
        Args:
            normalize_text: Enable text normalization
            extract_entities: Enable entity extraction
            detect_language: Enable language detection
            remove_stopwords: Enable stopword removal
            min_chunk_length: Minimum chunk length
            spacy_model: spaCy model name
            nlp: Pre-loaded spaCy nlp object (recommended)
        """
        self.normalize_text = normalize_text
        self.extract_entities = extract_entities
        self.detect_language = detect_language
        self.remove_stopwords = remove_stopwords
        self.min_chunk_length = min_chunk_length
        self.spacy_model = spacy_model
        self._nlp: Optional[Any] = nlp
        self._nlp_loaded = False
    
    def _ensure_nlp(self):
        """Lazy load spaCy model"""
        if self._nlp is not None:
            self._nlp_loaded = True
            return
        
        try:
            import spacy
            self._nlp = spacy.load(self.spacy_model)
            self._nlp_loaded = True
        except Exception as e:
            print(f"⚠️ Warning: Could not load spaCy model '{self.spacy_model}': {e}")
            print("   Entity extraction and POS tagging will be disabled.")
            self._nlp_loaded = False
    
    def process_chunks(self, chunk_set: ChunkSet) -> ChunkSet:
        """
        Process all chunks in ChunkSet.
        
        Args:
            chunk_set: Input ChunkSet
            
        Returns:
            Processed ChunkSet
        """
        # Ensure spaCy loaded (if needed)
        if self.extract_entities or self.detect_language:
            self._ensure_nlp()
        
        # Process each chunk
        processed_chunks = []
        for chunk in chunk_set.chunks:
            processed = self.process_chunk(chunk)
            if self.validate_chunk(processed):
                processed_chunks.append(processed)
        
        # Return new ChunkSet
        return ChunkSet(
            doc_id=chunk_set.doc_id,
            chunks=processed_chunks,
            chunk_strategy=chunk_set.chunk_strategy,
            metadata={
                **chunk_set.metadata,
                'provider': 'SemanticChunkerProvider',
                'original_chunk_count': len(chunk_set.chunks),
                'processed_chunk_count': len(processed_chunks),
                'spacy_model': self.spacy_model if self._nlp_loaded else None
            }
        )
    
    def process_chunk(self, chunk: Chunk) -> Chunk:
        """
        Process single chunk.
        
        Args:
            chunk: Chunk cần process
            
        Returns:
            Processed chunk
        """
        # Preprocess text
        processed_text = self._preprocess_text(chunk.text)
        
        # Enrich metadata
        enriched_metadata = self._enrich_metadata(chunk, processed_text)
        
        # Return new chunk
        return Chunk(
            chunk_id=chunk.chunk_id,
            text=processed_text,
            doc_id=chunk.doc_id,
            chunk_type=chunk.chunk_type,
            token_count=chunk.token_count,
            char_count=len(processed_text),
            metadata=enriched_metadata,
            provenance=chunk.provenance,
            score=chunk.score
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text trước embedding.
        
        Args:
            text: Raw text
            
        Returns:
            Processed text
        """
        if not text or len(text) < self.min_chunk_length:
            return text
        
        processed = text
        
        # Normalize
        if self.normalize_text:
            processed = self._normalize(processed)
        
        # Remove stopwords
        if self.remove_stopwords:
            processed = self._remove_stopwords(processed)
        
        return processed
    
    def _normalize(self, text: str) -> str:
        """
        Normalize text: whitespace, control chars.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        return text.strip()
    
    def validate_chunk(self, chunk: Chunk) -> bool:
        """
        Validate chunk.
        
        Args:
            chunk: Chunk to validate
            
        Returns:
            True if valid
        """
        return (
            len(chunk.text) >= self.min_chunk_length and
            chunk.token_count > 0
        )
    
    def _enrich_metadata(self, chunk: Chunk, processed_text: str) -> Dict[str, Any]:
        """
        Enrich metadata with spaCy analysis.
        
        Args:
            chunk: Original chunk
            processed_text: Processed text
            
        Returns:
            Enriched metadata
        """
        metadata = chunk.metadata.copy()
        
        # Basic preprocessing info
        metadata['preprocessed'] = True
        metadata['original_length'] = chunk.char_count
        metadata['processed_length'] = len(processed_text)
        
        # Language detection
        if self.detect_language:
            metadata['language'] = self._detect_language(processed_text)
        
        # Entity extraction
        if self.extract_entities:
            metadata['entities'] = self._extract_entities(processed_text)
        
        # spaCy analysis
        if self._nlp_loaded and self._nlp is not None:
            try:
                doc = self._nlp(processed_text[:1000])
                
                # POS distribution
                pos_counts = {}
                for token in doc:
                    pos = token.pos_
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                
                metadata['pos_distribution'] = pos_counts
                metadata['token_count_spacy'] = len(doc)
                metadata['sentence_count_spacy'] = len(list(doc.sents))
                
            except Exception:
                pass
        
        return metadata
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Language code
        """
        if not self._nlp_loaded or self._nlp is None:
            return 'unknown'
        
        try:
            doc = self._nlp(text[:1000])  # Analyze first 1000 chars
            # Use spaCy's language detection or heuristics
            lang = doc.lang_ if hasattr(doc, 'lang_') else 'en'
            return lang
        except Exception:
            return 'unknown'
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of entities with labels
        """
        if not self._nlp_loaded or self._nlp is None:
            return []
        
        try:
            doc = self._nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
        except Exception:
            return []
    
    def _remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Text without stopwords
        """
        if not self._nlp_loaded or self._nlp is None:
            return text
        
        try:
            doc = self._nlp(text)
            filtered_tokens = [
                token.text for token in doc
                if not token.is_stop and not token.is_punct
            ]
            return ' '.join(filtered_tokens)
        except Exception:
            return text


# Factory function
def create_semantic_provider(
    normalize: bool = True,
    stopwords: bool = False,
    entities: bool = True,
    language: bool = True,
    min_length: int = 10,
    spacy_model: str = "en_core_web_sm",
    nlp=None
) -> SemanticChunkerProvider:
    """
    Factory function to create SemanticChunkerProvider.
    
    Args:
        normalize: Enable text normalization
        stopwords: Enable stopword removal
        entities: Enable entity extraction
        language: Enable language detection
        min_length: Minimum chunk length
        spacy_model: spaCy model name
        nlp: Pre-loaded spaCy nlp object
        
    Returns:
        SemanticChunkerProvider instance
    """
    return SemanticChunkerProvider(
        normalize_text=normalize,
        extract_entities=entities,
        detect_language=language,
        remove_stopwords=stopwords,
        min_chunk_length=min_length,
        spacy_model=spacy_model,
        nlp=nlp
    )

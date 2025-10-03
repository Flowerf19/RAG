"""
spaCy utilities for chunking.
"""
import spacy
from typing import List, Optional
import sys
import os

# Add parent directory to import loaders spacy_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from loaders.normalizers.spacy_utils import get_nlp, sent_tokenize
except ImportError:
    # Fallback implementation
    _nlp_cache = {}
    
    def get_nlp(lang='en'):
        """Get spaCy model for language."""
        global _nlp_cache
        if lang not in _nlp_cache:
            if lang == 'en':
                _nlp_cache[lang] = spacy.load('en_core_web_sm')
            elif lang == 'vi':
                _nlp_cache[lang] = spacy.load('vi_core_news_sm')
            else:
                _nlp_cache[lang] = spacy.load('en_core_web_sm')
        return _nlp_cache[lang]
    
    def sent_tokenize(text, lang='en'):
        """Tokenize text into sentences."""
        nlp = get_nlp(lang)
        doc = nlp(text)
        return [sent.text for sent in doc.sents]


class SpacyChunker:
    """
    Chunker sử dụng spaCy cho sentence và paragraph detection.
    """
    
    def __init__(self, lang: str = 'en', max_length: int = 1000000):
        """
        Initialize spaCy chunker.
        
        Args:
            lang: Language code ('en', 'vi', etc.)
            max_length: Max text length for spaCy processing
        """
        self.lang = lang
        self.nlp = get_nlp(lang)
        # Tăng max_length nếu cần process large documents
        self.nlp.max_length = max_length
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as e:
            print(f"Error in sentence splitting: {e}")
            # Fallback to simple splitting
            return self._fallback_sentence_split(text)
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Text to split
        
        Returns:
            List of paragraphs
        """
        # Split by multiple newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs
    
    def merge_sentences_to_chunks(
        self,
        sentences: List[str],
        max_chunk_size: int = 800,
        min_chunk_size: int = 100,
        respect_sentence_boundary: bool = True
    ) -> List[str]:
        """
        Merge sentences into chunks respecting size constraints.
        
        Args:
            sentences: List of sentences
            max_chunk_size: Maximum chunk size (chars)
            min_chunk_size: Minimum chunk size (chars)
            respect_sentence_boundary: Don't break sentences
        
        Returns:
            List of chunks
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If single sentence exceeds max_chunk_size
            if sentence_len > max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                if respect_sentence_boundary:
                    # Add as single chunk even if too large
                    chunks.append(sentence)
                else:
                    # Split the large sentence
                    chunks.extend(self._split_large_sentence(sentence, max_chunk_size))
                continue
            
            # Check if adding this sentence exceeds max_chunk_size
            if current_size + sentence_len + 1 > max_chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= min_chunk_size or not chunks:
                    chunks.append(chunk_text)
                else:
                    # Merge with previous chunk if too small
                    if chunks:
                        chunks[-1] = chunks[-1] + ' ' + chunk_text
                    else:
                        chunks.append(chunk_text)
                
                current_chunk = [sentence]
                current_size = sentence_len
            else:
                current_chunk.append(sentence)
                current_size += sentence_len + 1  # +1 for space
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_chunk_size or not chunks:
                chunks.append(chunk_text)
            else:
                if chunks:
                    chunks[-1] = chunks[-1] + ' ' + chunk_text
                else:
                    chunks.append(chunk_text)
        
        return chunks
    
    def merge_paragraphs_to_chunks(
        self,
        paragraphs: List[str],
        max_chunk_size: int = 800,
        min_chunk_size: int = 100
    ) -> List[str]:
        """
        Merge paragraphs into chunks.
        
        Args:
            paragraphs: List of paragraphs
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
        
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            # If paragraph is too large, split by sentences
            if para_len > max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                sentences = self.split_into_sentences(para)
                para_chunks = self.merge_sentences_to_chunks(
                    sentences,
                    max_chunk_size,
                    min_chunk_size
                )
                chunks.extend(para_chunks)
                continue
            
            # Check if adding this paragraph exceeds max
            if current_size + para_len + 2 > max_chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = [para]
                current_size = para_len
            else:
                current_chunk.append(para)
                current_size += para_len + 2  # +2 for \n\n
        
        # Add last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= min_chunk_size or not chunks:
                chunks.append(chunk_text)
            elif chunks:
                chunks[-1] = chunks[-1] + '\n\n' + chunk_text
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def _split_large_sentence(self, sentence: str, max_size: int) -> List[str]:
        """Split a large sentence into smaller chunks."""
        chunks = []
        words = sentence.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_len = len(word) + 1
            if current_size + word_len > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_len
            else:
                current_chunk.append(word)
                current_size += word_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """Fallback sentence splitting using regex."""
        import re
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text (useful for semantic chunking).
        
        Args:
            text: Text to analyze
        
        Returns:
            List of noun phrases
        """
        try:
            doc = self.nlp(text)
            return [chunk.text for chunk in doc.noun_chunks]
        except:
            return []
    
    def extract_entities(self, text: str) -> List[tuple]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of (entity_text, entity_label) tuples
        """
        try:
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        except:
            return []

"""
Semantic Chunker (spaCy-optimized, compact)
- Chia theo ranh giới câu bằng spaCy (fallback regex).
- Nhóm câu theo coherence: discourse markers + lexical overlap + (optional) entity overlap.
- Không thực hiện embedding; chỉ chuẩn bị text + provenance + score.
"""

import hashlib
import re
from typing import List, Optional
from pathlib import Path
from .base_chunker import BaseChunker
from .model import (
    Chunk, ChunkSet, ChunkType, ChunkStrategy,
    ProvenanceAgg, BlockSpan, Score
)
from PDFLoaders.provider import PDFDocument

# Block removed - using PageContent


class SemanticChunker(BaseChunker):
    # spaCy model mapping for different languages (fallback if config not found)
    SPACY_MODEL_MAP = {
        "en": "en_core_web_sm",        # English (small)
        "vi": "vi_core_news_lg",       # Vietnamese (large)
        "zh": "zh_core_web_sm",        # Chinese (small)
        "de": "de_core_news_sm",       # German (small)
        "fr": "fr_core_news_sm",       # French (small)
        "es": "es_core_news_sm",       # Spanish (small)
        "it": "it_core_news_sm",       # Italian (small)
        "pt": "pt_core_news_sm",       # Portuguese (small)
        "nl": "nl_core_news_sm",       # Dutch (small)
        "pl": "pl_core_news_sm",       # Polish (small)
        "ru": "ru_core_news_sm",       # Russian (small)
        "ja": "ja_core_news_sm",       # Japanese (small)
        "multilingual": "xx_ent_wiki_sm",  # Multilingual (small, basic)
    }
    
    def _load_config(self) -> dict:
        """Load semantic chunker config from chunker_config.yaml"""
        try:
            from .config_loader import load_chunker_config
            config = load_chunker_config()
            return config.get('semantic', {
                'max_tokens': 500,
                'overlap_tokens': 50,
                'similarity_threshold': 0.7,
                'min_chunk_size': 100,
                'spacy_models': self.SPACY_MODEL_MAP,
                'default_language': 'auto',
                'discourse_weight': 0.4,
                'lexical_weight': 0.4,
                'entity_weight': 0.2
            })
        except Exception:
            # Fallback to default values
            return {
                'max_tokens': 500,
                'overlap_tokens': 50,
                'similarity_threshold': 0.7,
                'min_chunk_size': 100,
                'spacy_models': self.SPACY_MODEL_MAP,
                'default_language': 'auto',
                'discourse_weight': 0.4,
                'lexical_weight': 0.4,
                'entity_weight': 0.2
            }
    
    def __init__(
        self,
        config: Optional[dict] = None,      # Load from config
        max_tokens: Optional[int] = None,   # Override config
        overlap_tokens: Optional[int] = None,
        language: Optional[str] = None,     # Language code for auto model selection
        nlp=None,                           # inject sẵn nlp (khuyên dùng)
        **kwargs                            # Backward compatibility
    ):
        """
        Args:
            config: Config dict from chunker_config.yaml['semantic']
            max_tokens: Override config max_tokens 
            overlap_tokens: Override config overlap_tokens
            language: language code (en, vi, zh, etc.) để auto-select model
            nlp: spaCy Language đã được tạo sẵn & truyền vào (khuyến nghị)
        """
        # Load config from file if not provided
        if config is None:
            config = self._load_config()
        
        # Apply overrides
        final_max_tokens = max_tokens or config.get('max_tokens', 500)
        final_overlap_tokens = overlap_tokens or config.get('overlap_tokens', 50)
        
        super().__init__(final_max_tokens, final_overlap_tokens)
        
        # Load from config
        self.min_sentences_per_chunk = config.get('min_chunk_size', 100) // 20  # Approx chars to sentences
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.discourse_weight = config.get('discourse_weight', 0.4)
        self.lexical_weight = config.get('lexical_weight', 0.4)
        self.entity_weight = config.get('entity_weight', 0.2)
        self.use_entity_overlap = self.entity_weight > 0
        
        # Language and model selection
        self.language = language or config.get('default_language', 'auto')
        self.spacy_models = config.get('spacy_models', self.SPACY_MODEL_MAP)

        self._nlp = nlp
        if self._nlp is None:
            # Auto-select model based on language
            if self.language != 'auto':
                spacy_model = self.spacy_models.get(self.language.lower(), "en_core_web_sm")
            else:
                spacy_model = "en_core_web_sm"  # Default to English for auto
            
            # Try to load model (KHÔNG auto-download)
            try:
                import spacy
                self._nlp = spacy.load(spacy_model)
            except Exception:
                self._nlp = None  # fallback regex

        # encoder cache cho estimate_tokens (tránh import tiktoken nhiều lần)
        self._encoder = None

        # Discourse markers (EN + VI gọn)
        self._cont_markers = {
            "furthermore","moreover","additionally","also","besides",
            "in addition","as well as","ngoài ra","thêm vào đó"
        }
        self._contrast_markers = {"however","but","nevertheless","nonetheless","yet","on the other hand","tuy nhiên","nhưng"}
        self._conclusion_markers = {"therefore","thus","hence","consequently","as a result","in conclusion","do đó","vì vậy","kết luận"}

    # ---------------- Public API ----------------

    def chunk(self, document: PDFDocument) -> ChunkSet:
        # Lưu file_path để sử dụng trong chunk_blocks
        self._current_file_path = document.file_path
        
        cs = ChunkSet(
            doc_id=document.metadata.get("doc_id", Path(document.file_path).stem),
            file_path=document.file_path,
            chunk_strategy="semantic",
        )
        # Use pages directly instead of blocks
        pages = document.pages
        if not pages:
            return cs

        for ch in self.chunk_pages(pages, cs.doc_id):
            cs.add_chunk(ch)
        cs.link_chunks()
        return cs

    def chunk_pages(self, pages, doc_id: str):
        sentences: List[str] = []
        sent2page = []

        for page in pages:
            # Aggregate ALL content: text + tables + figures (OCR-extracted)
            full_page_text = self._aggregate_page_content(page)
            
            for s in self._split_into_sentences(full_page_text):
                if s:
                    sentences.append(s)
                    sent2page.append(page)

        if not sentences:
            return []

        groups = self._group_by_coherence(sentences)
        out: List[Chunk] = []
        for idx, g in enumerate(groups):
            ch = self._build_chunk_from_group(g, sentences, sent2page, doc_id, self._current_file_path, idx)
            if ch:
                out.append(ch)
        
        # No artificial overlap - semantic grouping already provides natural context overlap
        # Applying additional overlap causes duplicate content issues
        
        return out

    # ---------------- Sentence splitting ----------------

    def _split_into_sentences(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        if self._nlp is not None:
            doc = self._nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            # fix nhanh bullets dính sát
            merged: List[str] = []
            i = 0
            while i < len(sents):
                cur = sents[i]
                if i < len(sents) - 1 and len(cur) < 10 and re.match(r"^([•\-\*]|\d+\.)", cur):
                    merged.append(f"{cur} {sents[i+1]}")
                    i += 2
                else:
                    merged.append(cur)
                    i += 1
            return merged

        # Regex fallback (EN+VI cơ bản)
        patt = r'(?<=[.!?])\s+(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỬỮỰỲÝỶỸỴĐ])'
        return [s.strip() for s in re.split(patt, text) if s.strip()]

    # ---------------- Page Content Aggregation ----------------
    
    def _aggregate_page_content(self, page) -> str:
        """
        Aggregate ALL page content: text + tables + figures (OCR-extracted).
        
        Critical: This ensures tables/figures (image-based) aren't lost during chunking.
        PDFProvider extracts them with OCR, but we must include them in chunks!
        
        Args:
            page: PageContent object from PDFProvider
            
        Returns:
            Full page text with tables and figures integrated
        """
        parts = []
        
        # 1. Main text content
        if page.text and page.text.strip():
            parts.append(page.text.strip())
        
        # 2. Tables (already extracted as text by PDFProvider with OCR enhancement)
        if page.tables:
            for table_idx, table in enumerate(page.tables, 1):
                table_text = f"\n[Table {table_idx}]\n"
                for row in table:
                    # Convert row to readable text
                    row_text = " | ".join(str(cell) for cell in row)
                    table_text += row_text + "\n"
                parts.append(table_text.strip())
        
        # 3. Figures (OCR-extracted text from images/diagrams)
        if page.figures:
            for fig_idx, figure in enumerate(page.figures, 1):
                # Figures contain OCR text in 'text' field
                fig_text = figure.get('text', '').strip()
                if fig_text:
                    parts.append(f"\n[Figure {fig_idx}]\n{fig_text}")
        
        return "\n\n".join(parts)

    # ---------------- Grouping by coherence ----------------

    def _group_by_coherence(self, sentences: List[str]) -> List[List[int]]:
        groups: List[List[int]] = []
        cur = [0]
        cur_tokens = self.estimate_tokens(sentences[0])

        for i in range(1, len(sentences)):
            s_prev, s_cur = sentences[i-1], sentences[i]
            tok_cur = self.estimate_tokens(s_cur)
            coh = self._coherence(s_prev, s_cur)

            exceed = (cur_tokens + tok_cur) > self.max_tokens
            weak = coh < 0.5
            enough = len(cur) >= self.min_sentences_per_chunk
            para_break = s_prev.endswith("\n\n") or s_prev.endswith("\n")

            if exceed or (weak and enough and para_break):
                groups.append(cur)
                cur = [i]
                cur_tokens = tok_cur
            else:
                cur.append(i)
                cur_tokens += tok_cur

        if cur:
            groups.append(cur)
        return groups

    # ---------------- Coherence scoring ----------------

    def _coherence(self, s1: str, s2: str) -> float:
        """
        0..1: discourse markers + lexical overlap + (optional) entity overlap.
        spaCy optional — nếu không có nlp, dùng heuristic nhẹ.
        """
        base = 0.5
        s2low = s2.lower()

        if any(m in s2low for m in self._cont_markers):
            base += 0.25
        elif any(m in s2low for m in self._contrast_markers):
            base += 0.1
        elif any(m in s2low for m in self._conclusion_markers):
            base += 0.15

        if self._nlp is None:
            # fallback lexical overlap đơn giản
            w1 = set(self._simple_content_words(s1))
            w2 = set(self._simple_content_words(s2))
            if w1 and w2:
                base += 0.2 * (len(w1 & w2) / max(len(w1), len(w2)))
            if "\n\n" in s1 or s1.endswith("\n"):
                base -= 0.2
            return max(0.0, min(1.0, base))

        # spaCy-based
        d1 = self._nlp(s1)
        d2 = self._nlp(s2)
        c1 = {t.lemma_.lower() for t in d1 if t.is_alpha and not t.is_stop and t.pos_ in ("NOUN","VERB","ADJ","PROPN")}
        c2 = {t.lemma_.lower() for t in d2 if t.is_alpha and not t.is_stop and t.pos_ in ("NOUN","VERB","ADJ","PROPN")}
        if c1 and c2:
            base += 0.25 * (len(c1 & c2) / max(len(c1), len(c2)))

        if self.use_entity_overlap:
            e1 = {e.text.lower() for e in d1.ents}
            e2 = {e.text.lower() for e in d2.ents}
            if e1 and e2 and e1 & e2:
                base += 0.15

        # Pronoun cue at start
        if any(t.pos_ == "PRON" for t in d2[:3]):
            base += 0.1

        if "\n\n" in s1 or s1.endswith("\n"):
            base -= 0.2

        return max(0.0, min(1.0, base))

    def _simple_content_words(self, text: str) -> List[str]:
        stops = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","as","is","was","are","were","be",
            "các","của","và","là","có","được","trong","với","cho"
        }
        return [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in stops and len(w) > 2]

    # ---------------- Chunk building ----------------

    def _build_chunk_from_group(
        self,
        idxs: List[int],
        sents: List[str],
        sent2page,
        doc_id: str,
        file_path: str,
        chunk_index: int,
    ) -> Optional[Chunk]:
        if not idxs:
            return None

        text = " ".join(sents[i] for i in idxs).strip()
        if not text:
            return None

        prov = ProvenanceAgg(doc_id=doc_id, file_path=file_path)
        seen = set()
        for i in idxs:
            page = sent2page[i]
            if id(page) in seen:
                continue
            seen.add(id(page))
            prov.add_span(BlockSpan(
                block_id=f"page_{page.page_number}",
                start_char=0,
                end_char=len(page.text or ""),
                page_number=page.page_number
            ))

        token_count = self.estimate_tokens(text)
        # coherence trung bình cặp kề
        coh_vals = []
        for j in range(len(idxs) - 1):
            coh_vals.append(self._coherence(sents[idxs[j]], sents[idxs[j+1]]))
        avg_coh = sum(coh_vals) / len(coh_vals) if coh_vals else 0.7

        score = Score(
            coherence_score=avg_coh,
            completeness_score=0.8,
            token_ratio=min(1.0, token_count / max(1, self.max_tokens)),
            structural_integrity=0.7,
        )

        # Collect unique page numbers for metadata
        page_numbers = sorted(set(sent2page[i].page_number for i in idxs))
        
        return Chunk(
            chunk_id=self._make_chunk_id(doc_id, chunk_index, text),
            text=text,
            doc_id=doc_id,
            token_count=token_count,
            char_count=len(text),
            chunk_type=ChunkType.SEMANTIC,
            strategy=ChunkStrategy.SEMANTIC_COHERENCE,
            provenance=prov,
            score=score,
            metadata={
                "sentence_count": len(idxs),
                "chunk_index": chunk_index,
                "avg_coherence": round(avg_coh, 3),
                "page_numbers": page_numbers,
                "page_span": f"{page_numbers[0]}-{page_numbers[-1]}" if page_numbers else "unknown"
            },
        )

    def _make_chunk_id(self, doc_id: str, idx: int, text: str) -> str:
        raw = f"{doc_id}|{idx}|{text[:64]}"
        return f"chunk_{doc_id}_{idx}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"

    # ---------------- Token estimate (cached encoder) ----------------

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoder = None  # fallback to None if encoder cannot be loaded
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

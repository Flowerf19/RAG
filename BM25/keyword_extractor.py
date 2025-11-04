# -*- coding: utf-8 -*-
"""
Keyword extraction utilities powered by spaCy.

The module centralises multi-language text normalisation so both the BM25
ingest and query path reuse the same tokenisation rules. We currently
support English and Vietnamese and load spaCy models lazily to avoid
initialisation overhead until the first call.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, Iterable, List, Optional

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "spaCy is required for BM25 keyword extraction. "
        "Install the appropriate models (e.g. en_core_web_sm, vi_core_news_lg)."
    ) from exc

logger = logging.getLogger(__name__)


DEFAULT_LANGUAGE_MODELS: Dict[str, str] = {
    "en": "en_core_web_sm",
    "vi": "vi_core_news_lg",
}


class KeywordExtractor:
    """
    Extracts keyword candidates from raw text using spaCy.

    The extractor focuses on unigram keywords (lemmas) and light noun-phrase
    chunks. It performs simple language heuristics when the caller does not
    specify a language.
    """

    def __init__(self, language_models: Optional[Dict[str, str]] = None) -> None:
        self.language_models = language_models or DEFAULT_LANGUAGE_MODELS.copy()
        if "en" not in self.language_models:
            self.language_models["en"] = "en_core_web_sm"
        self._nlp_cache: Dict[str, Language] = {}
        # ă \u0103, â \u00E2, đ \u0111, ê \u00EA, ô \u00F4, ơ \u01A1, ư \u01B0
        self._vi_markers = re.compile(r"[\u0103\u00E2\u0111\u00EA\u00F4\u01A1\u01B0]", re.IGNORECASE)

        logger.debug("KeywordExtractor initialised with languages: %s", list(self.language_models))

    def ensure_pipeline(self, lang: str) -> Language:
        """
        Lazily load spaCy pipeline for the given language.
        """
        if lang not in self.language_models:
            raise ValueError(f"Unsupported language '{lang}'. Configure a spaCy model before use.")

        if lang not in self._nlp_cache:
            model_name = self.language_models[lang]
            logger.info("Loading spaCy model '%s' for language '%s'", model_name, lang)
            try:
                self._nlp_cache[lang] = spacy.load(model_name, disable=("ner",))
            except OSError as e:
                raise RuntimeError(
                    f"Cannot load spaCy model '{model_name}' for '{lang}'. "
                    f"Install it with: python -m spacy download {model_name}"
                ) from e
        return self._nlp_cache[lang]

    def detect_language(self, text: str) -> str:
        """
        Lightweight language detection based on accented characters.
        Default to English when no clear Vietnamese markers appear.
        """
        if not text:
            return "en"
        text = unicodedata.normalize("NFC", text)
        return "vi" if self._vi_markers.search(text) else "en"

    def extract_keywords(
        self,
        text: str,
        lang: Optional[str] = None,
        *,
        include_phrases: bool = True,
        max_terms: Optional[int] = None,
    ) -> List[str]:
        """
        Extract keyword candidates from text.

        Args:
            text: Raw input text.
            lang: Optional ISO language code ("en" or "vi"). Auto-detected if omitted.
            include_phrases: Whether to include noun phrases in the result.
            max_terms: Optional limit on the number of returned terms.
        """
        if not text:
            return []
        text = unicodedata.normalize("NFC", text)

        language = (lang or self.detect_language(text)).lower()
        if language not in self.language_models:
            logger.debug("Language '%s' not configured — falling back to English model.", language)
            language = "en"
        nlp = self.ensure_pipeline(language)
        doc = nlp(text)

        terms = list(self._iter_token_terms(doc))
        if include_phrases:
            terms.extend(self._iter_phrase_terms(doc))

        # Preserve insertion order while removing duplicates.
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
                if max_terms and len(unique_terms) >= max_terms:
                    break
        return unique_terms

    def _iter_token_terms(self, doc: Doc) -> Iterable[str]:
        """
        Yield keyword candidates from individual tokens.
        """
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space or token.like_num:
                continue
            if len(token.lemma_) < 2:
                continue
            yield token.lemma_.lower()

    def _iter_phrase_terms(self, doc: Doc) -> Iterable[str]:
        """
        Yield simple noun-phrase candidates.
        """
        for chunk in doc.noun_chunks:
            chunk_text = chunk.lemma_.strip().lower()
            if not chunk_text or len(chunk_text) < 4:
                continue
            yield chunk_text

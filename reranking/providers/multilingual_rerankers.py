"""
Multilingual Reranker Classes
============================
Specialized reranker classes for multilingual models
"""

import logging
from typing import List
from reranking.i_reranker import IReranker, RerankResult, RerankerProfile
from reranking.providers.base_local_reranker import BaseLocalReranker

logger = logging.getLogger(__name__)


class JinaRerankerV2BaseMultilingual(BaseLocalReranker):
    """
    Jina Reranker V2 Base Multilingual

    Model: jinaai/jina-reranker-v2-base-multilingual
    Đặc điểm: Cross-encoder relevance scoring, multi-data
    Ngôn ngữ: >95
    Kích thước: ~0.3GB (137M params)
    """

    MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"

    def __init__(self, device: str = "cpu"):
        super().__init__(self.MODEL_NAME, device)
        self._profile = RerankerProfile(
            model_id=self.MODEL_NAME,
            provider="huggingface",
            max_query_length=512
        )

    def _load_model(self):
        """Load Jina reranker model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self._model.to(self.device)
            self._model.eval()
            logger.info(f"✅ Loaded Jina reranker: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load Jina reranker: {e}")
            raise


class GTEMultilingualRerankerBase(BaseLocalReranker):
    """
    GTE Multilingual Reranker Base

    Model: Alibaba-NLP/gte-multilingual-reranker-base
    Đặc điểm: Multi-task reranker, bfloat16 support
    Ngôn ngữ: >70
    Kích thước: ~0.5GB (278M params)
    """

    MODEL_NAME = "Alibaba-NLP/gte-multilingual-reranker-base"

    def __init__(self, device: str = "cpu"):
        super().__init__(self.MODEL_NAME, device)
        self._profile = RerankerProfile(
            model_id=self.MODEL_NAME,
            provider="huggingface",
            max_query_length=512
        )

    def _load_model(self):
        """Load GTE reranker model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self._model.to(self.device)
            self._model.eval()
            logger.info(f"✅ Loaded GTE reranker: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load GTE reranker: {e}")
            raise


class BGERerankerBase(BaseLocalReranker):
    """
    BGE Reranker Base

    Model: BAAI/bge-reranker-base
    Đặc điểm: Lightweight cross-encoder, fast inference
    Ngôn ngữ: >50
    Kích thước: ~0.4GB (110M params)
    """

    MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self, device: str = "cpu"):
        super().__init__(self.MODEL_NAME, device)
        self._profile = RerankerProfile(
            model_id=self.MODEL_NAME,
            provider="huggingface",
            max_query_length=512
        )

    def _load_model(self):
        """Load BGE reranker model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info(f"✅ Loaded BGE reranker: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load BGE reranker: {e}")
            raise


class Qwen3Reranker06B(BaseLocalReranker):
    """
    Qwen3 Reranker 0.6B

    Model: Qwen/Qwen3-Reranker-0.6B
    Đặc điểm: Instruct-customizable ranking, multilingual
    Ngôn ngữ: >100
    Kích thước: ~1.2GB (0.6B params)
    """

    MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

    def __init__(self, device: str = "cpu"):
        super().__init__(self.MODEL_NAME, device)
        self._profile = RerankerProfile(
            model_id=self.MODEL_NAME,
            provider="huggingface",
            max_query_length=512
        )

    def _load_model(self):
        """Load Qwen3 reranker model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info(f"✅ Loaded Qwen3 reranker: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen3 reranker: {e}")
            raise


class ProvenceRerankerDebertaV3V1(BaseLocalReranker):
    """
    Provence Reranker DeBERTaV3 V1

    Model: naver/provence-reranker-debertav3-v1
    Đặc điểm: Pruning + reranking, multilingual version
    Ngôn ngữ: >50
    Kích thước: ~1.7GB (435M params)
    """

    MODEL_NAME = "naver/provence-reranker-debertav3-v1"

    def __init__(self, device: str = "cpu"):
        super().__init__(self.MODEL_NAME, device)
        self._profile = RerankerProfile(
            model_id=self.MODEL_NAME,
            provider="huggingface",
            max_query_length=512
        )

    def _load_model(self):
        """Load Provence reranker model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self._model.to(self.device)
            self._model.eval()
            logger.info(f"✅ Loaded Provence reranker: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load Provence reranker: {e}")
            raise
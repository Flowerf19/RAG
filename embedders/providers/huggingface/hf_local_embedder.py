"""
HuggingFace Local Embedder
==========================
Implementation sử dụng transformers library local.
Tương tự như GemmaEmbedder nhưng dùng transformers thay vì Ollama.
"""

from typing import Optional, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .base_huggingface_embedder import BaseHuggingFaceEmbedder
from ...model.embedding_profile import EmbeddingProfile


class HuggingFaceLocalEmbedder(BaseHuggingFaceEmbedder):
    """
    HuggingFace Local embedding provider using transformers.
    Single Responsibility: Tạo embeddings sử dụng local transformers models.
    
    Config:
        - Model: BAAI/bge-small-en-v1.5 (default)
        - Dimension: 384
        - Max tokens: 512
        - Provider: huggingface
    """
    
    # Class-level constants
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DIMENSION = 384
    MAX_TOKENS = 512
    PROVIDER = "huggingface"

    def __init__(self,
                 profile: Optional[EmbeddingProfile] = None,
                 model_name: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize HF Local embedder.

        Args:
            profile: Embedding profile, nếu None sẽ tạo từ class constants
            model_name: Override model name
            device: Device for inference ("cpu", "cuda")
        """
        super().__init__(profile, model_name)
        
        self.device = device
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        
        self._load_model()

    def _load_model(self):
        """Load transformers model and tokenizer."""
        try:
            print(f"🔄 Loading local HF model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)

            # Move to device
            if self.device != "cpu" and torch.cuda.is_available():
                self._model = self._model.to(self.device)
                print(f"✅ Model loaded on {self.device}")
            else:
                print("✅ Model loaded on CPU")

        except ImportError:
            raise ImportError(
                "transformers and torch required for local HF models. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load local model {self.model_name}: {e}")

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using local transformers.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector
        """
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Tokenize
            inputs = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

                # Normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy().tolist()[0]

        except Exception as e:
            raise RuntimeError(f"Local embedding failed: {e}")

    def test_connection(self) -> bool:
        """
        Test if local model is loaded.

        Returns:
            bool: True if model loaded successfully
        """
        return self._model is not None and self._tokenizer is not None

    @classmethod
    def create_default(cls, device: str = "cpu", **kwargs) -> 'HuggingFaceLocalEmbedder':
        """
        Factory method để tạo HF Local embedder với cấu hình mặc định.

        Args:
            device: Device for inference
            **kwargs: Additional arguments

        Returns:
            HuggingFaceLocalEmbedder: Configured local embedder
        """
        return cls(device=device)
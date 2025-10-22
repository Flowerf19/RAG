"""
HuggingFace API Embedder
========================
Implementation sử dụng HuggingFace Inference API.
Tương tự như GemmaEmbedder nhưng gọi HF API thay vì Ollama.
"""

from typing import Optional, List
import requests
import time

from .base_huggingface_embedder import BaseHuggingFaceEmbedder
from ...model.embedding_profile import EmbeddingProfile
from .token_manager import get_hf_token


class HuggingFaceApiEmbedder(BaseHuggingFaceEmbedder):
    """
    HuggingFace API embedding provider.
    Single Responsibility: Tạo embeddings sử dụng HF Inference API.
    
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
                 api_token: Optional[str] = None):
        """
        Initialize HF API embedder.

        Args:
            profile: Embedding profile, nếu None sẽ tạo từ class constants
            model_name: Override model name
            api_token: HF API token (auto-loaded if None)
        """
        super().__init__(profile, model_name)
        
        # Load API token
        self.api_token = get_hf_token(api_token)
        if not self.api_token:
            raise ValueError(
                "HF_TOKEN required for API mode. "
                "Set environment variable or pass api_token parameter."
            )
        
        # Test connection
        self._test_api_connection()

    def _test_api_connection(self):
        """Test API connection and token validity."""
        headers = {"Authorization": f"Bearer {self.api_token}"}
        url = "https://huggingface.co/api/whoami-v2"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                user_info = response.json()
                print(f"✅ HF API authenticated as: {user_info.get('name', 'Unknown')}")
            else:
                raise ValueError(f"API authentication failed: {response.status_code}")
        except Exception as e:
            raise ValueError(f"HF API connection failed: {e}")

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using HF Inference API.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector
        """
        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        
        # Send as JSON data
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True, "use_cache": False}
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    # Handle different response formats
                    if isinstance(result, list):
                        if len(result) == 0:
                            raise ValueError("Empty response from API")
                        
                        # Check first element type
                        first = result[0]
                        
                        if isinstance(first, (int, float)):
                            # Direct embedding vector: [0.1, 0.2, ...]
                            return result
                        elif isinstance(first, list):
                            # Nested list: [[0.1, 0.2, ...]]
                            return first
                        elif isinstance(first, dict) and "embedding" in first:
                            # Dict with embedding key
                            return first["embedding"]
                    
                    raise ValueError(f"Unexpected API response format: {type(result)}, first element: {type(result[0]) if result else 'empty'}")

                elif response.status_code == 503:
                    print(f"⏳ Model loading (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(5 * (attempt + 1))
                    continue

                elif response.status_code == 429:
                    print(f"🚫 Rate limited (attempt {attempt + 1}/{max_retries})")
                    time.sleep(10 * (attempt + 1))
                    continue

                else:
                    raise ValueError(f"API error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"API request failed after {max_retries} attempts: {e}")
                time.sleep(2 * (attempt + 1))

        raise RuntimeError("Max retries exceeded")

    def test_connection(self) -> bool:
        """
        Test connection to HuggingFace API.

        Returns:
            bool: True if connection successful
        """
        try:
            self._test_api_connection()
            # Try a small embedding to verify it works
            test_emb = self.embed("test")
            return len(test_emb) > 0
        except Exception:
            return False

    @classmethod
    def create_default(cls, api_token: Optional[str] = None, **kwargs) -> 'HuggingFaceApiEmbedder':
        """
        Factory method để tạo HF API embedder với cấu hình mặc định.

        Args:
            api_token: HF API token
            **kwargs: Additional arguments

        Returns:
            HuggingFaceApiEmbedder: Configured API embedder
        """
        return cls(api_token=api_token)
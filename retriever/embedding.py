from typing import List, Optional
import requests
import yaml
import os

class GemmaEmbedder:
    def __init__(self, config_path: Optional[str] = None, model: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Khởi tạo Gemma embedding model qua Ollama API.
        
        Args:
            config_path: Đường dẫn đến file config YAML (mặc định: config/embedding_config.yaml)
            model: Tên model Gemma (override config nếu cung cấp)
            endpoint: Endpoint Ollama (override config nếu cung cấp)
        """
        # Load config từ file YAML
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "embedding_config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        embedding_config = config.get("embedding", {})
        
        # Sử dụng tham số truyền vào hoặc lấy từ config
        self.model = model or embedding_config.get("model", "bge-m3")
        self.endpoint = endpoint or embedding_config.get("endpoint", "http://localhost:11434/api/embed")
        self.timeout = embedding_config.get("timeout", 120)
        self.batch_size = embedding_config.get("batch_size", 16)
        self.normalize = embedding_config.get("normalize", True)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Nhận vào list văn bản, trả về list embedding vector.
        
        Args:
            texts: Danh sách văn bản cần embed
            
        Returns:
            List các vector embedding
        """
        embeddings = []
        # Xử lý theo batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for text in batch:
                payload = {
                    "model": self.model,
                    "input": text
                }
                try:
                    response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
                    response.raise_for_status()
                    result = response.json()
                    # Ollama /api/embed trả về {"embeddings": [[...]]}
                    if "embeddings" in result and len(result["embeddings"]) > 0:
                        embeddings.append(result["embeddings"][0])
                    elif "embedding" in result:
                        embeddings.append(result["embedding"])
                    else:
                        print(f"Unexpected response format: {result}")
                        embeddings.append([0.0] * 1024)  # Default dimension
                except Exception as e:
                    print(f"Error embedding text: {e}")
                    # Trả về vector 0 nếu lỗi
                    embeddings.append([0.0] * 1024)  # BGE-M3 dimension
        return embeddings

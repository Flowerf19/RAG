from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class RerankerAPITokens(BaseModel):
    hf: Optional[str] = None
    cohere: Optional[str] = None
    jina: Optional[str] = None


class RetrievalOptions(BaseModel):
    top_k: int = Field(default=3, ge=0, le=25)
    max_context_chars: int = Field(default=8000, ge=500, le=20000)
    embedder_type: Literal["ollama", "huggingface_local", "huggingface_api"] = "huggingface_local"
    reranker_type: Literal[
        "none",
        "bge_m3_hf_local",
        "bge_m3_ollama",
        "bge_m3_hf_api",
        "cohere",
        "jina",
    ] = "bge_m3_hf_local"
    use_query_enhancement: bool = True
    api_tokens: Optional[RerankerAPITokens] = None

    def tokens_dict(self) -> Optional[Dict[str, str]]:
        return self.api_tokens.model_dump(exclude_none=True) if self.api_tokens else None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: Optional[List[ChatHistoryMessage]] = None
    provider: Optional[str] = None
    top_k: int = Field(default=3, ge=0, le=25)
    documents: Optional[List[str]] = Field(default=None)
    include_sources: bool = True
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)

    def resolve_retrieval(self) -> RetrievalOptions:
        config = self.retrieval
        if "top_k" in self.model_fields_set:
            config = config.model_copy(update={"top_k": self.top_k})
        return config


class ProcessDocumentsRequest(BaseModel):
    pdf_dir: Optional[str] = None
    asynchronous: bool = False

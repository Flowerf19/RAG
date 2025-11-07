from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: Optional[List[ChatHistoryMessage]] = None
    provider: Optional[str] = None
    top_k: int = Field(default=3, ge=0, le=25)
    include_sources: bool = True
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)


class ProcessDocumentsRequest(BaseModel):
    pdf_dir: Optional[str] = None
    asynchronous: bool = False

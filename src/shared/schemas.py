from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.shared.enums import InteractionType, SourceType, DocType


class HealthResponse(BaseModel):
    status: str
    db_connection: str
    sheets_connection: str


class InteractionMessage(BaseModel):
    role: InteractionType
    message: str
    tool_calls: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InteractionRequest(BaseModel):
    sessionId: str = Field(..., min_length=4)
    practiceId: Optional[str] = None
    message: InteractionMessage
    user_data: Optional[Dict[str, Any]] = None


class InteractionResponse(BaseModel):
    sessionId: str
    messages: List[InteractionMessage]
    toolCall: Optional[str] = None
    states: List[str]


class QAPair(BaseModel):
    question: str
    answer: str


class DocumentData(BaseModel):
    name: str
    docType: Optional[DocType] = None
    data: Optional[str] = None


class SourceData(BaseModel):
    webPageURL: Optional[str] = None
    qa_pair: Optional[QAPair] = None
    document: Optional[DocumentData] = None


class CreateEmbeddingsRequest(BaseModel):
    practiceId: str
    sourceType: SourceType
    sourceData: SourceData


class CreateEmbeddingsResponse(BaseModel):
    status: str
    message: str


class DeleteEmbeddingsRequest(BaseModel):
    practiceId: str
    sourceType: SourceType
    sourceData: SourceData


class DeleteEmbeddingsResponse(BaseModel):
    status: str
    message: str
    deleted_count: int

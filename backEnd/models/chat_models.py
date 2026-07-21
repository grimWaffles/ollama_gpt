from typing import List
from uuid import UUID

from pydantic import BaseModel, Field

class ModelInfo(BaseModel):
    id: int
    name: str
    modelKey: str

class ChatRequest(BaseModel):
    userId: int
    chatId: int
    message: str
    modelName:str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    userId: int
    chatId: int
    messages: List[ChatMessage] = Field(default_factory=list)
from typing import List
from uuid import UUID

from fastapi import File
from pydantic import BaseModel, Field

class ModelInfo(BaseModel):
    id: int
    name: str
    modelKey: str


class PendingAttachment(BaseModel):
    file: str
    previewUrl: str = ""


class ChatAttachment(BaseModel):
    name: str
    size: int
    type: str
    previewUrl: str

class ChatRequest(BaseModel):
    userId: int
    chatId: int
    message: str
    modelName:str = ""
    attachments: List[ChatAttachment] = []

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    userId: int
    chatId: int
    messages: List[ChatMessage] = Field(default_factory=list)

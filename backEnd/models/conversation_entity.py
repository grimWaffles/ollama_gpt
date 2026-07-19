# entities/conversation_entity.py

from uuid import UUID, uuid4
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConversationEntity:
    chatId: int
    userId: int
    chatName: str
    created_at: datetime

    @staticmethod
    def create(user_id: int, chat_name: str) -> "ConversationEntity":
        return ConversationEntity(
            chatId=int(),
            userId=user_id,
            chatName=chat_name,
            created_at=datetime.now(),
        )
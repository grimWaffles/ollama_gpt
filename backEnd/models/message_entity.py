# entities/message_entity.py

from uuid import UUID, uuid4
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MessageEntity:
    id: UUID
    chatId: UUID
    role: str
    message: str
    sequenceNo: int
    created_at: datetime

    @staticmethod
    def create(
        chat_id: UUID,
        role: str,
        message: str,
        sequence_no: int,
    ) -> "MessageEntity":
        return MessageEntity(
            id=uuid4(),
            chatId=chat_id,
            role=role,
            message=message,
            sequenceNo=sequence_no,
            created_at=datetime.now(),
        )
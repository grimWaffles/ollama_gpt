from datetime import datetime
import uuid
from dataclasses import dataclass
from uuid import UUID

@dataclass
class UserEntity:
    id: UUID
    username:str
    created_at: datetime

    @staticmethod
    def create(username:str):
        return UserEntity(
            id=uuid.uuid4(),
            username=username,
            created_at=datetime.now()
        )
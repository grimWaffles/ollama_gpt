import os
import psycopg2
from uuid import UUID
from datetime import datetime
from dotenv import load_dotenv

from models.conversation_entity import ConversationEntity

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)


class ConversationRepository:
    def __init__(self):
        try:
            self.connection = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", 5432),
                database=os.getenv("DB_NAME", "postgres"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD"),
            )
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.connection = None
            raise

    def close(self):
        try:
            if self.connection:
                self.connection.close()
        except Exception as e:
            print(f"Error closing connection: {e}")

    # --------------- Conversation CRUD ---------------
    def create_conversation(self, conversation):
        query = """
            INSERT INTO conversations (
                user_id, chat_name, created_at
            )
            VALUES (%s, %s, %s)
            RETURNING chat_id
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (
                    conversation.userId,
                    conversation.chatName,
                    conversation.created_at
                ))

                new_id = cursor.fetchone()[0]
            self.connection.commit()

            return new_id
        except Exception as e:
            print(f"Error creating conversation: {e}")
            self.connection.rollback()
            raise

    def get_conversation(self, chat_id: int):
        query = """
                SELECT chat_id, user_id, chat_name, created_at
                FROM conversations
                WHERE chat_id = %s \
                """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (chat_id,))
                return cursor.fetchone()
        except Exception as e:
            print(f"Error fetching conversation {chat_id}: {e}")
            self.connection.rollback()
            return None

    def get_user_conversations(self, user_id: int) -> list[ConversationEntity]:
        query = """
                SELECT chat_id, user_id, chat_name, created_at
                FROM conversations
                WHERE user_id = %s
                ORDER BY created_at DESC
                """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (user_id,))
                rows = cursor.fetchall()
                return [
                    ConversationEntity(
                        chatId=row[0],
                        userId=row[1],
                        chatName=row[2],
                        created_at=row[3],
                    )
                    for row in rows
                ]
        except Exception as e:
            print(f"Error fetching conversations for user {user_id}: {e}")
            self.connection.rollback()
            return []

    def get_max_chat_id(self) -> int:
        """Returns the highest existing chat_id, or 0 if no conversations exist yet."""
        query = "SELECT COALESCE(MAX(chat_id), 0) FROM conversations"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            print(f"Error fetching max chat_id: {e}")
            self.connection.rollback()
            return 0

    def update_conversation_name(self, chat_id: int, name: str):
        query = """
                UPDATE conversations
                SET chat_name = %s
                WHERE chat_id = %s \
                """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (name, chat_id))
            self.connection.commit()
        except Exception as e:
            print(f"Error updating conversation name for {chat_id}: {e}")
            self.connection.rollback()
            raise

    def delete_conversation(self, chat_id: int):
        query = """
                DELETE FROM conversations
                WHERE chat_id = %s \
                """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (chat_id,))
            self.connection.commit()
        except Exception as e:
            print(f"Error deleting conversation {chat_id}: {e}")
            self.connection.rollback()
            raise

    # --------------- Message CRUD ---------------
    def create_message(self, message):
        query = """
                INSERT INTO messages (chat_id, role, message, sequence_no, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (
                    message.chatId,
                    message.role,
                    message.message,
                    message.sequenceNo,
                    message.created_at
                ))
                new_id = cursor.fetchone()[0]

            self.connection.commit()
            return new_id

        except Exception as e:
            print(f"Error creating message for chat {message.chatId}: {e}")
            self.connection.rollback()
            raise

    def get_messages(self, chat_id: int):
        query = """
                SELECT id, chat_id, role, message, sequence_no, created_at
                FROM messages
                WHERE chat_id = %s
                ORDER BY sequence_no \
                """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (chat_id,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error fetching messages for chat {chat_id}: {e}")
            self.connection.rollback()
            return []

    def update_message(self, message_id: int, new_message: str):
        query = """
                UPDATE messages
                SET message = %s
                WHERE id = %s \
                """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (new_message, message_id))
            self.connection.commit()
        except Exception as e:
            print(f"Error updating message {message_id}: {e}")
            self.connection.rollback()
            raise

    def delete_message(self, message_id: int):
        query = """
                DELETE FROM messages
                WHERE id = %s \
                """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (message_id,))
            self.connection.commit()
        except Exception as e:
            print(f"Error deleting message {message_id}: {e}")
            self.connection.rollback()
            raise
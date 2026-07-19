# services/llm_service.py
import traceback
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import List

from agents.ollama_agent import OllamaAgent
from models import conversation_entity
from models.chat_models import ChatMessage
from repository.repository import ConversationRepository

class LlmService:
    def __init__(self):
        self.repo = ConversationRepository()

        self.local_models = {
            1: ("Gemma 3", "ollama:gemma3"),
            2: ("Gemma 4", "ollama:gemma4"),
            3: ("Qwen 3", "ollama:qwen3"),
            4: ("Llama 3.1", "ollama:llama3.1"),
        }

        self.cloud_models = {
            1: ("GPT-OSS", "ollama:gpt-oss:120b-cloud"),
            2: ("GPT-5 Mini", "openai:gpt-5-mini"),
            3: ("Claude Sonnet 4", "anthropic:claude-sonnet-4"),
            4: ("Gemini 2.5 Pro", "google_genai:gemini-2.5-pro"),
        }

        self.starter_system_prompt = (
            "You are a helpful assistant. "
            "Answer the user's questions. "
            "Only call tools when they are actually needed."
        )

    def chatWithLlm(
            self,
            user_id: int,
            chat_id: int,
            message: str,
    ) -> tuple[int, List[ChatMessage]]:

        conversation: List[ChatMessage] = []

        try:
            # --- New chat: assign the next available chat_id ---
            if not chat_id or chat_id == 0:
                max_chat_id = self.repo.get_max_chat_id()
                chat_id = self.repo.create_conversation(
                    SimpleNamespace(
                        chatId=chat_id,
                        userId=user_id,
                        chatName=f"Conversation #{max_chat_id+1}",
                        created_at=datetime.utcnow(),
                    )
                )

            # --- Fetch existing conversation from DB ---
            rows = self.repo.get_messages(chat_id)

            last_sequence_no = 0
            for row in rows:
                _, _, role, msg_text, sequence_no, _ = row
                conversation.append(ChatMessage(role=role, content=msg_text))
                last_sequence_no = max(last_sequence_no, sequence_no)

            if not conversation:
                conversation.append(
                    ChatMessage(
                        role="system",
                        content=self.starter_system_prompt,
                    )
                )

            # --- Add the user's current message ---
            conversation.append(
                ChatMessage(
                    role="user",
                    content=message,
                )
            )

            # Select the desired model (example: GPT-5 Mini)
            _, model_name = self.cloud_models[1]

            agent = OllamaAgent(
                model_name=model_name,
                tools=[],
                system_prompt=self.starter_system_prompt,
            )

            full_response = agent.invoke(conversation)
            new_messages = full_response[len(conversation):]

            # --- Save user message ---
            now = datetime.utcnow()
            last_sequence_no += 1
            self.repo.create_message(
                SimpleNamespace(
                    id=0,
                    chatId=chat_id,
                    role="user",
                    message=message,
                    sequenceNo=last_sequence_no,
                    created_at=now,
                )
            )

            # --- Save assistant response message(s) ---
            for new_msg in new_messages:
                last_sequence_no += 1
                self.repo.create_message(
                    SimpleNamespace(
                        id=0,
                        chatId=chat_id,
                        role=new_msg.role,
                        message=new_msg.content,
                        sequenceNo=last_sequence_no,
                        created_at=datetime.utcnow(),
                    )
                )

            return chat_id, new_messages
        except Exception as e:
            print(f"Error: {e}")
            return chat_id, []

    def getConversation(self,user_id):
        user_conversations = self.repo.get_user_conversations(user_id)

        return user_conversations

    def getLocalModelList(self) -> List[dict]:
        """Returns the list of available local models as {id, name, modelKey}."""
        try:
            return [
                {"id": model_id, "name": name, "modelKey": model_key}
                for model_id, (name, model_key) in self.local_models.items()
            ]
        except Exception as e:
            print(f"Error fetching local model list: {e}")
            return []

    def getCloudModelList(self) -> List[dict]:
        """Returns the list of available cloud models as {id, name, modelKey}."""
        try:
            return [
                {"id": model_id, "name": name, "modelKey": model_key}
                for model_id, (name, model_key) in self.cloud_models.items()
            ]
        except Exception as e:
            print(f"Error fetching cloud model list: {e}")
            return []

    def getMessagesForChat(self, chat_id: int) -> List[ChatMessage]:
        """Fetches all messages for a given chat_id, ordered by sequence_no."""
        try:
            rows = self.repo.get_messages(chat_id)
            return [
                ChatMessage(role=row[2], content=row[3])
                for row in rows
            ]
        except Exception as e:
            print(f"Error fetching messages for chat {chat_id}: {e}")
            return []
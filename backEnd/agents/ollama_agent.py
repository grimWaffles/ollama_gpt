import traceback
from pathlib import Path
from typing import Any, List, Optional, Union, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent

from models.chat_models import ChatMessage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}

# A conversation entry can now be a ChatMessage (plain string content) or a raw
# dict with role/content, where content may be a string OR a list of multimodal
# content blocks (used for enriched, file-aware turns).
ConversationEntry = Union[ChatMessage, Dict[str, Any]]


class OllamaAgent:
    def __init__(
            self,
            model_name: str,
            tools: Optional[List[Any]] = None,
            system_prompt: str = "",
    ):
        self.agent = create_agent(
            model=model_name,
            tools=tools or [],
            system_prompt=system_prompt,
        )

    def extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        parts.append(block["text"])
                    else:
                        parts.append(str(block))
                else:
                    parts.append(str(block))
            return "".join(parts)

        return str(content)

    def _to_message_dict(self, entry: ConversationEntry) -> Dict[str, Any]:
        """
        Normalizes a ChatMessage or a raw dict into the {"role", "content"} shape
        the agent expects. `content` may be a string or a list of content blocks
        (e.g. text + image blocks) — we don't touch it, just pass it through.
        """
        if isinstance(entry, dict):
            return {"role": entry["role"], "content": entry["content"]}
        return {"role": entry.role, "content": entry.content}

    def invoke(self, conversation: List[ConversationEntry]) -> List[ChatMessage]:
        messages = [self._to_message_dict(msg) for msg in conversation]

        try:
            result = self.agent.invoke({"messages": messages})
            returned_messages: List[ChatMessage] = []

            for message in result["messages"]:
                content = self.extract_text(getattr(message, "content", None))

                msg_type = getattr(message, "type", None)
                if msg_type in _ROLE_MAP:
                    role = _ROLE_MAP[msg_type]
                else:
                    role = getattr(message, "role", None) or type(message).__name__.replace("Message", "").lower()

                returned_messages.append(ChatMessage(role=role, content=content))

            return returned_messages

        except Exception:
            print(f"Error: {traceback.format_exc()}")
            return []

    def stream(self, conversation: List[ConversationEntry]):
        messages = [self._to_message_dict(msg) for msg in conversation]
        seen = len(messages)

        try:
            for chunk in self.agent.stream({"messages": messages}, stream_mode="values"):
                msgs = chunk["messages"]
                for message in msgs[seen:]:
                    content = self.extract_text(getattr(message, "content", None))

                    msg_type = getattr(message, "type", None)
                    if msg_type in _ROLE_MAP:
                        role = _ROLE_MAP[msg_type]
                    else:
                        role = getattr(message, "role", None) or type(message).__name__.replace("Message", "").lower()

                    yield ChatMessage(role=role, content=content)
                seen = len(msgs)

        except Exception:
            print(f"Error: {traceback.format_exc()}")
            return
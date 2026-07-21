# agents/ollama_agent.py
import traceback
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent

from models.chat_models import ChatMessage

# Load ../.env relative to the Agents folder
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# LangChain's internal message ".type" values -> our app's role strings
_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


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

    def extract_text(self,content: Any) -> str:
        """
        LangChain message .content can be a plain string OR a list of content
        blocks (e.g. when tool calls / tool results are involved). Normalize
        to a single string so it fits our ChatMessage(content: str) model.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    # Most text-bearing blocks have a "text" key (e.g. {"type": "text", "text": "..."})
                    if "text" in block:
                        parts.append(block["text"])
                    else:
                        parts.append(str(block))
                else:
                    parts.append(str(block))
            return "".join(parts)

        return str(content)

    def invoke(self, conversation: List[ChatMessage]) -> List[ChatMessage]:
        """
        Runs the agent on the full conversation and returns the FULL set of
        messages produced by the graph (including the messages that were
        passed in). Callers that only want the newest reply should slice
        the result themselves (see LlmService.chatWithLlm).
        """
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in conversation
        ]

        try:
            result = self.agent.invoke({"messages": messages})
            returned_messages: List[ChatMessage] = []

            for message in result["messages"]:
                content = self.extract_text(getattr(message, "content", None))

                msg_type = getattr(message, "type", None)
                if msg_type in _ROLE_MAP:
                    role = _ROLE_MAP[msg_type]
                else:
                    # Fallback for anything unexpected (e.g. the generic
                    # langchain_core ChatMessage class, which does have .role)
                    role = getattr(message, "role", None) or type(message).__name__.replace("Message", "").lower()

                returned_messages.append(
                    ChatMessage(
                        role=role,
                        content=content,
                    )
                )

            return returned_messages

        except Exception:
            print(f"Error: {traceback.format_exc()}")
            return []

    def stream(self, conversation: List[ChatMessage]):
        messages = [{"role": m.role, "content": m.content} for m in conversation]
        seen = len(messages)
        try:
            for chunk in self.agent.stream({"messages": messages}, stream_mode="values"):
                msgs = chunk["messages"]
                for message in msgs[seen:]:
                    content = self.extract_text(getattr(message, "content", None))
                    msg_type = getattr(message, "type", None)
                    role = _ROLE_MAP.get(msg_type) or getattr(message, "role", None) or type(message).__name__.replace(
                        "Message", "").lower()
                    yield ChatMessage(role=role, content=content)
                seen = len(msgs)
        except Exception:
            print(f"Error: {traceback.format_exc()}")
            return
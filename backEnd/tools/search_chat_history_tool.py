from typing import Literal, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from repository.vector_repository import VectorRepository
from services.embedding_service import EmbeddingService


class ChatHistorySearchInput(BaseModel):
    query: str = Field(description="The topic or question to search for in past chat messages.")
    scope: Literal["this_conversation", "all_conversations"] = Field(
        default="this_conversation",
        description=(
            "'this_conversation' searches only earlier messages in the current chat "
            "(use when the current conversation is long and something earlier may be relevant). "
            "'all_conversations' searches across the user's entire chat history "
            "(use when the user references something from a previous, different conversation)."
        ),
    )


class ChatHistorySearchTool(BaseTool):
    name: str = "search_chat_history"
    description: str = (
        "Search past chat messages for relevant context. Use 'this_conversation' scope "
        "when earlier parts of the current chat may hold the answer but aren't in view. "
        "Use 'all_conversations' scope when the user refers to something discussed in a "
        "different, earlier conversation."
    )
    args_schema: Type[BaseModel] = ChatHistorySearchInput

    vector_repo: VectorRepository
    embedding_service: EmbeddingService
    chat_id: int
    user_id: int
    k: int = 6

    def _run(self, query: str, scope: str = "this_conversation") -> str:
        query_embedding = self.embedding_service.embed_query(query)
        search_chat_id = self.chat_id if scope == "this_conversation" else None

        results = self.vector_repo.similarity_search_messages(
            user_id=self.user_id,
            query_embedding=query_embedding,
            chat_id=search_chat_id,
            k=self.k,
        )

        if not results:
            return "No relevant prior messages found."

        return "\n\n---\n\n".join(
            f"[{r['role']} · chat {r['chat_id']}]\n{r['content']}"
            for r in results
        )

    async def _arun(self, query: str, scope: str = "this_conversation") -> str:
        return self._run(query, scope)
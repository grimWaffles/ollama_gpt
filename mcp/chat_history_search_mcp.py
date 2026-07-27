"""
chat_history_search_mcp.py

Tool function for the chat-history-search MCP server. The FastMCP app
instance ("mcp") and the request-scoped context (current user_id, chat_id,
and the shared vector_repo / embedding_service singletons) are created and
managed elsewhere (the main server module) — this file just imports them
and registers its tool on the shared "mcp" instance.

NOTE: adjust the import below to match how your main module actually
exposes these — this assumes a `get_context()` accessor that returns the
current request's (vector_repo, embedding_service, user_id, chat_id).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from server import mcp, get_context  # shared FastMCP instance + per-request context


@mcp.tool(
    name="search_chat_history",
    description=(
        "Search past chat messages for relevant context. Use 'this_conversation' scope "
        "when earlier parts of the current chat may hold the answer but aren't in view. "
        "Use 'all_conversations' scope when the user refers to something discussed in a "
        "different, earlier conversation."
    ),
)
def search_chat_history(
    query: str = Field(description="The topic or question to search for in past chat messages."),
    scope: Literal["this_conversation", "all_conversations"] = Field(
        default="this_conversation",
        description=(
            "'this_conversation' searches only earlier messages in the current chat "
            "(use when the current conversation is long and something earlier may be relevant). "
            "'all_conversations' searches across the user's entire chat history "
            "(use when the user references something from a previous, different conversation)."
        ),
    ),
) -> str:
    ctx = get_context()
    query_embedding = ctx.embedding_service.embed_query(query)
    search_chat_id = ctx.chat_id if scope == "this_conversation" else None

    results = ctx.vector_repo.similarity_search_messages(
        user_id=ctx.user_id,
        query_embedding=query_embedding,
        chat_id=search_chat_id,
        k=6,
    )

    if not results:
        return "No relevant prior messages found."

    return "\n\n---\n\n".join(
        f"[{r['role']} · chat {r['chat_id']}]\n{r['content']}"
        for r in results
    )
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from repository.vector_repository import VectorRepository
from services.embedding_service import EmbeddingService


class KnowledgeBaseSearchInput(BaseModel):
    query: str = Field(description="The question or topic to search for in the uploaded documents.")


class KnowledgeBaseSearchTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = (
        "Search the documents the user has uploaded in this conversation for relevant "
        "passages. Use this when the user asks a question that likely requires "
        "information from an uploaded file rather than your own knowledge."
    )
    args_schema: Type[BaseModel] = KnowledgeBaseSearchInput

    vector_repo: VectorRepository
    embedding_service: EmbeddingService
    chat_id: int
    k: int = 4

    def _run(self, query: str) -> str:
        query_embedding = self.embedding_service.embed_query(query)
        results = self.vector_repo.similarity_search(self.chat_id, query_embedding, k=self.k)

        if not results:
            return "No relevant content found in the uploaded documents."

        return "\n\n---\n\n".join(
            f"[Source: {r['filename']}, chunk {r['chunk_index']}]\n{r['content']}"
            for r in results
        )

    async def _arun(self, query: str) -> str:
        return self._run(query)
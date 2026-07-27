from typing import Any, Dict, List
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "nomic-embed-text"   # must match VECTOR(768) dimension above
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


class EmbeddingService:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def chunk_and_embed(self, text: str) -> List[Dict[str, Any]]:
        """Returns [{"content", "embedding", "chunk_index", "token_count"}, ...]"""
        if not text or not text.strip():
            return []

        pieces = self.splitter.split_text(text)
        vectors = self.embeddings.embed_documents(pieces)

        return [
            {
                "content": piece,
                "embedding": vector,
                "chunk_index": i,
                "token_count": max(1, len(piece) // 4),  # rough estimate
            }
            for i, (piece, vector) in enumerate(zip(pieces, vectors))
        ]

    def embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)
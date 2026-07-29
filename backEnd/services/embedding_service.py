import logging
from typing import List, Dict, Any

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.env_service import EnvService

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, vector_repo, app_config):
        self.config = EnvService().get_embedder_settings()
        self.embeddings = OllamaEmbeddings(model=self.config.embed_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )
        self.vector_repo = vector_repo
        self.app_config = app_config

    def chunk_and_embed(self, text: str) -> List[Dict[str, Any]]:
        """Returns [{"content", "embedding", "chunk_index", "token_count"}, ...]"""
        if not text or not text.strip():
            return []

        pieces = self.splitter.split_text(text)

        logger.debug(
            "chunking_complete",
            extra={"event": "chunking_complete", "num_chunks": len(pieces)},
        )

        vectors = self.embeddings.embed_documents(pieces)

        logger.info(
            "embedding_complete",
            extra={
                "event": "embedding_complete",
                "num_chunks": len(pieces),
                "num_vectors": len(vectors),
                "model": self.config.embed_model,
            },
        )

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
        logger.debug(
            "embedding_query",
            extra={"event": "embedding_query", "query_len": len(query or "")},
        )
        return self.embeddings.embed_query(query)

    def ingest_message(self, message_id: int, chat_id: int, user_id: int, role: str, content: str) -> None:
        if role == "tool" or not content or not content.strip():
            logger.debug(
                "message_ingest_skipped",
                extra={
                    "event": "message_ingest_skipped",
                    "message_id": message_id,
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "role": role,
                    "reason": "empty_content" if not content or not content.strip() else "tool_role",
                },
            )
            return
        try:
            chunks = self.chunk_and_embed(content)
            if chunks:
                self.vector_repo.insert_message_chunks(
                    message_id, chat_id, user_id, role, chunks
                )
                logger.info(
                    "message_ingested",
                    extra={
                        "event": "message_ingested",
                        "message_id": message_id,
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "role": role,
                        "num_chunks": len(chunks),
                    },
                )
            else:
                logger.info(
                    "message_ingest_no_chunks",
                    extra={
                        "event": "message_ingest_no_chunks",
                        "message_id": message_id,
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "role": role,
                    },
                )
        except Exception as e:
            logger.exception(
                "message_ingest_failed",
                extra={
                    "event": "message_ingest_failed",
                    "message_id": message_id,
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "role": role,
                    "error": str(e),
                },
            )
            pass  # don't let embedding failures break the chat flow

    def ingest_documents(self, chat_id: int, user_id: int, parsedFileData: List[Dict[str, Any]]) -> None:
        """
        For each parsed file with extractable text, store it as a `documents` row,
        chunk + embed it, and store chunks in `document_chunks`. Mutates each
        entry in parsedFileData in place, adding entry["document_id"], so
        the caller can persist the link (e.g. attachment records).
        """
        for entry in parsedFileData:
            if entry["kind"] not in ("text", "pdf", "docx") or not entry.get("text"):
                logger.debug(
                    "document_ingest_skipped",
                    extra={
                        "event": "document_ingest_skipped",
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "filename": entry.get("filename"),
                        "kind": entry.get("kind"),
                    },
                )
                continue

            doc_id = self.vector_repo.insert_document(
                chat_id=chat_id,
                user_id=user_id,
                filename=entry["filename"],
                path=entry["path"],
                content_type=entry["content_type"],
                kind=entry["kind"],
                char_count=len(entry["text"]),
                is_inlined=len(entry["text"]) <= self.app_config.inline_text_char_limit,
            )
            entry["document_id"] = doc_id

            logger.info(
                "document_created",
                extra={
                    "event": "document_created",
                    "document_id": doc_id,
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "filename": entry["filename"],
                    "kind": entry["kind"],
                    "char_count": len(entry["text"]),
                },
            )

            try:
                chunks = self.chunk_and_embed(entry["text"])
                if chunks:
                    self.vector_repo.insert_chunks(doc_id, chat_id, user_id, chunks)
                    self.vector_repo.update_document_status(doc_id, "embedded")
                    logger.info(
                        "document_embedded",
                        extra={
                            "event": "document_embedded",
                            "document_id": doc_id,
                            "chat_id": chat_id,
                            "user_id": user_id,
                            "num_chunks": len(chunks),
                        },
                    )
                else:
                    self.vector_repo.update_document_status(doc_id, "skipped")
                    logger.info(
                        "document_embed_skipped",
                        extra={
                            "event": "document_embed_skipped",
                            "document_id": doc_id,
                            "chat_id": chat_id,
                            "user_id": user_id,
                        },
                    )
            except Exception as e:
                self.vector_repo.update_document_status(doc_id, "failed")
                logger.exception(
                    "document_embed_failed",
                    extra={
                        "event": "document_embed_failed",
                        "document_id": doc_id,
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "error": str(e),
                    },
                )

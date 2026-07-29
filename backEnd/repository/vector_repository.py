import os
from typing import Any, Dict, List, Optional
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector
import numpy as np

from services.env_service import EnvService


class VectorRepository:
    def __init__(self, minconn: int = 1, maxconn: int = 5):
        db_config = EnvService().get_db_config()
        conn_str = (
            f"host={db_config.host} "
            f"port={db_config.port} "
            f"dbname={db_config.name} "
            f"user={db_config.user} "
            f"password={db_config.password}"
        )
        self.pool = SimpleConnectionPool(minconn, maxconn, dsn=conn_str)

    def _get_conn(self):
        conn = self.pool.getconn()
        register_vector(conn)
        return conn

    def _put_conn(self, conn) -> None:
        self.pool.putconn(conn)

    def insert_document(self, chat_id: int, user_id: int, filename: str, path: str,
                        content_type: Optional[str], kind: str, char_count: Optional[int],
                        is_inlined: bool = False) -> int:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (chat_id, user_id, filename, path, content_type, kind, char_count, is_inlined)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (chat_id, user_id, filename, path, content_type, kind, char_count, is_inlined),
                )
                doc_id = cur.fetchone()[0]
            conn.commit()
            return doc_id
        finally:
            self._put_conn(conn)

    def update_document_status(self, document_id: int, status: str) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                    SET status = %s, embedded_at = CASE WHEN %s = 'embedded' THEN now() ELSE embedded_at END
                    WHERE id = %s
                    """,
                    (status, status, document_id),
                )
            conn.commit()
        finally:
            self._put_conn(conn)

    def insert_chunks(self, document_id: int, chat_id: int, user_id: int,
                      chunks: List[Dict[str, Any]]) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO document_chunks
                    (document_id, chat_id, user_id, chunk_index, content, embedding, token_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (document_id, chat_id, user_id, c["chunk_index"], c["content"],
                         np.array(c["embedding"], dtype=np.float32), c.get("token_count"))  # <-- convert
                        for c in chunks
                    ],
                )
            conn.commit()
        finally:
            self._put_conn(conn)

    def similarity_search(self, chat_id: int, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        print("Semantic search called")
        query_vec = np.array(query_embedding, dtype=np.float32)  # <-- convert
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT dc.content,
                           dc.chunk_index,
                           d.filename,
                           d.id AS document_id,
                           dc.embedding <=> %s AS distance
                    FROM document_chunks dc
                        JOIN documents d
                    ON d.id = dc.document_id
                    WHERE dc.chat_id = %s
                    ORDER BY dc.embedding <=> %s
                        LIMIT %s
                    """,
                    (query_vec, chat_id, query_vec, k),  # <-- use query_vec
                )
                return [dict(row) for row in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def delete_document(self, document_id: int) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))  # cascades to chunks
            conn.commit()
        finally:
            self._put_conn(conn)

    def list_documents(self, chat_id: int) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, filename, kind, status, created_at FROM documents WHERE chat_id = %s ORDER BY created_at",
                    (chat_id,),
                )
                return [dict(row) for row in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def close(self) -> None:
        self.pool.closeall()

    def insert_message_chunks(self, message_id: int, chat_id: int, user_id: int,
                              role: str, chunks: List[Dict[str, Any]]) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO message_chunks
                        (message_id, chat_id, user_id, role, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (message_id, chat_id, user_id, role, c["chunk_index"], c["content"], np.array(c["embedding"], dtype=np.float32))
                        for c in chunks
                    ],
                )
            conn.commit()
        finally:
            self._put_conn(conn)

    def similarity_search_messages(self, user_id: int, query_embedding: List[float],
                                   chat_id: Optional[int] = None, k: int = 6) -> list[
        dict[Any, Any] | dict[str, Any] | dict[str, str] | dict[bytes, bytes]]:
        """
        chat_id=None -> searches across ALL of the user's past conversations (long-term memory).
        chat_id=<id> -> searches only within that conversation (in-chat recall for long threads).
        """
        query_vec = np.array(query_embedding, dtype=np.float32)
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if chat_id is not None:
                    cur.execute(
                        """
                        SELECT content,
                               role,
                               chat_id,
                               message_id,
                               chunk_index,
                               embedding <=> %s AS distance
                        FROM message_chunks
                        WHERE chat_id = %s
                        ORDER BY embedding <=> %s
                            LIMIT %s
                        """,
                        (query_vec, chat_id, query_vec, k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT content,
                               role,
                               chat_id,
                               message_id,
                               chunk_index,
                               embedding <=> %s AS distance
                        FROM message_chunks
                        WHERE user_id = %s
                        ORDER BY embedding <=> %s
                            LIMIT %s
                        """,
                        (query_vec, user_id, query_vec, k),
                    )
                return [dict(row) for row in cur.fetchall()]
        finally:
            self._put_conn(conn)
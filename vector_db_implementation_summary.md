# Vector DB (pgvector) Implementation — Summary

Adds RAG-over-uploaded-documents to the existing chat system. Every uploaded
text/pdf/docx file is now embedded and stored in Postgres via `pgvector`, and
the agent gets a new `search_knowledge_base` tool to query it. Small files are
still inlined in full in the prompt (as before); large files are truncated in
the prompt with a pointer telling the agent to use the retrieval tool instead.

Frontend work is deferred — backend-only for now.

---

## 1. Database schema (new)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded file
CREATE TABLE documents (
    id           BIGSERIAL PRIMARY KEY,
    chat_id      BIGINT NOT NULL REFERENCES conversations(chat_id) ON DELETE CASCADE,
    user_id      BIGINT NOT NULL,
    filename     TEXT NOT NULL,
    path         TEXT NOT NULL,
    content_type TEXT,
    kind         TEXT NOT NULL,                 -- text | pdf | docx | image | unsupported | error
    status       TEXT NOT NULL DEFAULT 'pending', -- pending | embedded | failed | skipped
    char_count   INTEGER,
    is_inlined   BOOLEAN NOT NULL DEFAULT false, -- true if full text was small enough to inline in-prompt
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedded_at  TIMESTAMPTZ
);

-- One row per chunk of a document, with its embedding
CREATE TABLE document_chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chat_id      BIGINT NOT NULL,   -- denormalized for fast filtering
    user_id      BIGINT NOT NULL,   -- denormalized for fast filtering
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    embedding    VECTOR(768) NOT NULL,  -- 768 = nomic-embed-text via Ollama; change if embedding model changes
    token_count  INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX document_chunks_chat_id_idx ON document_chunks (chat_id);
CREATE INDEX document_chunks_user_id_idx ON document_chunks (user_id);
CREATE INDEX documents_chat_id_idx ON documents (chat_id);
```

**Note:** `VECTOR(768)` assumes `nomic-embed-text`. If a different embedding
model is used (e.g. OpenAI `text-embedding-3-small` = 1536 dims), the column
dimension must be changed before any data is inserted — it can't be altered
in place once populated.

### Existing table — no structural change required

`messages.attachments` (JSON/JSONB) is reused as-is. The only change is at
the application level: each attachment record now also carries a
`document_id` linking back to the `documents` table (see `LlmService`
changes below).

Optional (not required): if `messages.attachments` is currently `TEXT`
rather than `JSONB`:
```sql
ALTER TABLE messages ALTER COLUMN attachments TYPE JSONB USING attachments::jsonb;
```

---

## 2. New file: `repositories/vector_repository.py`

Handles all Postgres/pgvector reads and writes. Builds its own DSN from the
same env vars `ConversationRepository` uses (`DB_HOST`, `DB_PORT`, `DB_NAME`,
`DB_USER`, `DB_PASSWORD`), uses `psycopg2` with a `SimpleConnectionPool`, and
registers the `pgvector` adapter (`register_vector`) on each checked-out
connection so Python `List[float]` values serialize correctly to/from the
`vector` column type.

Methods:
- `insert_document(chat_id, user_id, filename, path, content_type, kind, char_count, is_inlined)` → `int` (new document id)
- `update_document_status(document_id, status)`
- `insert_chunks(document_id, chat_id, user_id, chunks)` — bulk insert via `executemany`
- `similarity_search(chat_id, query_embedding, k=4)` — cosine distance (`<=>`) search, scoped to a single `chat_id`
- `delete_document(document_id)` — cascades to chunks
- `list_documents(chat_id)`
- `close()` — closes the pool

```python
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector
from typing import Any, Dict, List, Optional


class VectorRepository:
    def __init__(self, minconn: int = 1, maxconn: int = 5):
        conn_str = (
            f"host={os.getenv('DB_HOST', 'localhost')} "
            f"port={os.getenv('DB_PORT', 5432)} "
            f"dbname={os.getenv('DB_NAME', 'postgres')} "
            f"user={os.getenv('DB_USER', 'postgres')} "
            f"password={os.getenv('DB_PASSWORD')}"
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
                         c["embedding"], c.get("token_count"))
                        for c in chunks
                    ],
                )
            conn.commit()
        finally:
            self._put_conn(conn)

    def similarity_search(self, chat_id: int, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT dc.content, dc.chunk_index, d.filename, d.id AS document_id,
                           dc.embedding <=> %s AS distance
                    FROM document_chunks dc
                    JOIN documents d ON d.id = dc.document_id
                    WHERE dc.chat_id = %s
                    ORDER BY dc.embedding <=> %s
                    LIMIT %s
                    """,
                    (query_embedding, chat_id, query_embedding, k),
                )
                return [dict(row) for row in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def delete_document(self, document_id: int) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
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
```

**Dependencies:** `pip install pgvector psycopg2-binary` (or `psycopg2` if compiling against system libpq).

---

## 3. New file: `services/embedding_service.py`

Chunks raw text (`RecursiveCharacterTextSplitter`, 800 chars / 100 overlap)
and embeds via Ollama's `nomic-embed-text` model.

```python
from typing import Any, Dict, List
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "nomic-embed-text"   # must match VECTOR(768) dimension in schema
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


class EmbeddingService:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def chunk_and_embed(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        pieces = self.splitter.split_text(text)
        vectors = self.embeddings.embed_documents(pieces)

        return [
            {
                "content": piece,
                "embedding": vector,
                "chunk_index": i,
                "token_count": max(1, len(piece) // 4),
            }
            for i, (piece, vector) in enumerate(zip(pieces, vectors))
        ]

    def embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)
```

---

## 4. New file: `tools/knowledge_base_tool.py`

Separate tool class (parallels `FolderReadTool`, `WebSearchTool`), bound to a
specific `chat_id` at construction time so retrieval is scoped per
conversation.

```python
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from repositories.vector_repository import VectorRepository
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
```

---

## 5. Additions to `LlmService`

### New constant (config/constants file)
```python
# Files at or under this size get their full text inlined into the prompt
# for THIS turn, in addition to being embedded. Files over this size get
# a short notice instead — the agent must use search_knowledge_base for them.
INLINE_TEXT_CHAR_LIMIT = 6000
```

### Constructor — add
```python
from repositories.vector_repository import VectorRepository
from services.embedding_service import EmbeddingService
from tools.knowledge_base_tool import KnowledgeBaseSearchTool

# inside __init__, after self.repo = ConversationRepository()
self.vector_repo = VectorRepository()
self.embedding_service = EmbeddingService()
```

### New method — `_ingest_documents`
Called right after `parse_files` in both `chatWithLlm` and
`chatWithLlmStream`. Stores each text-bearing file as a `documents` row,
chunks + embeds it, and stores chunks in `document_chunks`. Mutates each
entry in `parsedFileData` in place, adding `entry["document_id"]`.

```python
def _ingest_documents(self, chat_id: int, user_id: int,
                       parsedFileData: List[Dict[str, Any]]) -> None:
    for entry in parsedFileData:
        if entry["kind"] not in ("text", "pdf", "docx") or not entry.get("text"):
            continue

        doc_id = self.vector_repo.insert_document(
            chat_id=chat_id,
            user_id=user_id,
            filename=entry["filename"],
            path=entry["path"],
            content_type=entry["content_type"],
            kind=entry["kind"],
            char_count=len(entry["text"]),
            is_inlined=len(entry["text"]) <= INLINE_TEXT_CHAR_LIMIT,
        )
        entry["document_id"] = doc_id

        try:
            chunks = self.embedding_service.chunk_and_embed(entry["text"])
            if chunks:
                self.vector_repo.insert_chunks(doc_id, chat_id, user_id, chunks)
                self.vector_repo.update_document_status(doc_id, "embedded")
            else:
                self.vector_repo.update_document_status(doc_id, "skipped")
        except Exception:
            self.vector_repo.update_document_status(doc_id, "failed")
```

### Call site — in both `chatWithLlm` and `chatWithLlmStream`
Right after `parsedFileData = await self.parse_files(chat_id, files)`:
```python
parsedFileData = await self.parse_files(chat_id, files)
self._ingest_documents(chat_id, user_id, parsedFileData)   # NEW
attachment_records = self._build_attachment_records(parsedFileData)
```

### `_build_attachment_records` — add one field
```python
{
    "filename": entry["filename"],
    "path": entry["path"],
    "content_type": entry["content_type"],
    "kind": entry["kind"],
    "document_id": entry.get("document_id"),  # NEW
}
```

### `_build_enriched_conversation` — replaced body
Small files are inlined in full (unchanged behavior). Large files get a
short notice instead of their full text, since they were already chunked
and embedded.

```python
def _build_enriched_conversation(self, conversation: List[ChatMessage], parsedFileData: List[Dict[str, Any]]) -> \
    List[Dict[str, Any]]:
    agent_conversation = [{"role": m.role, "content": m.content} for m in conversation]

    extra_text = ""
    for entry in parsedFileData:
        text = entry.get("text")
        if not text:
            continue

        if len(text) <= INLINE_TEXT_CHAR_LIMIT:
            extra_text += f"\n\n---\nFile: {entry['filename']}\n{text}"
        else:
            extra_text += (
                f"\n\n---\nFile: {entry['filename']} "
                f"({len(text)} characters — too large to inline in full). "
                f"This file has been indexed; use the search_knowledge_base tool "
                f"with a specific query to retrieve relevant passages from it."
            )

    if extra_text and agent_conversation:
        agent_conversation[-1]["content"] += extra_text

    return agent_conversation
```

### Agent construction — in both `chatWithLlm` and `chatWithLlmStream`
Add the retrieval tool, bound to `chat_id`:
```python
kb_tool = KnowledgeBaseSearchTool(
    vector_repo=self.vector_repo,
    embedding_service=self.embedding_service,
    chat_id=chat_id,
)
agent = OllamaAgent(
    model_name=model_name,
    tools=self._fs_tools + [kb_tool],   # was: tools=self._fs_tools
    system_prompt=self.starter_system_prompt,
)
```

### System prompt — add a third tool entry
```
3. Search Knowledge Base
   - Use this tool to search the content of files the user has uploaded in this conversation.
   - Small files are already included in full in the current message — you do not
     need to call this tool for those unless the user asks about earlier files
     from previous turns in this same conversation.
   - Large files are NOT included in full — you must call this tool with a
     specific, targeted query to retrieve relevant passages from them.
   - Cite the source filename when you use information from it.
```

---

## 6. Optional API additions (`main.py`) — backend-only, no UI yet

```python
@app.get("/documents/{chat_id}")
async def list_documents(chat_id: int, llmService: LlmService = Depends(getLlmService)):
    return llmService.vector_repo.list_documents(chat_id)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, llmService: LlmService = Depends(getLlmService)):
    llmService.vector_repo.delete_document(document_id)
    return {"deleted": document_id}
```

---

## 7. Chat history chunking / embedding (long-term + in-conversation memory)

Extends the same pipeline to chat messages themselves, so the agent can
recall things either from earlier in the *same* long conversation, or from
a *different* past conversation entirely (long-term memory). One table +
one tool, with a `scope` parameter, since the chunk/embed logic is identical
to documents — only the `WHERE` clause on retrieval differs.

**Prerequisite check (done):** confirmed `ConversationRepository.create_message`
already does `INSERT ... RETURNING id` and returns `new_id` — no changes
needed there. The only change is capturing that returned id at each call site.

### Schema — new table
```sql
CREATE TABLE message_chunks (
    id           BIGSERIAL PRIMARY KEY,
    message_id   BIGINT NOT NULL,   -- references messages.id
    chat_id      BIGINT NOT NULL,
    user_id      BIGINT NOT NULL,
    role         TEXT NOT NULL,     -- user | assistant | system | tool
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    embedding    VECTOR(768) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (message_id, chunk_index)
);

CREATE INDEX message_chunks_embedding_idx
    ON message_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX message_chunks_chat_id_idx ON message_chunks (chat_id);
CREATE INDEX message_chunks_user_id_idx ON message_chunks (user_id);
```

### Additions to `repositories/vector_repository.py`
```python
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
                    (message_id, chat_id, user_id, role, c["chunk_index"], c["content"], c["embedding"])
                    for c in chunks
                ],
            )
        conn.commit()
    finally:
        self._put_conn(conn)

def similarity_search_messages(self, user_id: int, query_embedding: List[float],
                                chat_id: Optional[int] = None, k: int = 6) -> List[Dict[str, Any]]:
    """
    chat_id=None -> searches across ALL of the user's past conversations (long-term memory).
    chat_id=<id> -> searches only within that conversation (in-chat recall for long threads).
    """
    conn = self._get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if chat_id is not None:
                cur.execute(
                    """
                    SELECT content, role, chat_id, message_id, chunk_index,
                           embedding <=> %s AS distance
                    FROM message_chunks
                    WHERE chat_id = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (query_embedding, chat_id, query_embedding, k),
                )
            else:
                cur.execute(
                    """
                    SELECT content, role, chat_id, message_id, chunk_index,
                           embedding <=> %s AS distance
                    FROM message_chunks
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (query_embedding, user_id, query_embedding, k),
                )
            return [dict(row) for row in cur.fetchall()]
    finally:
        self._put_conn(conn)
```

### New file: `tools/chat_history_tool.py`
```python
from typing import Literal, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from repositories.vector_repository import VectorRepository
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
```

### Additions to `LlmService`

New helper:
```python
def _ingest_message(self, message_id: int, chat_id: int, user_id: int,
                     role: str, content: str) -> None:
    if role == "tool" or not content or not content.strip():
        return
    try:
        chunks = self.embedding_service.chunk_and_embed(content)
        if chunks:
            self.vector_repo.insert_message_chunks(message_id, chat_id, user_id, role, chunks)
    except Exception:
        pass  # don't let embedding failures break the chat flow
```

Every existing `self.repo.create_message(...)` call site (in both
`chatWithLlm` and `chatWithLlmStream` — user message, image-unavailable
assistant message, and the `new_messages` / streamed-message loops) is
updated to capture the returned id and call `_ingest_message`, e.g.:

```python
message_id = self.repo.create_message(SimpleNamespace(...))
self._ingest_message(message_id, chat_id, user_id, "user", message)  # NEW
```

Agent construction gets a third tool alongside `kb_tool`:
```python
chat_history_tool = ChatHistorySearchTool(
    vector_repo=self.vector_repo,
    embedding_service=self.embedding_service,
    chat_id=chat_id,
    user_id=user_id,
)
agent = OllamaAgent(
    model_name=model_name,
    tools=self._fs_tools + [kb_tool, chat_history_tool],
    system_prompt=self.starter_system_prompt,
)
```

System prompt — 4th tool entry:
```
4. Search Chat History
   - Use this to recall things from past messages that are not visible in the
     current context — either earlier in this same conversation (if it's long)
     or from a different past conversation the user refers to.
   - Prefer scope="this_conversation" unless the user clearly references a
     separate, earlier chat.
```

### Optional: bound how much history gets inlined per turn
Once messages are embedded, long chats no longer need every row inlined
verbatim — older messages can be left to `search_chat_history` instead:
```python
MAX_INLINE_MESSAGES = 30  # tune to your model's context window

rows = self.repo.get_messages(chat_id)
if len(rows) > MAX_INLINE_MESSAGES:
    rows = rows[:1] + rows[-(MAX_INLINE_MESSAGES - 1):]  # keep system prompt + recent tail
```

---

## Files touched / added in this pass

| File | Status |
|---|---|
| `repositories/vector_repository.py` | **New** — documents/chunks methods + `insert_message_chunks`, `similarity_search_messages` |
| `services/embedding_service.py` | **New** |
| `tools/knowledge_base_tool.py` | **New** |
| `tools/chat_history_tool.py` | **New** |
| `services/llm_service.py` (or equivalent) | Modified — constructor, `_ingest_documents`, `_ingest_message` (new), `_build_attachment_records`, `_build_enriched_conversation`, agent construction in `chatWithLlm` / `chatWithLlmStream` now includes `kb_tool` + `chat_history_tool`, all `create_message` call sites updated to capture returned id, system prompt |
| `main.py` | Modified (optional) — two new document endpoints |
| Postgres schema | New tables: `documents`, `document_chunks`, `message_chunks` (+ optional `messages.attachments` type change) |
| `repositories/conversation_repository.py` | No change required — `create_message` already returns `id` via `RETURNING id` |

## Not yet done
- Frontend: upload management UI, document list/delete UI, tool-call rendering for `search_knowledge_base` / `search_chat_history` in chat bubbles.
- Embedding model dimension is hardcoded to 768 (`nomic-embed-text`) — changing embedding models requires a schema migration.
- Decide whether to bound inlined message history (`MAX_INLINE_MESSAGES`) now or only once chats grow long enough to need it.
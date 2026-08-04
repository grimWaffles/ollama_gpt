# Contextual Memory Architecture — Project Summary

## Overview

The chat service (`LlmService`) has moved from loading and replaying full
conversation history on every turn to assembling **retrieved context** from
a vector store. The service remains the single orchestration layer, but its
job changed:

- **Before:** load every stored message for the chat, append the new user
  message, send the whole thing to the model.
- **Now:** persist the new user message, embed it, run similarity search
  against chat memory + uploaded knowledge, assemble a compact prompt
  (system prompt + retrieved memories/snippets + current message only),
  and invoke the model with that.

This keeps prompt size roughly constant regardless of how long a
conversation runs, while preserving long-term recall through embeddings,
periodic summarization, and archival.

The original `chatWithLlm` / `chatWithLlmStream` methods that loaded and
replayed raw history have been removed. `chat_with_llm_2` is now the only
orchestration path.

---

## Flow

### First message in a new conversation
1. Create the conversation row.
2. Persist the user's message, embed it.
3. Send **only** the system prompt + user message to the model — no
   retrieval, since there's nothing in the store yet.
4. Persist + embed the assistant's reply.

### Every subsequent message
1. Persist the new user message first, then embed it.
2. Embed the message text as the retrieval query.
3. Run similarity search against:
   - `message_chunks` scoped to this chat (chat memory)
   - `chat_summaries` scoped to this chat (compressed long-term memory)
   - `document_chunks` scoped to this chat (uploaded knowledge base)
4. Rank, deduplicate, and optionally threshold the combined results by
   relevance (cosine distance) and recency.
5. Build a compact prompt: system prompt + retrieved memories + retrieved
   knowledge snippets + the current user message. No raw history is
   loaded from `messages` in this path.
6. Invoke the agent with the compact prompt.
7. Persist + embed the assistant's reply.
8. If the unarchived message count has crossed a threshold, summarize and
   archive the oldest cluster of messages so the working set stays bounded.

`get_messages` (raw full history) is retained only for administrative /
display purposes (e.g. rendering full chat history in the UI) — it is no
longer used to build the model's context.

---

## Database Schema

### Existing tables (unchanged)
`conversations`, `documents`, `document_chunks`, `message_chunks` — as
previously defined, plus one new column on `messages` and one new table.

### `messages` — new column
```sql
ALTER TABLE messages ADD COLUMN archived BOOLEAN NOT NULL DEFAULT false;
```
Messages that have been rolled into a summary are marked `archived = true`.
They remain in the table (audit/admin purposes) but are excluded from the
active working set once archived.

### `chat_summaries` (new)
```sql
CREATE TABLE chat_summaries (
    id                BIGSERIAL PRIMARY KEY,
    chat_id           BIGINT NOT NULL REFERENCES conversations(chat_id) ON DELETE CASCADE,
    user_id           BIGINT NOT NULL,
    content           TEXT NOT NULL,
    embedding         VECTOR(768) NOT NULL,
    start_sequence_no INTEGER NOT NULL,
    end_sequence_no   INTEGER NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chat_summaries_chat_range_key UNIQUE (chat_id, start_sequence_no, end_sequence_no)
);

CREATE INDEX chat_summaries_embedding_idx ON chat_summaries USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chat_summaries_chat_id_idx ON chat_summaries (chat_id);
```

One row per compressed cluster of older messages. The unique constraint on
`(chat_id, start_sequence_no, end_sequence_no)` makes summary inserts
idempotent — re-running summarization over the same range is a no-op
rather than a duplicate row.

---

## `repository/conversation_repository.py` — additions

```python
def get_last_sequence_no(self, chat_id: int) -> int:
    query = "SELECT COALESCE(MAX(sequence_no), 0) FROM messages WHERE chat_id = %s"
    try:
        with self.connection.cursor() as cursor:
            cursor.execute(query, (chat_id,))
            result = cursor.fetchone()
            return result[0] if result else 0
    except Exception as e:
        print(f"Error fetching last sequence_no for chat {chat_id}: {e}")
        self.connection.rollback()
        return 0

def get_unarchived_message_count(self, chat_id: int) -> int:
    query = "SELECT COUNT(*) FROM messages WHERE chat_id = %s AND archived = false"
    try:
        with self.connection.cursor() as cursor:
            cursor.execute(query, (chat_id,))
            result = cursor.fetchone()
            return result[0] if result else 0
    except Exception as e:
        print(f"Error counting unarchived messages for chat {chat_id}: {e}")
        self.connection.rollback()
        return 0

def get_oldest_unarchived_messages(self, chat_id: int, limit: int):
    """Oldest-first, excluding tool messages (nothing to summarize there)."""
    query = """
            SELECT id, role, message, sequence_no
            FROM messages
            WHERE chat_id = %s AND archived = false AND role != 'tool'
            ORDER BY sequence_no ASC
            LIMIT %s
            """
    try:
        with self.connection.cursor() as cursor:
            cursor.execute(query, (chat_id, limit))
            return cursor.fetchall()
    except Exception as e:
        print(f"Error fetching oldest unarchived messages for chat {chat_id}: {e}")
        self.connection.rollback()
        return []

def archive_messages(self, message_ids: list[int]) -> None:
    if not message_ids:
        return
    query = "UPDATE messages SET archived = true WHERE id = ANY(%s)"
    try:
        with self.connection.cursor() as cursor:
            cursor.execute(query, (message_ids,))
        self.connection.commit()
    except Exception as e:
        print(f"Error archiving messages {message_ids}: {e}")
        self.connection.rollback()
        raise
```

All other methods (`create_conversation`, `create_message`, `get_messages`,
etc.) are unchanged.

---

## `repository/vector_repository.py` — updates

All chunk/summary insert methods are now idempotent via `ON CONFLICT DO
NOTHING`, so a duplicate ingest call (retry, double-invocation, etc.)
becomes a harmless no-op instead of raising `UniqueViolation`.

```python
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
                ON CONFLICT (document_id, chunk_index) DO NOTHING
                """,
                [
                    (document_id, chat_id, user_id, c["chunk_index"], c["content"],
                     np.array(c["embedding"], dtype=np.float32), c.get("token_count"))
                    for c in chunks
                ],
            )
        conn.commit()
    finally:
        self._put_conn(conn)


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
                ON CONFLICT (message_id, chunk_index) DO NOTHING
                """,
                [
                    (message_id, chat_id, user_id, role, c["chunk_index"], c["content"],
                     np.array(c["embedding"], dtype=np.float32))
                    for c in chunks
                ],
            )
        conn.commit()
    finally:
        self._put_conn(conn)


def insert_summary(self, chat_id: int, user_id: int, content: str,
                    embedding: List[float], start_sequence_no: int, end_sequence_no: int) -> Optional[int]:
    conn = self._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_summaries
                    (chat_id, user_id, content, embedding, start_sequence_no, end_sequence_no)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (chat_id, start_sequence_no, end_sequence_no) DO NOTHING
                RETURNING id
                """,
                (chat_id, user_id, content, np.array(embedding, dtype=np.float32),
                 start_sequence_no, end_sequence_no),
            )
            row = cur.fetchone()
        conn.commit()
        return row[0] if row else None
    finally:
        self._put_conn(conn)


def similarity_search_summaries(self, chat_id: int, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
    query_vec = np.array(query_embedding, dtype=np.float32)
    conn = self._get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT content, start_sequence_no, end_sequence_no, created_at,
                       embedding <=> %s AS distance
                FROM chat_summaries
                WHERE chat_id = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vec, chat_id, query_vec, k),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        self._put_conn(conn)
```

`similarity_search_messages` gains `created_at` in its `SELECT` list (both
branches) so the service layer can apply a recency threshold:

```python
SELECT content, role, chat_id, message_id, chunk_index, created_at,
       embedding <=> %s AS distance
FROM message_chunks
...
```

`insert_document`, `similarity_search`, `delete_document`,
`list_documents`, `close` are unchanged.

---

## `services/embedding_service.py` — additions

```python
def ingest_summary(self, chat_id: int, user_id: int, content: str,
                    start_sequence_no: int, end_sequence_no: int) -> Optional[int]:
    """Summaries are stored as a single dense vector, not chunked — they're
    already compact by construction."""
    if not content or not content.strip():
        return None
    try:
        embedding = self.embed_query(content)
        summary_id = self.vector_repo.insert_summary(
            chat_id, user_id, content, embedding, start_sequence_no, end_sequence_no
        )
        logger.info("summary_ingested", extra={
            "event": "summary_ingested", "chat_id": chat_id, "user_id": user_id,
            "summary_id": summary_id, "start_sequence_no": start_sequence_no,
            "end_sequence_no": end_sequence_no,
        })
        return summary_id
    except Exception as e:
        logger.exception("summary_ingest_failed", extra={
            "event": "summary_ingest_failed", "chat_id": chat_id, "error": str(e),
        })
        return None
```

`chunk_and_embed`, `embed_query`, `ingest_message`, `ingest_documents` are
unchanged.

---

## `services/llm_service.py` — `chat_with_llm_2` (rewritten)

This is now the only chat orchestration method. The previous
`chatWithLlm` / `chatWithLlmStream` methods — which loaded full history
from the DB and replayed it on every turn — have been removed.

```python
async def chat_with_llm_2(self, user_id: int, chat_id: int, model_name: str,
                           message: str, files: Optional[List[UploadFile]] = None) -> tuple[int, List[ChatMessage]]:
    files = files or []
    try:
        is_new_conversation = not chat_id or chat_id == 0

        if is_new_conversation:
            max_chat_id = self.convo_repo.get_max_chat_id()
            chat_id = self.convo_repo.create_conversation(
                SimpleNamespace(
                    chatId=chat_id,
                    userId=user_id,
                    chatName=f"Conversation #{max_chat_id + 1}",
                    created_at=datetime.now(),
                )
            )

        last_sequence_no = self.convo_repo.get_last_sequence_no(chat_id)

        # --- Files: ingest into the knowledge base regardless of turn number ---
        parsedFileData = await self._parse_files(chat_id, files)
        attachment_records = None
        if parsedFileData:
            self.embedding_service.ingest_documents(chat_id, user_id, parsedFileData)
            attachment_records = self._build_attachment_records(parsedFileData)

        # --- 1. Persist the user's message first, then embed it ---
        now = datetime.now()
        last_sequence_no += 1
        user_message_id = self.convo_repo.create_message(
            SimpleNamespace(
                id=0,
                chatId=chat_id,
                role="user",
                message=message,
                sequenceNo=last_sequence_no,
                created_at=now,
                attachments=json.dumps(attachment_records) if attachment_records else None,
            )
        )

        if self.app_config.use_embedding:
            self.embedding_service.ingest_message(user_message_id, chat_id, user_id, "user", message)

        # --- Image-attachment short-circuit (unchanged behavior) ---
        if parsedFileData and _has_image(parsedFileData):
            assistant_msg = ChatMessage(role="assistant", content=self.app_config.image_unavailable_message)
            last_sequence_no += 1
            self.convo_repo.create_message(
                SimpleNamespace(
                    id=0, chatId=chat_id, role="assistant", message=assistant_msg.content,
                    sequenceNo=last_sequence_no, created_at=datetime.now(), attachments=None,
                )
            )
            return chat_id, [assistant_msg]

        # --- 2. Build the prompt: full context only on turn 1, retrieved context after ---
        if is_new_conversation:
            agent_conversation = [
                {"role": "system", "content": self.starter_system_prompt},
                {"role": "user", "content": self._compose_user_content(message, parsedFileData)},
            ]
        else:
            retrieved = self._retrieve_context(chat_id, user_id, message)
            agent_conversation = self._build_compact_prompt(message, retrieved, parsedFileData)

        # --- 3. Tools ---
        embedding_tools = []
        if self.app_config.use_embedding:
            kb_tool = KnowledgeBaseSearchTool(
                vector_repo=self.vector_repo, embedding_service=self.embedding_service, chat_id=chat_id,
            )
            chat_history_tool = ChatHistorySearchTool(
                vector_repo=self.vector_repo, embedding_service=self.embedding_service,
                chat_id=chat_id, user_id=user_id,
            )
            embedding_tools = [kb_tool, chat_history_tool]

        mcp_tools = []

        agent = OllamaAgent(
            model_name=model_name,
            tools=self._fs_tools + mcp_tools + embedding_tools,
            system_prompt=self.starter_system_prompt,
        )

        # --- 4. Invoke with the compact prompt, not raw history ---
        full_response = await agent.ainvoke(agent_conversation)
        new_messages = full_response[len(agent_conversation):]

        # --- 5. Persist + embed assistant output ---
        for new_msg in new_messages:
            last_sequence_no += 1
            new_msg_id = self.convo_repo.create_message(
                SimpleNamespace(
                    id=0, chatId=chat_id, role=new_msg.role, message=new_msg.content,
                    sequenceNo=last_sequence_no, created_at=datetime.now(), attachments=None,
                )
            )
            if self.app_config.use_embedding and new_msg.role not in ("tool",):
                self.embedding_service.ingest_message(new_msg_id, chat_id, user_id, new_msg.role, new_msg.content)

        # --- 6. Periodic summarization/archival so prompt size stays bounded ---
        if self.app_config.use_embedding:
            await self._maybe_summarize_old_messages(chat_id, user_id)

        return chat_id, new_messages

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return chat_id, []
```

### Attachment records

```python
def _build_attachment_records(self, parsedFileData: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "filename": entry["filename"],
            "path": entry["path"],
            "content_type": entry["content_type"],
            "kind": entry["kind"],
            "document_id": entry.get("document_id"),
        }
        for entry in parsedFileData
    ]
```

### Retrieval + ranking

```python
def _retrieve_context(self, chat_id: int, user_id: int, query: str) -> Dict[str, list]:
    query_embedding = self.embedding_service.embed_query(query)

    memory_k = getattr(self.app_config, "memory_k", 6)
    summary_k = getattr(self.app_config, "summary_k", 3)
    kb_k = getattr(self.app_config, "kb_k", 4)

    memory_hits = self.vector_repo.similarity_search_messages(
        user_id=user_id, query_embedding=query_embedding, chat_id=chat_id, k=memory_k,
    )
    summary_hits = self.vector_repo.similarity_search_summaries(
        chat_id=chat_id, query_embedding=query_embedding, k=summary_k,
    )
    kb_hits = self.vector_repo.similarity_search(chat_id, query_embedding, k=kb_k)

    # Summaries are curated to already hold the categories we want to
    # prioritize (preferences, decisions, unresolved tasks), so give them a
    # small edge over raw individual turns when ranking.
    summary_boost = getattr(self.app_config, "summary_relevance_boost", 0.85)
    for hit in summary_hits:
        hit["distance"] = hit.get("distance", 1.0) * summary_boost
        hit["is_summary"] = True

    combined_memory = self._rank_and_filter(
        memory_hits + summary_hits,
        max_distance=getattr(self.app_config, "memory_relevance_threshold", None),
        max_age_days=getattr(self.app_config, "memory_recency_days", None),
        limit=getattr(self.app_config, "memory_result_limit", 8),
    )
    filtered_kb = self._rank_and_filter(
        kb_hits,
        max_distance=getattr(self.app_config, "kb_relevance_threshold", None),
        limit=getattr(self.app_config, "kb_result_limit", 4),
    )

    return {"memories": combined_memory, "knowledge": filtered_kb}


def _rank_and_filter(self, results: list, max_distance: Optional[float] = None,
                      max_age_days: Optional[float] = None, limit: int = 8) -> list:
    seen_content = set()
    deduped = []
    now = datetime.now()

    for r in sorted(results, key=lambda r: r.get("distance", 1.0)):
        content_key = (r.get("content") or "").strip()
        if not content_key or content_key in seen_content:
            continue

        if max_distance is not None and r.get("distance", 1.0) > max_distance:
            continue

        # Recency threshold only applies to raw memories with a created_at;
        # summaries represent long-term memory and are exempt by design.
        if max_age_days is not None and not r.get("is_summary") and r.get("created_at"):
            age_days = (now - r["created_at"]).total_seconds() / 86400
            if age_days > max_age_days:
                continue

        seen_content.add(content_key)
        deduped.append(r)
        if len(deduped) >= limit:
            break

    return deduped
```

### Compact prompt assembly

```python
def _compose_user_content(self, message: str, parsedFileData: List[Dict[str, Any]]) -> str:
    if not parsedFileData:
        return message

    inline_limit = getattr(self.app_config, "inline_text_char_limit", 6000)
    extra_text = ""
    for entry in parsedFileData:
        text = entry.get("text")
        if not text:
            continue
        if len(text) <= inline_limit:
            extra_text += f"\n\n---\nFile: {entry['filename']}\n{text}"
        else:
            extra_text += (
                f"\n\n---\nFile: {entry['filename']} ({len(text)} characters — "
                f"too large to inline). This file has been indexed; use "
                f"search_knowledge_base with a specific query to retrieve from it."
            )
    return message + extra_text


def _build_compact_prompt(self, message: str, retrieved: Dict[str, list],
                           parsedFileData: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system_content = self.starter_system_prompt
    context_sections = []

    if retrieved["memories"]:
        memory_block = "\n\n".join(
            f"[{'summary' if r.get('is_summary') else r.get('role', 'memory')}]\n{r['content']}"
            for r in retrieved["memories"]
        )
        context_sections.append(f"### Relevant memory from this conversation\n{memory_block}")

    if retrieved["knowledge"]:
        knowledge_block = "\n\n".join(
            f"[Source: {r['filename']}]\n{r['content']}"
            for r in retrieved["knowledge"]
        )
        context_sections.append(f"### Relevant knowledge from uploaded documents\n{knowledge_block}")

    if context_sections:
        system_content += "\n\n" + "\n\n".join(context_sections)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": self._compose_user_content(message, parsedFileData)},
    ]
```

### Summarization / archival

```python
async def _maybe_summarize_old_messages(self, chat_id: int, user_id: int) -> None:
    cluster_size = getattr(self.app_config, "summary_cluster_size", 20)
    keep_recent = getattr(self.app_config, "summary_keep_recent", 6)

    unarchived_count = self.convo_repo.get_unarchived_message_count(chat_id)
    if unarchived_count < cluster_size + keep_recent:
        return

    to_summarize = self.convo_repo.get_oldest_unarchived_messages(chat_id, cluster_size)
    if not to_summarize:
        return

    transcript = "\n".join(f"{role}: {text}" for (_id, role, text, _seq) in to_summarize)
    summary_text = await self._generate_summary(transcript)

    start_seq = to_summarize[0][3]
    end_seq = to_summarize[-1][3]
    self.embedding_service.ingest_summary(chat_id, user_id, summary_text, start_seq, end_seq)

    self.convo_repo.archive_messages([row[0] for row in to_summarize])


async def _generate_summary(self, transcript: str) -> str:
    summarizer_prompt = (
        "Compress the following conversation excerpt into dense, durable factual "
        "notes: user preferences, decisions made, unresolved tasks, and any key "
        "facts worth remembering. Bullet points, no filler, no meta-commentary.\n\n"
        f"{transcript}"
    )
    summarizer_model = getattr(self.app_config, "summary_model_name", None) or self.cloud_models[1][1]
    summarizer = OllamaAgent(
        model_name=summarizer_model,
        tools=[],
        system_prompt="You compress conversation excerpts into compact factual memory notes.",
    )
    result = await summarizer.ainvoke([{"role": "user", "content": summarizer_prompt}])
    return result[-1].content if result else transcript[:800]
```

---

## `agents/ollama_agent.py`

Unchanged from the last revision — supports both `ChatMessage` objects and
raw `{"role", "content"}` dicts as conversation entries (`ConversationEntry
= Union[ChatMessage, Dict[str, Any]]`), has both `invoke` and `ainvoke`,
and filters out `"tool"`-role entries before sending to the model since
tool-call ids aren't persisted and can't be reconstructed on replay.

---

## `tools/knowledge_base_tool.py` / `tools/chat_history_tool.py`

Unchanged. Both remain available to the agent as on-demand retrieval tools
for cases where the compact prompt's automatic retrieval didn't surface
what the model needs — e.g. a very specific follow-up query the model
decides to issue itself mid-turn.

---

## Known idempotency fix

Symptom seen in testing:
```
psycopg2.errors.UniqueViolation: duplicate key value violates unique
constraint "message_chunks_message_id_chunk_index_key"
DETAIL:  Key (message_id, chunk_index)=(5, 0) already exists.
```

Root cause: `ingest_message` (or `ingest_documents` / `ingest_summary`)
being invoked twice for the same id — either a duplicate request from the
frontend, or a reused id variable in the service layer. The `ON CONFLICT
DO NOTHING` guards added to `insert_chunks`, `insert_message_chunks`, and
`insert_summary` make repeated ingestion calls safe (no crash, no
duplicate rows), but they don't address the underlying double-invocation
— that still wastes an embedding call to Ollama each time it happens and
should be tracked down separately (check for duplicate POSTs from the
Angular client, or a reused message id across call sites in
`chat_with_llm_2`).

---

## Config values referenced (not yet centralized)

These are read via `getattr(self.app_config, ..., default)` and should be
promoted to real fields on `AppConfig` once settled:

| Name | Default | Purpose |
|---|---|---|
| `memory_k` | 6 | raw message-chunk hits to retrieve per query |
| `summary_k` | 3 | summary hits to retrieve per query |
| `kb_k` | 4 | document-chunk hits to retrieve per query |
| `memory_relevance_threshold` | `None` (off) | max cosine distance for memory hits |
| `memory_recency_days` | `None` (off) | max age for raw (non-summary) memory hits |
| `memory_result_limit` | 8 | cap on combined memory results after ranking |
| `kb_relevance_threshold` | `None` (off) | max cosine distance for KB hits |
| `kb_result_limit` | 4 | cap on KB results after ranking |
| `summary_relevance_boost` | 0.85 | multiplier applied to summary distances to prioritize them |
| `summary_cluster_size` | 20 | number of oldest messages summarized per pass |
| `summary_keep_recent` | 6 | minimum unarchived messages to always leave unsummarized |
| `summary_model_name` | falls back to `cloud_models[1][1]` | model used for summarization |
| `inline_text_char_limit` | 6000 | file size below which text is inlined vs. pointer-only |

---

## Not yet done
- Frontend: no changes needed for this pass, but the UI's "load messages"
  path should be confirmed to still use `get_messages` (unaffected) rather
  than anything from the new retrieval path.
- Root-cause the double-ingest call seen in the `UniqueViolation` logs
  (frontend duplicate request vs. reused id in `chat_with_llm_2`).
- Decide whether assistant messages should also be recency/relevance
  filtered the same as user messages, or always retrieved regardless of
  role.
- Promote the config table above into `AppConfig` proper.
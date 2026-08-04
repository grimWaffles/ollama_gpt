# Sage Chat — AI Chat Platform with Streaming & Document/Memory RAG

A full-stack chat application (FastAPI + LangChain backend, Angular frontend) that supports
real-time streaming responses, tool-using agents, and retrieval-augmented generation (RAG)
over both **uploaded documents** and **chat history** (in-conversation + long-term memory),
backed by Postgres + `pgvector`.

---

## Features

- **Streaming & non-streaming chat** — toggle between a classic request/response endpoint
  and a Server-Sent-Events (SSE) streaming endpoint, with a typewriter animation on the
  frontend for streamed messages only.
- **Tool-using agent** (LangChain `create_agent` on top of Ollama-served models), currently
  wired with:
  - Filesystem tools (`_fs_tools`, pre-existing)
  - `search_knowledge_base` — semantic search over the current chat's uploaded documents
  - `search_chat_history` — semantic search over past messages, scoped to the current
    conversation or across the user's entire chat history (long-term memory)
- **Document ingestion (RAG)** — uploaded text/PDF/DOCX files are chunked, embedded, and
  stored in `pgvector`. Small files are still inlined in full in the prompt; large files are
  truncated with a pointer telling the agent to retrieve relevant passages via the tool.
- **Chat history memory (RAG)** — every user/assistant message is likewise chunked and
  embedded, enabling recall of earlier parts of a long conversation or of entirely different
  past conversations.
- **Angular frontend** — chat bubbles as a standalone component, copy/retry actions,
  streaming toggle, and a character-by-character typewriter effect for streamed replies.

---

## Architecture

```
┌─────────────────────────────┐        ┌───────────────────────────────────┐
│           Angular            │  HTTP  │              FastAPI               │
│  chat.service.ts             │◄──────►│  main.py                            │
│  home-component / chat-msg   │  SSE   │   ├── /chat/            (sync)      │
│  (typewriter, streaming ui)  │        │   ├── /chat/stream/     (SSE)       │
└─────────────────────────────┘        │   ├── /documents/{chat_id} (GET)    │
                                        │   └── /documents/{document_id} (DEL)│
                                        └─────────────┬───────────────────────┘
                                                       │
                                        ┌──────────────▼──────────────┐
                                        │          LlmService          │
                                        │  chatWithLlm / chatWithLlmStream
                                        │  _ingest_documents / _ingest_message
                                        │  _build_enriched_conversation │
                                        └────┬───────────────┬─────────┘
                                             │               │
                              ┌──────────────▼───┐   ┌───────▼────────────┐
                              │   OllamaAgent      │   │  EmbeddingService   │
                              │  (LangChain agent, │   │  (chunk + embed via │
                              │   .invoke/.stream)  │   │   nomic-embed-text) │
                              └────────┬───────────┘   └───────┬─────────────┘
                                       │                        │
                     tools: fs_tools, kb_tool, chat_history_tool
                                       │                        │
                              ┌────────▼────────────────────────▼──────┐
                              │        VectorRepository (psycopg2)      │
                              │   documents / document_chunks /         │
                              │   message_chunks   (Postgres + pgvector)│
                              └──────────────────────────────────────────┘
```

---

## Backend

### Core services

| File | Purpose |
|---|---|
| `agents/ollama_agent.py` | Wraps LangChain's `create_agent`; `invoke()` for full responses, `stream()` for incremental SSE-ready output. Fixed a missing-`self` bug in `extract_text`. |
| `services/llm_service.py` | Orchestrates conversation loading, document/message ingestion, agent construction (with tools), and persistence. Houses `chatWithLlm` and `chatWithLlmStream`. |
| `services/embedding_service.py` | Chunks text (`RecursiveCharacterTextSplitter`, 800 chars / 100 overlap) and embeds via Ollama's `nomic-embed-text`. |
| `repositories/vector_repository.py` | All pgvector reads/writes: document + chunk inserts, similarity search over documents and over messages, document listing/deletion. |
| `repositories/conversation_repository.py` | Pre-existing; `create_message` already returns the new row id via `RETURNING id`. |
| `tools/knowledge_base_tool.py` | `search_knowledge_base` — LangChain `BaseTool`, scoped to a `chat_id`, queries `document_chunks`. |
| `tools/chat_history_tool.py` | `search_chat_history` — LangChain `BaseTool`, scoped to `this_conversation` or `all_conversations`, queries `message_chunks`. |

### API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat/` | Standard request/response chat completion. |
| `POST` | `/chat/stream/` | SSE stream of `ChatResponse` chunks (`text/event-stream`). |
| `GET` | `/documents/{chat_id}` | List documents uploaded/ingested for a conversation. |
| `DELETE` | `/documents/{document_id}` | Delete a document and its chunks (cascades). |

### Data model (Postgres / pgvector)

| Table | Purpose |
|---|---|
| `documents` | One row per uploaded file (filename, path, kind, status, whether it was inlined). |
| `document_chunks` | One row per chunk of a document, with a `VECTOR(768)` embedding; HNSW cosine index. |
| `message_chunks` | One row per chunk of a chat message (user/assistant/system/tool), with the same embedding scheme; supports both in-conversation and cross-conversation recall. |

> **Embedding dimension is hardcoded to 768** (`nomic-embed-text` via Ollama). Switching
> embedding models (e.g. to a 1536-dim OpenAI model) requires a schema migration —
> `VECTOR(n)` can't be resized once data is inserted.

### Retrieval flow

1. On upload, `_ingest_documents` stores a `documents` row, chunks + embeds the text, and
   writes rows to `document_chunks`.
2. On every message (user or assistant), `_ingest_message` chunks + embeds the content into
   `message_chunks` (skipped for `tool` role or empty content).
3. `_build_enriched_conversation` inlines small files in full; large files get a short
   notice instead, directing the agent to `search_knowledge_base`.
4. The agent is constructed per-request with three tool groups: filesystem tools, the
   knowledge-base tool (bound to `chat_id`), and the chat-history tool (bound to `chat_id`
   + `user_id`).
5. The system prompt documents both retrieval tools and instructs the agent to cite the
   source filename when using retrieved content.

### Dependencies added

```
pip install pgvector psycopg2-binary langchain-ollama langchain-text-splitters
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Frontend (Angular)

| File | Purpose |
|---|---|
| `chat.service.ts` | `sendMessage()` (non-streaming) and `sendMessageStream()` (fetch + `ReadableStream` SSE consumer). Streamed messages are flagged `animate: true`; the thinking loader turns off on the **first** received chunk rather than only at completion. |
| `chat-message.component.ts` | Standalone chat bubble component. Renders assistant messages via `ngx-markdown`; drives a typewriter effect (12 ms/char) for messages flagged `animate: true`, leaving historical/non-streamed messages rendered instantly. Includes Copy and Retry actions. |
| `home-component.ts` / `.html` | Streaming toggle (`streamingEnabled` signal), branches `submitMessage()` / `retryMessage()` between the sync and streaming service calls; hosts the message loop via `<app-chat-message>`. |

---

## Known open items / not yet done

- **Frontend for documents & memory** — no upload-management UI, document list/delete UI,
  or tool-call rendering for `search_knowledge_base` / `search_chat_history` in chat bubbles
  yet (backend-only for this pass).
- **Typing-speed toggle** — typewriter speed is hardcoded at 12 ms/char; a user-facing speed
  control was discussed but not built.
- **Inline history bound** — `MAX_INLINE_MESSAGES` is sketched but not decided/wired in;
  needed once conversations get long enough that inlining every message stops being cheap.
- **Embedding model migration path** — changing embedding models means changing
  `VECTOR(768)` and re-embedding existing rows; no migration tooling exists yet.

---

## Quick reference: request lifecycle

1. Client calls `/chat/` or `/chat/stream/` with `{ userId, chatId, message }`.
2. `LlmService` creates the conversation if `chatId == 0`, loads prior messages, ingests any
   newly parsed files (`_ingest_documents`), and appends the user's message.
3. An `OllamaAgent` is built with the filesystem, knowledge-base, and chat-history tools, and
   invoked (`invoke()`) or streamed (`stream()`).
4. Every message — user and assistant — is persisted via `create_message` and simultaneously
   chunked/embedded via `_ingest_message` for future recall.
5. The response (or SSE stream of responses) is returned to the client; the Angular frontend
   updates `messages`, optionally typewriter-animating streamed content.

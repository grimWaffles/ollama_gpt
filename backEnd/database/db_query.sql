-- Table: public.conversations

-- DROP TABLE IF EXISTS public.conversations;

CREATE TABLE IF NOT EXISTS public.conversations
(
    chat_id integer NOT NULL GENERATED ALWAYS AS IDENTITY ( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 2147483647 CACHE 1 ),
    user_id integer NOT NULL,
    chat_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT conversations_pkey PRIMARY KEY (chat_id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.conversations
    OWNER to postgres;

-- Table: public.messages

-- DROP TABLE IF EXISTS public.messages;

CREATE TABLE IF NOT EXISTS public.messages
(
    id integer NOT NULL GENERATED ALWAYS AS IDENTITY ( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 2147483647 CACHE 1 ),
    chat_id integer NOT NULL,
    role character varying(50) COLLATE pg_catalog."default" NOT NULL,
    message text COLLATE pg_catalog."default" NOT NULL,
    sequence_no integer NOT NULL,
    created_at timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    attachments jsonb,
    CONSTRAINT messages_pkey PRIMARY KEY (id),
    CONSTRAINT uq_messages_chat_sequence UNIQUE (chat_id, sequence_no),
    CONSTRAINT messages_chat_id_fkey FOREIGN KEY (chat_id)
        REFERENCES public.conversations (chat_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.messages
    OWNER to postgres;


-- Requires pgvector extension (Postgres 18 supports it natively via CREATE EXTENSION)
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded file (parallels what parse_files() already produces)
CREATE TABLE documents (
    id           BIGSERIAL PRIMARY KEY,
    chat_id      BIGINT NOT NULL REFERENCES conversations(chat_id) ON DELETE CASCADE,
    user_id      BIGINT NOT NULL,
    filename     TEXT NOT NULL,
    path         TEXT NOT NULL,          -- matches entry["path"] from parse_files
    content_type TEXT,
    kind         TEXT NOT NULL,          -- text | pdf | docx | image | unsupported | error
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending | embedded | failed | skipped
    char_count   INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedded_at  TIMESTAMPTZ
);

-- One row per chunk of a document, with its embedding
CREATE TABLE document_chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chat_id      BIGINT NOT NULL,   -- denormalized for fast filtering, mirrors documents.chat_id
    user_id      BIGINT NOT NULL,   -- denormalized for fast filtering
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    embedding    VECTOR(768) NOT NULL,  -- 768 = nomic-embed-text via Ollama; change if you pick another model
    token_count  INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX document_chunks_chat_id_idx ON document_chunks (chat_id);
CREATE INDEX document_chunks_user_id_idx ON document_chunks (user_id);
CREATE INDEX documents_chat_id_idx ON documents (chat_id);

ALTER TABLE documents ADD COLUMN is_inlined BOOLEAN NOT NULL DEFAULT false;


CREATE TABLE message_chunks (
    id           BIGSERIAL PRIMARY KEY,
    message_id   BIGINT NOT NULL,   -- FK to messages.id (see note below)
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


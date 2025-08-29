-- ===== Extensions =====
CREATE EXTENSION IF NOT EXISTS vector;

-- ===== Documents (one per Confluence page) =====
CREATE TABLE IF NOT EXISTS documents (
  id            BIGSERIAL PRIMARY KEY,
  space_key     TEXT NOT NULL,
  page_id       TEXT NOT NULL UNIQUE,
  title         TEXT NOT NULL,
  url           TEXT NOT NULL,
  last_modified TIMESTAMPTZ NOT NULL,
  html_path     TEXT NOT NULL,
  sha256        TEXT NOT NULL,
  assets        JSONB NOT NULL DEFAULT '[]'::jsonb
);


-- ===== Chunks (RAG units) =====
CREATE TABLE IF NOT EXISTS document_chunks (
  id            BIGSERIAL PRIMARY KEY,
  doc_id        BIGINT REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index   INT NOT NULL,
  text          TEXT NOT NULL,
  embedding     vector(1536),  -- 1536 for text-embedding-3-*
  metadata      JSONB DEFAULT '{}'::jsonb
);

-- Full-text search (generated column) + index
ALTER TABLE document_chunks
  ADD COLUMN IF NOT EXISTS tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_tsv
  ON document_chunks USING GIN (tsv);

-- Speed up search
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
  ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Text filter on documents
CREATE INDEX IF NOT EXISTS idx_documents_space ON documents(space_key);

-- ===== Chat/session storage for the app =====
CREATE TABLE IF NOT EXISTS chat_sessions (
  id         BIGSERIAL PRIMARY KEY,
  title      TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id         BIGSERIAL PRIMARY KEY,
  session_id BIGINT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role       TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content    TEXT NOT NULL,
  citations  JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_memory (
  id         BIGSERIAL PRIMARY KEY,
  session_id BIGINT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  key        TEXT NOT NULL,
  value      TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_chat_messages_session_created
  ON chat_messages (session_id, created_at);

CREATE INDEX IF NOT EXISTS ix_chat_memory_session_created
  ON chat_memory (session_id, created_at);

-- ===== Users & Auth =====
CREATE TABLE IF NOT EXISTS users (
  id            BIGSERIAL PRIMARY KEY,
  email         TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  is_active     BOOLEAN NOT NULL DEFAULT TRUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_settings (
  user_id    BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  tone       TEXT NOT NULL DEFAULT 'neutral',
  depth      TEXT NOT NULL DEFAULT 'balanced',
  extra_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- tie chat artifacts to users (nullable when AUTH_MODE=none)
ALTER TABLE chat_sessions
  ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE chat_memory
  ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;

-- uploaded user files
CREATE TABLE IF NOT EXISTS user_files (
  id          BIGSERIAL PRIMARY KEY,
  user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind        TEXT NOT NULL CHECK (kind IN ('image','document','other')),
  filename    TEXT NOT NULL,
  path        TEXT NOT NULL,
  size_bytes  BIGINT NOT NULL,
  sha256      TEXT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version integer NOT NULL DEFAULT 1;

-- Library / visibility
ALTER TABLE user_files
  ADD COLUMN IF NOT EXISTS visibility text DEFAULT 'private';  -- 'private' | 'library'

-- Tag documents created from uploads so retrieval can scope by user/session
ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS user_id    bigint,
  ADD COLUMN IF NOT EXISTS session_id bigint,
  ADD COLUMN IF NOT EXISTS visibility text DEFAULT 'private';  -- 'private' | 'library'

CREATE INDEX IF NOT EXISTS idx_docs_user_session ON documents(user_id, session_id);

-- Optional: explicit link table (lets you reuse one file across chats)
CREATE TABLE IF NOT EXISTS chat_files (
  id bigserial PRIMARY KEY,
  session_id bigint REFERENCES chat_sessions(id) ON DELETE CASCADE,
  file_id    bigint REFERENCES user_files(id)    ON DELETE CASCADE,
  created_at timestamptz DEFAULT now(),
  UNIQUE(session_id, file_id)
);


-- helpful indexes
CREATE INDEX IF NOT EXISTS ix_sessions_user     ON chat_sessions(user_id, created_at);
CREATE INDEX IF NOT EXISTS ix_memory_user       ON chat_memory(user_id, created_at);
CREATE INDEX IF NOT EXISTS ix_user_files_user   ON user_files(user_id, created_at);
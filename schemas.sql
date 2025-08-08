CREATE EXTENSION IF NOT EXISTS vector;

-- documents (one per Confluence page)
CREATE TABLE IF NOT EXISTS documents (
  id            BIGSERIAL PRIMARY KEY,
  space_key     TEXT NOT NULL,
  page_id       TEXT NOT NULL UNIQUE,
  title         TEXT NOT NULL,
  url           TEXT NOT NULL,
  last_modified TIMESTAMPTZ NOT NULL,
  html_path     TEXT NOT NULL,
  sha256        TEXT NOT NULL
);

-- chunks (RAG units)
CREATE TABLE IF NOT EXISTS document_chunks (
  id            BIGSERIAL PRIMARY KEY,
  doc_id        BIGINT REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index   INT NOT NULL,
  text          TEXT NOT NULL,
  embedding     vector(1536),  -- 1536 for text-embedding-3-*
  metadata      JSONB DEFAULT '{}'::jsonb
);

-- speed up search
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- text filter
CREATE INDEX IF NOT EXISTS idx_documents_space ON documents(space_key);

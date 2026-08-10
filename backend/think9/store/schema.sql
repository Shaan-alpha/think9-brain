CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id             uuid PRIMARY KEY,
    source_system  text        NOT NULL,
    source_id      text        NOT NULL,
    deep_link      text        NOT NULL,
    title          text        NOT NULL,
    doc_type       text        NOT NULL,
    brand_id       text        NOT NULL,
    function       text        NOT NULL,
    author         text        NOT NULL,
    created_at     timestamptz NOT NULL,
    effective_date date        NOT NULL,
    supersedes_id  uuid,
    acl            text[]      NOT NULL,
    sensitive      boolean     NOT NULL DEFAULT false,
    -- Set by the ingest reconciliation pass. Lets the temporal layer identify a stale
    -- document without needing its successor to be in the same result set.
    is_superseded  boolean     NOT NULL DEFAULT false,
    content_hash   text        NOT NULL,
    ingested_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_system, source_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           uuid PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      int  NOT NULL,
    heading_path text NOT NULL,
    text         text NOT NULL,
    embedding    vector(384) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS documents_acl_idx    ON documents USING gin (acl);

CREATE TABLE IF NOT EXISTS owners (
    id          uuid PRIMARY KEY,
    brand_id    text NOT NULL,
    function    text NOT NULL,
    person_name text NOT NULL,
    contact     text NOT NULL,
    note        text NOT NULL DEFAULT '',
    UNIQUE (brand_id, function)
);

CREATE TABLE IF NOT EXISTS query_log (
    id             uuid PRIMARY KEY,
    asked_at       timestamptz NOT NULL DEFAULT now(),
    user_id        text        NOT NULL,
    question       text        NOT NULL,
    route          text        NOT NULL,
    coverage_score double precision NOT NULL,
    outcome        text        NOT NULL,
    answer_text    text        NOT NULL DEFAULT '',
    citations      jsonb       NOT NULL DEFAULT '[]'::jsonb,
    as_of          date,
    trace          jsonb       NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS canon (
    id              uuid PRIMARY KEY,
    question        text NOT NULL,
    answer          text NOT NULL,
    author          text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    source_query_id uuid REFERENCES query_log(id),
    effective_date  date NOT NULL
);

-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- 1. Documents Table
create table if not exists documents (
  id uuid primary key default uuid_generate_v4(),
  user_id text not null, -- Links to auth.users or custom user string
  filename text not null,
  storage_path text not null, -- Path in Supabase Storage
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  processed boolean default false
);

-- 2. Chunks Table
create table if not exists chunks (
  id uuid primary key default uuid_generate_v4(),
  document_id uuid references documents(id) on delete cascade not null,
  chunk_index integer not null,
  text_content text not null,
  summary text, -- Short summary of the chunk
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 3. Topics Table
create table if not exists topics (
  id uuid primary key default uuid_generate_v4(),
  name text unique not null
);

-- 4. Chunk Topics Junction Table (Many-to-Many)
create table if not exists chunk_topics (
  chunk_id uuid references chunks(id) on delete cascade not null,
  topic_id uuid references topics(id) on delete cascade not null,
  primary key (chunk_id, topic_id)
);

-- 5. FAISS Map Table (Maps generic logic integer ID to Chunk UUID)
create table if not exists faiss_map (
  faiss_index bigint primary key, -- The integer ID used in FAISS
  chunk_id uuid references chunks(id) on delete cascade not null,
  unique (chunk_id) -- Ensure 1:1 mapping for simplicity
);

-- Indexes for performance
create index if not exists idx_documents_user_id on documents(user_id);
create index if not exists idx_chunks_document_id on chunks(document_id);
create index if not exists idx_chunk_topics_topic_id on chunk_topics(topic_id);

-- 6. Knowledge Graph Nodes
create table if not exists knowledge_graph_nodes (
  id uuid primary key default uuid_generate_v4(),
  entity_name text not null,
  document_id uuid references documents(id) on delete set null,
  chunk_ids uuid[] default '{}'::uuid[],
  unique (entity_name)
);

-- 7. Knowledge Graph Edges
create table if not exists knowledge_graph_edges (
  id uuid primary key default uuid_generate_v4(),
  source_node_id uuid references knowledge_graph_nodes(id) on delete cascade not null,
  target_node_id uuid references knowledge_graph_nodes(id) on delete cascade not null,
  relation_label text,
  weight float default 1.0,
  unique (source_node_id, target_node_id, relation_label)
);

-- Note: The test_questions table should be modified if it already exists, or updated in its schema.
-- Assuming an existing test_questions table, we alter it here:
-- ALTER TABLE test_questions ADD COLUMN IF NOT EXISTS bert_f1 float;
-- ALTER TABLE test_questions ADD COLUMN IF NOT EXISTS rouge1 float;
-- ALTER TABLE test_questions ADD COLUMN IF NOT EXISTS rouge2 float;
-- ALTER TABLE test_questions ADD COLUMN IF NOT EXISTS rougeL float;

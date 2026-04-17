# StudyHub AI — Complete Developer Documentation

> **Purpose:** This document is a complete technical manual for the StudyHub AI project. It details the architecture, file structure, module responsibilities, database schema, and system workflows.

---

## 1. Project Overview

**StudyHub AI** is an agentic AI-powered educational platform.
- **Teachers** can upload PDFs, create classrooms, assign quizzes, and review/grade them.
- **Students** can generate practice questions and take formal tests on uploaded materials.
- **Admins** manage users, classrooms, and resources.

It uses a **Retrieval-Augmented Generation (RAG)** pipeline backed by two multi-agent workflows (System 1 for ingestion, System 2 for query).

---

## 2. Technology Stack

- **Web Framework**: Flask 3.0.0
- **Database**: Supabase (PostgreSQL)
- **Vector Search**: FAISS (HNSW index, 768-dim)
- **Embedding Model**: `all-mpnet-base-v2`
- **LLM**: Groq API (`llama-3.1-8b-instant`)
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (for RAG selection)
- **PDF Processing**: PyMuPDF (`fitz`), LangChain (`RecursiveCharacterTextSplitter`)
- **UI**: Jinja2 + Vanilla JS Single Page App

---

## 3. Directory & File Structure

```
SOP/
├── app.py                      # Main entrypoint, contains all Flask routes & OAuth.
├── requirements.txt            # Python dependencies.
├── .env                        # Environment variables.
├── supabase_schema.sql         # Supabase Database schema definitions.
├── index.faiss                 # The saved FAISS HNSW vector index binary.
├── ui & static/
│   ├── templates/index.html    # Main Single Page App UI.
│   ├── templates/login.html    # Login page.
│   └── templates/registration.html # Profile building page.
│   └── static/                 # Images & Avatars.
├── agents/                     # Agentic AI logic
│   ├── system1_ingestion.py    # Pipeline to parse PDF, chunk, embed, & index.
│   └── system2_query.py        # Pipeline for RAG retrieval, ranking, & generation.
├── services/                   # Core business logic
│   ├── client.py               # Supabase Client singleton.
│   ├── db.py                   # DB ops (CRUD for docs, chunks, users, classrooms).
│   ├── storage.py              # Supabase Storage Wrapper for PDF uploads.
│   ├── vector.py               # FAISS interactions (creating/querying the index).
│   ├── llm.py                  # LLM integrations (calls to Groq API).
│   ├── selector.py             # CrossEncoder for re-ranking retrieved chunks.
│   ├── generator_agent.py      # Draft-Critique-Refine generator logic.
│   ├── eval_metrics.py         # Advanced RAG metrics (ExactMatch, Faithfulness).
│   └── ingestion.py            # Legacy ingestion pipeline (use system1 instead).
├── utils/
│   └── text_processing.py      # PDF text extraction and chunking wrappers.
├── scripts/                    # Maintenance & Benchmarking
│   └── (benchmark & migration scripts)
└── tests/                      # Synthetic data generators & integration tests
```

---

## 4. Module-by-Module Breakdown

### 4.1. `app.py`
The web server's core. Contains:
- Google OAuth setup.
- General user auth (`/login`, `/auth/google`, `/register`).
- Uploading docs (`/upload`).
- Generating quizzes (`/generate-questions`, `/generate-test`).
- Assignments & Testing (`/start-assignment`, `/submit-answer`, `/finish-test`).
- Social/Classrooms (`/create-group`, `/create-class`, `/join-class`).
- Admin panel operations.

### 4.2. `agents/system1_ingestion.py`
Data ingestion pipeline. Consists of 4 Agents orchestrated by `System1Orchestrator`:
1. **ParsingAgent**: `fitz` extracts raw text from PDF.
2. **ChunkingAgent**: Splits text into 1000-char chunks using `TextProcessor`.
3. **VerificationAgent**: Uses LLM to extract summary and topics for each chunk.
4. **EmbeddingAgent**: Maps chunks in DB, calculates 768-dim embeddings, and adds them to the FAISS `VectorService`.

### 4.3. `agents/system2_query.py`
Retrieval & question generation pipeline:
1. **RetrieverAgent**: Queries FAISS for top 30 chunks based on topics/keywords.
2. **SelectorAgentProxy**: Trims irrelevant chunks using a Cross-Encoder (top 50% kept).
3. **ResponseAgent**: Prompts the LLM with the context to generate the quiz JSON.
4. **EvaluatorAgent**: Validates formatting and tests for hallucinations/faithfulness. If failed, feeds back into `ResponseAgent`.

### 4.4. `services/db.py`
Static helper methods wrapping `supabase` read/writes. Handles tables: `documents`, `chunks`, `topics`, `chunk_topics`, `faiss_map`, `users`, `classrooms`, `tests`.

### 4.5. `services/vector.py`
Handles `faiss.IndexHNSWFlat`. Exposes `add_texts` and `search`. Maps FAISS integer indexes back to chunks via `DBService.add_faiss_map`. 

### 4.6. `services/llm.py`
Interacts with the internal `groq` wrapper:
- `extract_metadata`: Pulls summaries and keywords.
- `evaluate_answer`: Sophisticated grading logic testing logic, final answer correctness, etc.
- `evaluate_batch`: Wraps evaluations in a concurrent ThreadPoolExecutor.

### 4.7. `services/eval_metrics.py`
Formal RAG evaluation metric formulations to test AI responses against context (Exact Match, Faithfulness, Relevance).

---

## 5. Database Schema Overview (Supabase Postgres)

| Table | Description |
|---|---|
| `users` | General profiles (Student, Teacher, Admin), bio, role. |
| `documents` | Uploaded PDFs (links to Google Cloud/Supabase store). |
| `chunks` | Text fragments (from the PDFs) & their generated summaries. |
| `topics` | Extracted core subject topics. linked through `chunk_topics`. |
| `faiss_map` | Ties FAISS integer indexes to PostgreSQL chunk UUIDs. |
| `tests` & `test_questions` | Student's test attempts, individual answers, AI graded score. |
| `classrooms` & `class_assignments` | Virtual classrooms and assigned tests. |

---

## 6. End-to-End Workflow Summaries

**1. Getting a New Textbook Ready:**
A user uploads an intro to biology PDF. `app.py` passes the stream to `system1.process_document`. The text is extracted, chunked, and parsed. Topics are attached via LLM. It's stored in Supabase & FAISS.

**2. Student Taking a Practice Quiz:**
A student queries for "Biology MCQ". `app.py` calls `system2.generate_quiz`. The retriever hits FAISS. The selector condenses it. The response agent drafts a test. The evaluator ensures accuracy. The questions are parsed and displayed on screen in JSON.

**3. Test Review & AI Re-Eval:**
Once a test is finished, `llm_service.evaluate_batch` invokes LLM threading to grade all test_questions rapidly. If a teacher objects to AI grading, they use `/teacher/submit-review`. Subsequent re-evaluations will inject the `[Teacher's Manual Review/Feedback]` directly into the LLM context.

---

## 7. Local Setup Instructions

1. Ensure the Python version is roughly `3.10+`. 
2. `pip install -r requirements.txt`.
3. Fill missing secrets inside `.env`.
4. Define your Supabase DB schemas via `supabase_schema.sql`.
5. Start app using `run_app.bat` or `python app.py`.

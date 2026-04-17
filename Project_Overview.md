# Project Overview: StudyHub AI

StudyHub is an AI-powered educational platform designed to transform static textbook content (PDFs) into interactive learning experiences. It leverages Large Language Models (LLMs) and Vector Databases to provide automated topic extraction, practice question generation, and collaborative study environments.

## Core Technology Stack

*   **Backend**: Flask (Python)
*   **AI / LLM**: Groq (Llama-3.3-70b-versatile) via LangChain
*   **Vector Database**: ChromaDB (Local persistence for document indexing)
*   **Database & Auth**: Supabase (PostgreSQL for user data, Google OAuth for authentication)
*   **PDF Processing**:
    *   **PyMuPDF (fitz)**: For text extraction, Table of Contents (TOC) analysis, and font styling detection.
    *   **RapidOCR**: For processing scanned or image-heavy PDF pages.
*   **Frontend**: Vanilla HTML/JS with a responsive dashboard design.
*   **Document Generation**: ReportLab (for generating downloadable PDF tests).

## Current Implementation Status

### 1. Document Processing & Indexing
*   **Smart Topic Extraction**: Implements a three-tier strategy:
    *   **Level 1**: Native PDF Table of Contents (TOC).
    *   **Level 2**: Advanced font-style analysis (detecting bold/large headings, specifically tuned for NCERT-style textbooks).
    *   **Level 3**: LLM-based extraction from initial pages for documents without clear structure.
*   **Vector Indexing**: Documents are chunked using `RecursiveCharacterTextSplitter` and stored in ChromaDB with topic-level metadata, allowing for precise retrieval.

### 2. AI Features
*   **Practice Generation**: Users can select specific topics to generate MCQs, Conceptual questions, Analytical cases, and Fill-in-the-blanks.
*   **Test Mode**: Full-screen timed tests with automated scoring and difficulty selection.
*   **Document Chat**: Context-aware AI chat that answers questions based solely on the uploaded document's content.

### 3. User Management & Social
*   **Role-Based Access**: Specialized dashboards for Students, Teachers, and Admins.
*   **Community Features**:
    *   **Study Groups**: Password-protected groups for collective testing.
    *   **Classrooms**: Teacher-led environments for assigning tests to students.
*   **Teacher Verification**: A workflow for manual admin verification of teacher accounts.

## Future Roadmap & Needs Implementation

Based on current project needs and requirements:

*   **[High Priority] External Vector Storage**: Migration from local ChromaDB to a scalable, online vector database (e.g., **FAISS** hosted or a managed service like Pinecone/Supabase Vector) to support "Facebook online vector db" requirements.
*   **Automated Topic Discovery**: Enhancing topic detection by using vector clustering to "find topics" within the document chunks rather than relying primarily on font/TOC analysis.
*   **Advanced AI Prompting**: Refining how chunks are "given to AI" to support more complex tasks like multi-document summarization or cross-topic relationship mapping.
*   **Real-time Collaboration**: WebSocket integration for live study sessions within groups.
*   **Comprehensive Assessment Tracking**: More detailed analytics for students to track their performance over time per topic.

import logging
import fitz
from services.db import DBService
from services.storage import StorageService
from services.vector import VectorService
from services.llm import LLMService
from utils.text_processing import TextProcessor
from config import config
from services.knowledge_graph import knowledge_graph_service

logger = logging.getLogger(__name__)

class ParsingAgent:
    """Agent 1: Converts PDF -> text and extracts content."""
    def parse_pdf(self, file_bytes: bytes) -> str:
        logger.info("ParsingAgent: Extracting text from PDF...")
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

class ChunkingAgent:
    """Agent 2: Breaks text into smaller chunks with overlap."""
    def chunk_text(self, text: str) -> list:
        logger.info("ChunkingAgent: Chunking text for LLM processing...")
        return TextProcessor.chunk_text(text)

class VerificationAgent:
    """Agent 4: Checks correctness, metadata, and quality. (Executed alongside chunking)"""
    def __init__(self):
        self.llm_service = LLMService()

    def verify_and_annotate(self, chunks: list, doc_id: str) -> list:
        logger.info(f"VerificationAgent: Annotating {len(chunks)} chunks with metadata...")
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            meta = self.llm_service.extract_metadata(chunk_text)
            chunk_objects.append({
                "document_id": doc_id,
                "chunk_index": i,
                "text_content": chunk_text,
                "summary": meta.get("summary"),
                "topics": meta.get("topics", [])
            })
        return chunk_objects

class EmbeddingAgent:
    """Agent 3: Converts chunks -> vectors and stores them in HNSW graph."""
    def __init__(self):
        self.vector_service = VectorService()

    def embed_and_index(self, chunk_objects: list):
        logger.info("EmbeddingAgent: DB Insertion and FAISS Indexing...")
        
        # Insert Chunks to Postgres
        chunks_to_insert = [{
            "document_id": c["document_id"],
            "chunk_index": c["chunk_index"],
            "text_content": c["text_content"],
            "summary": c["summary"]
        } for c in chunk_objects]
        
        inserted_chunks = DBService.add_chunks(chunks_to_insert)
        
        chunk_topic_links = []
        texts_for_vector = []
        chunk_ids_for_vector = []
        
        for i, row in enumerate(inserted_chunks):
            chunk_id = row['id']
            original = chunk_objects[i]
            
            if config.ENABLE_KNOWLEDGE_GRAPH:
                knowledge_graph_service.extract_and_store(
                    text_chunk=original['text_content'], 
                    chunk_id=chunk_id, 
                    document_id=original['document_id']
                )
            
            if original['topics']:
                topic_map = DBService.add_topics(original['topics'])
                for t_name in original['topics']:
                    if t_name in topic_map:
                        chunk_topic_links.append({"chunk_id": chunk_id, "topic_id": topic_map[t_name]})
            
            texts_for_vector.append(original['text_content'])
            chunk_ids_for_vector.append(chunk_id)

        DBService.link_chunk_topics(chunk_topic_links)
        
        # FAISS HNSW Embedding
        self.vector_service.add_texts(texts_for_vector, chunk_ids_for_vector)

class System1Orchestrator:
    """Orchestrates the System 1 Data Pipeline from raw PDF to Vector DB."""
    def __init__(self):
        self.parsing_agent = ParsingAgent()
        self.chunking_agent = ChunkingAgent()
        self.verification_agent = VerificationAgent()
        self.embedding_agent = EmbeddingAgent()

    def process_document(self, file_stream, filename: str, user_id: str, progress_callback=None):
        try:
            steps = 5
            curr = 0
            
            def update(msg):
                nonlocal curr; curr += 1
                if progress_callback: progress_callback(curr, steps, msg)
                
            update("Uploading Storage...")
            file_stream.seek(0)
            storage_path = f"{user_id}/{filename}"
            StorageService.upload_file(file_stream, storage_path)
            
            doc_id = DBService.create_document(user_id, filename, storage_path)
            
            # [Parsing Agent]
            update("ParsingAgent: Reading PDF...")
            file_stream.seek(0)
            text = self.parsing_agent.parse_pdf(file_stream.read())
            
            # [Chunking Agent]
            update("ChunkingAgent: Splitting Text...")
            chunks = self.chunking_agent.chunk_text(text)
            
            # [Verification Agent]
            update("VerificationAgent: Checking & Annotating...")
            annotated_chunks = self.verification_agent.verify_and_annotate(chunks, doc_id)
            
            # [Embedding Agent]
            update("EmbeddingAgent: Building Vector Graph...")
            self.embedding_agent.embed_and_index(annotated_chunks)
            
            DBService.update_document_processed(doc_id, True)
            update("System 1 Complete!")
            return doc_id
            
        except Exception as e:
            logger.error(f"System 1 Pipeline failed: {e}")
            raise

system1 = System1Orchestrator()

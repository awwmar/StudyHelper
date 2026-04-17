import logging
import io
import fitz
from services.db import DBService
from services.storage import StorageService
from services.vector import VectorService
from services.llm import LLMService
from utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.vector_service = VectorService()
        self.llm_service = LLMService()

    def process_document(self, file_stream, filename: str, user_id: str, progress_callback=None):
        """
        Orchestrates the full ingestion pipeline.
        progress_callback: function(current_step, total_steps, message)
        """
        try:
            steps = 5
            curr = 0
            
            def update(msg):
                nonlocal curr
                curr += 1
                if progress_callback:
                    progress_callback(curr, steps, msg)

            # 1. Upload to Storage
            update("Uploading to Storage...")
            file_stream.seek(0)
            storage_path = f"{user_id}/{filename}"
            StorageService.upload_file(file_stream, storage_path)

            # 2. Create Document Record
            doc_id = DBService.create_document(user_id, filename, storage_path)

            # 3. Extract Text
            update("Extracting Text...")
            file_stream.seek(0) # Reset stream
            # Note: extract_text_from_pdf expects bytes or filename. 
            # TextProcessor.extract_text_from_pdf uses fitz.open(stream=..., filetype="pdf")
            # We need to read bytes here.
            file_bytes = file_stream.read() 
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            
            # 4. Chunk Text
            update("Chunking Text...")
            chunks_text = TextProcessor.chunk_text(text)
            
            # 5. Process Chunks
            update(f"Analyzing {len(chunks_text)} chunks (Embedding & Topic Extraction)...")
            
            chunk_objects = []
            chunk_ids_for_vector = []
            texts_for_vector = []
            chunk_topic_links = []

            for i, chunk_text in enumerate(chunks_text):
                # LLM Extraction
                # We can skip LLM for every chunk to save time/cost if needed, 
                # but requirements say "Ai extracts 3-5 topics per chunk".
                meta = self.llm_service.extract_metadata(chunk_text)
                
                chunk_obj = {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "text_content": chunk_text,
                    "summary": meta.get("summary"),
                    "topics": meta.get("topics", [])
                }
                chunk_objects.append(chunk_obj)

            # 6. Bulk Insert Chunks
            chunks_to_insert = [{
                "document_id": c["document_id"],
                "chunk_index": c["chunk_index"],
                "text_content": c["text_content"],
                "summary": c["summary"]
            } for c in chunk_objects]
            
            inserted_chunks = DBService.add_chunks(chunks_to_insert) # Returns list of inserted rows with IDs
            
            # Map back IDs
            # Assuming sequential order
            for i, row in enumerate(inserted_chunks):
                chunk_id = row['id']
                original = chunk_objects[i]
                
                # Topics
                if original['topics']:
                    topic_map = DBService.add_topics(original['topics'])
                    for t_name in original['topics']:
                        if t_name in topic_map:
                            chunk_topic_links.append({
                                "chunk_id": chunk_id,
                                "topic_id": topic_map[t_name]
                            })
                
                texts_for_vector.append(original['text_content'])
                chunk_ids_for_vector.append(chunk_id)

            # 7. Link Topics
            DBService.link_chunk_topics(chunk_topic_links)
            
            # 8. Vector Indexing
            update("Indexing Vectors...")
            self.vector_service.add_texts(texts_for_vector, chunk_ids_for_vector)

            # 9. Finalize
            DBService.update_document_processed(doc_id, True)
            update("Complete!")
            
            return doc_id

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise


# Singleton
ingestion_service = IngestionService()

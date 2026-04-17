import logging
from services.client import supabase

logger = logging.getLogger(__name__)

class StorageService:
    BUCKET_NAME = "pdfs"

    @staticmethod
    def upload_file(file_stream, destination_path: str, content_type: str = "application/pdf") -> str:
        """
        Uploads a file to Supabase Storage.
        Returns the path if successful.
        """
        try:
            # Check if bucket exists, if not create 'pdfs'? 
            # Supabase API usually requires manual bucket creation or via SQL.
            # We assume 'pdfs' bucket exists or we handle error.
            
            # Read file stream
            file_bytes = file_stream.read()
            
            # Upload
            supabase.storage.from_(StorageService.BUCKET_NAME).upload(
                path=destination_path,
                file=file_bytes,
                file_options={"content-type": content_type, "x-upsert": "true"}
            )
            return destination_path
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise

    @staticmethod
    def get_public_url(path: str) -> str:
        try:
            return supabase.storage.from_(StorageService.BUCKET_NAME).get_public_url(path)
        except Exception as e:
            logger.error(f"Error getting public url: {e}")
            raise
    
    @staticmethod
    def download_file(path: str) -> bytes:
        try:
            return supabase.storage.from_(StorageService.BUCKET_NAME).download(path)
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise

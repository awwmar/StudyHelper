import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextProcessor:
    @staticmethod
    def extract_text_from_pdf(file_stream) -> str:
        """
        Extracts text from a PDF file stream.
        """
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
        """
        Splits text into chunks using RecursiveCharacterTextSplitter.
        Returns list of strings.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return splitter.split_text(text)

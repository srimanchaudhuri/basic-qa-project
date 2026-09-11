
from pathlib import Path
import tempfile
from typing import BinaryIO

from app.config import get_settings
from app.utils.logger import get_logger


logger = get_logger(__name__)

class DocumentProcessor:

    def __init__(self, file: BinaryIO, file_name: str, file_ext: str):
        from docling_core.transforms.chunker import HybridChunker
        from langchain_docling import DoclingLoader
        from langchain_docling.loader import ExportType

        self.settings = get_settings()
        self.file_name = file_name

        if not self.check_extension(file_ext):
            logger.error("File type not supported. Supported file types: [.pdf, .csv, .txt, .docx]")
            raise ValueError(f"File type not supported: '{file_ext}'. Supported: [.pdf, .csv, .txt, .docx]")

        self.file_path = self.save_temp_file(file_ext=file_ext, file=file)

        chunker = HybridChunker(
            model=self.settings.embedding_model,
            merge_peers=True,
            max_tokens = self.settings.max_tokens
        )
        

        self.doc_loader = DoclingLoader(
            export_type=ExportType.DOC_CHUNKS,
            file_path=self.file_path,
            chunker=chunker
        )

    def load_document(self):
        logger.info(f"Loading File: {self.file_name}")

        try:
            documents = self.doc_loader.load()
            logger.info(f"Loaded Document")
            for doc in documents:
                doc.metadata["source"] = self.file_name

        finally:
            Path(self.file_path).unlink(missing_ok=True)

        return documents

    def check_extension(self, file_ext) -> bool:
        if file_ext not in ['.pdf', '.txt', '.csv', '.docx']:
            return False
        return True

    def save_temp_file(self, file_ext:str, file: BinaryIO) -> str:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as tmp_file:
            tmp_file.write(file.read())
            return tmp_file.name
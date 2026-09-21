from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from web_crawler import CrawlResult

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WebContentProcessor:
    """Converts crawled web pages into chunked LangChain Documents."""

    def __init__(self):
        settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.max_tokens * 4,
            chunk_overlap=200,
        )

    def process(self, results: list[CrawlResult]) -> list[Document]:
        """Filter successful pages, split text, and return Documents with metadata."""
        documents: list[Document] = []

        for result in results:
            if result.status != "processed" or not result.content:
                continue

            chunks = self.text_splitter.split_text(result.content)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": result.url,
                        "source_type": "web",
                        "status_code": result.status_code,
                        "fetched_at": result.fetched_at,
                    },
                )
                documents.append(doc)

        logger.info(
            f"Processed {len(results)} crawled pages into {len(documents)} chunks"
        )
        return documents

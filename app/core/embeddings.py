from app.config import get_settings
from app.utils.logger import get_logger

from functools import lru_cache


logger = get_logger(__name__)

@lru_cache
def get_embeddings():
    from langchain_ollama import OllamaEmbeddings

    settings = get_settings()

    embeddings = OllamaEmbeddings(
        model=settings.embedding_model
    )

    logger.info(f"Embedding model: {settings.embedding_model} loaded")

    return embeddings

class EmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.embeddings = get_embeddings()
        self.model_name = settings.embedding_model

    def embed_query(self, query: str) -> list[float]:

        logger.debug(f"Generating embeded query: {query[:50]}...")
        return self.embeddings.embed_query(query)

    def embed_docs(self, docs: list[str]) -> list[list[float]]:

        logger.debug(f"Generating embeded query: {docs[:50]}")
        return self.embeddings.embed_documents(docs)
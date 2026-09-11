
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

from functools import lru_cache

logger = get_logger(__name__)


@lru_cache
def get_qdrant_client():
    from qdrant_client import QdrantClient

    settings = get_settings()

    client = QdrantClient(
        api_key=settings.qdrant_api_key or None,
        url=settings.qdrant_url,
        prefer_grpc=False,
    )

    logger.info(f"Qdrant client connected.")

    return client

class VectorStoreService:

    def __init__(self, collection_name: str | None = None):
        from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
        from app.core.embeddings import get_embeddings

        settings = get_settings()
        self.qdrant_client = get_qdrant_client()
        self.collection_name = collection_name if collection_name is not None else settings.collection_name
        self.ensure_collection()
        self.embeddings = get_embeddings()

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=FastEmbedSparse(),
            validate_collection_config=False
        )

        logger.info(f"Vector Store initialised with collection name -> {self.collection_name}")

    def ensure_collection(self):
        from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

        settings = get_settings()

        try:
            collection_info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            logger.info(
                f"Collection {self.collection_name} exists with "
                f"{collection_info.points_count} points"
            )
        except Exception:
            logger.info(f"Creating collection with collection name: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE
                ),
                sparse_vectors_config={
                    "langchain-sparse": SparseVectorParams()
                },
            )
            logger.info(f"Collection created successfully")

    def add_docs(self, docs: list[Document]) -> list[str]:

        if not docs:
            logger.warning("No documents to add")
            return []

        logger.info(f"Adding documents to vector store: {self.collection_name}")
        ids = [str(uuid4()) for _ in docs]

        self.vector_store.add_documents(
            documents=docs,
            ids=ids
        )

        logger.info(f"Documents added successfully")
        return ids

    def search(self, query: str, k: int | None = None) -> list[Document]:

        settings = get_settings()
        k = k if k is not None else settings.retrieval_k

        logger.debug(f"Searching for: {query[:50]}... (k={k})")
        results =  self.vector_store.similarity_search(
            query=query,
            k=k
        )

        logger.debug(f"Found {len(results)} results")
        return results

    def search_with_score(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
    
            settings = get_settings()
            k = k if k is not None else settings.retrieval_k
    
            logger.debug(f"Searching for: {query[:50]}... (k={k})")
            results =  self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
    
            logger.debug(f"Found {len(results)} results")
            return results

    def get_retriever(self, k: int | None = None) -> VectorStoreRetriever:

        settings = get_settings()
        k = k if k is not None else settings.retrieval_k

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":k}
        )

    def delete_collection(self) -> None:
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.qdrant_client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Collection: {self.collection_name} deleted.")

    def get_collection_info(self) -> dict:
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "indexed_vector_count": info.indexed_vectors_count,
                "status": info.status.value
            }
        except Exception:
            return {
                "name": self.collection_name,
                "points_count": 0,
                "indexed_vector_count": 0,
                "status": 'red'
            }

    def health_check(self) -> bool:
            try:
                self.qdrant_client.get_collections()
                return True
            except Exception as e:
                logger.error(f"Vector store health check failed: {e}")
                return False
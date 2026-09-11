from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf8",
        case_sensitive=False,
        extra="ignore"
    )
    api_host: str = "127.0.0.1"
    api_port: int = 3000
    log_level: str = "INFO"
    app_name: str = "basic_rag_qa"
    embedding_model: str = "nomic-embed-text:latest"
    embedding_dimension: int = 768
    max_tokens: int = 512
    qdrant_url: str
    qdrant_api_key: str | None = None
    collection_name: str = "rag_documents"
    retrieval_k: int = 3
    llm_model: str = "llama3.2"
    temperature: float = 0
    ragas_llm_model: str = "llama3.1"
    ragas_llm_temperature: float = 0
    eval_embedding_model: str = "nomic-embed-text:latest"
    ragas_log_result: bool = False

@lru_cache
def get_settings() -> Settings:
    return Settings()
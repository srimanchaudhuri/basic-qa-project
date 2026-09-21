import datetime
from typing import Any

from pydantic import BaseModel, Field

class HealthResponse(BaseModel):

    status: str = Field(..., description="Service Status")
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Response timestamp"
    )
    version: str= Field(..., description="Application version")

class ReadinessResponse(BaseModel):

    status: str = Field(..., description="Service Status")
    qdrant_connected: bool = Field(..., description="Qdrant connection status")
    collection_info: dict = Field(..., description="Collection information")

class DocumentUploadResponse(BaseModel):

    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    document_ids: list[str] = Field(..., description="List of document IDs")
    chunks_created: int = Field(..., description="Number of chunks created")

class DocumentInfo(BaseModel):

    source: str = Field(..., description="Document source/filename")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata of Document")

class DocumentListResponse(BaseModel):

    collection_name: str = Field(..., description="Name of the collection")
    total_documents: int = Field(..., description="Total document count")
    status: str = Field(..., description="Collection status")

class QueryRequest(BaseModel):

    question: str = Field(
        ...,
        description="Question to ask",
        min_length=1,
        max_length=1000
    )
    include_sources: bool = Field(
        default=False,
        description="Add sources with response"
    )
    enable_evaluation: bool = Field(
        default=False,
        description="Enable RAGAs evaluation"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "question": "What is RAG?",
                "include_sources": True,
                "enable_evaluation": False
            }]
        }
    }

class SourceDocument(BaseModel):
    content: str = Field(..., description="Document content excerpt")
    metadata: dict[str, Any] = Field(..., description="Document metadata")

class EvaluationScores(BaseModel):
    faithfulness: float | None = Field(
        None, 
        description="faithfulness score",
        ge=0.0,
        le=1.0
    )
    answer_relevancy: float | None = Field(
        None, 
        description="answer relevancy",
        ge=0.0,
        le=1.0
    )
    evaluation_time_ms: float| None = Field(
        None,
        description="Time taken for evaluation"
    )
    error: str | None = Field(
        ...,
        description="error if evaluation failed"
    )

class QueryResponse(BaseModel):
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated Answer")
    sources: list[SourceDocument] | None = Field(None, description="Source documents used")
    processing_time_ms: float = Field(..., description="Query processing time in ms")
    evaluation: EvaluationScores | None = Field(None, description="RAGAS Evaluation scores")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Error details")

class ValidationErrorResponse(BaseModel):
    error: str = Field(default="Validation Error", description="Error type")
    message: str = Field(..., description="Error message")
    errors: list[dict] = Field(..., description="Validation errors")

class CrawlRequest(BaseModel):
    url: str = Field(
        ...,
        description="URL to crawl",
        min_length=1,
    )
    page_limit: int = Field(
        default=50,
        gt=0,
        le=10000,
        description="Maximum number of pages to crawl"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "url": "https://example.com",
                "page_limit": 50
            }]
        }
    }

class CrawlResponse(BaseModel):
    message: str = Field(..., description="Status message")
    url: str = Field(..., description="Crawled URL")
    pages_crawled: int = Field(..., description="Number of pages successfully crawled")
    chunks_created: int = Field(..., description="Number of chunks stored in vector store")
    document_ids: list[str] = Field(..., description="List of document IDs stored")

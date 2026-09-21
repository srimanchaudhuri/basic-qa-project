import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger
from app.api.schemas import CrawlRequest, CrawlResponse, DocumentListResponse, DocumentUploadResponse, ErrorResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses= {
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Upload and ingest a document",
    description="Upload a document (PDF, CSV or TXT) to be processed and added to the vectorstore"
)
async def upload_document(file: UploadFile = File(..., description="Document file to upload")) -> DocumentUploadResponse:

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    try:
        file_path = Path(file.filename)
        file_extension = file_path.suffix.lower()
        processor = DocumentProcessor(file=file.file, file_name=file.filename, file_ext=file_extension)
        logger.info(f"Received document upload request for {file.filename}")

        chunks = processor.load_document()
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the given document"
            )

        vector_store = VectorStoreService()
        document_ids = vector_store.add_docs(chunks)

        logger.info(
            f"Successfully processed file: {file.filename}"
            f"{len(chunks)} chunks, {len(document_ids)} documents"
        )

        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            document_ids=document_ids,
            chunks_created=len(chunks)
        )

    except ValueError as e:
        logger.error(f"Invalid file upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get(
    "/info",
    response_model=DocumentListResponse,
    summary="Get collection information",
    description="Get information about the document collection"
)
async def get_collection_info() -> DocumentListResponse:
    try:
        vector_store = VectorStoreService()
        info = vector_store.get_collection_info()

        return DocumentListResponse(
            collection_name=info["name"],
            total_documents=info["points_count"],
            status=info["status"]
        )
    except Exception as e:
        logger.error(f"Error getting collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting collection info: {str(e)}"
        )

@router.delete(
    "/collection",
    responses={
        200: {"message": "Collection deleted successfully"},
        500: {"model":ErrorResponse, "description": "Error deleting collection"}
    },
    summary="Delete entire collection",
    description="Delete all documents from the vector store"
)
async def delete_collection() -> dict:
    logger.warning("Deleting vector collection")

    try:
        vector_store = VectorStoreService()
        vector_store.delete_collection()

        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.error(f"Error while deleting collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection: {str(e)}"
        )

@router.post(
    "/crawl",
    response_model=CrawlResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid URL"},
        500: {"model": ErrorResponse, "description": "Crawl error"}
    },
    summary="Crawl a website and ingest its content",
    description="Crawl a website starting from the given URL, extract text content, and store it in the vector store for querying"
)
async def crawl_website(request: CrawlRequest) -> CrawlResponse:
    try:
        from web_crawler import Crawler, CrawlerConfig
        from app.core.web_processor import WebContentProcessor

        logger.info(f"Starting crawl of {request.url} with page_limit={request.page_limit}")

        db_path = Path(tempfile.mkdtemp()) / "crawl.db"
        config = CrawlerConfig(
            url=request.url,
            page_limit=request.page_limit,
            db_path=db_path,
        )

        crawler = Crawler(config)
        results = await crawler.crawl()

        pages_crawled = sum(1 for r in results if r.status == "processed")
        if pages_crawled == 0:
            raise HTTPException(
                status_code=400,
                detail="No content could be crawled from the given URL"
            )

        processor = WebContentProcessor()
        chunks = processor.process(results)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from crawled pages"
            )

        vector_store = VectorStoreService()
        document_ids = vector_store.add_docs(chunks)

        logger.info(
            f"Crawl complete: {request.url} — "
            f"{pages_crawled} pages, {len(chunks)} chunks stored"
        )

        return CrawlResponse(
            message="Website crawled and content stored successfully",
            url=request.url,
            pages_crawled=pages_crawled,
            chunks_created=len(chunks),
            document_ids=document_ids,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error crawling website: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error crawling website: {str(e)}"
        )
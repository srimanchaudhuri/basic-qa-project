import datetime

from fastapi import APIRouter, HTTPException

from app.api.schemas import HealthResponse, ReadinessResponse
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])

@router.get(
    "",
    response_model=HealthResponse,
    summary="Health check enpoint"
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.datetime.now(),
        version="1.0.0"
    )

@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Checks if the database is connected or not"
)
async def readiness_check() -> ReadinessResponse:
    vector_store = VectorStoreService()
    is_ready = vector_store.health_check()

    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not ready"
        )

    collection_info = vector_store.get_collection_info()

    return ReadinessResponse(
        status="healthy",
        qdrant_connected=is_ready,
        collection_info=collection_info
    )
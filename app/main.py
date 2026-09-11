import os
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import document, health, query
from app.config import get_settings
from app import __version__
from app.utils.logger import get_logger, setup_logger

load_dotenv()
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger(settings.log_level)
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{__version__}")
    logger.info(f"LOG LEVEL: {settings.log_level}")

    yield

    logger.info("Shutting down application")

app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health.router)
app.include_router(document.router)
app.include_router(query.router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


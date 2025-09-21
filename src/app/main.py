"""
FastAPI application main module
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.app.api.endpoints import ingest, query
from src.app.api.lifespan import lifespan
from src.app.settings import get_settings


@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """Application lifespan manager"""
    import threading

    thread_id = threading.get_ident()
    logger.info(f"[Lifespan Thread {thread_id}] Starting GenericRAG application...")

    # Initialize settings
    settings = get_settings()
    app.state.settings = settings

    # Initialize Qdrant client
    from src.app.api.state import create_qdrant_client

    app.state.qdrant_client = create_qdrant_client(settings)

    # Initialize image storage
    from src.app.services.image_storage import LocalImageStorage

    app.state.image_storage = LocalImageStorage(
        storage_path=settings.storage.image_storage_path,
        temp_path=settings.storage.temp_storage_path,
    )

    # Initialize Instructor client
    import instructor

    app.state.instructor_client = instructor.AsyncInstructor(
        client_type="httpx",
        base_url=settings.llm.gemma_base_url,
        api_key=settings.llm.gemma_api_key,
    )

    # Initialize DSPy integration service
    from src.app.services.dspy_integration import DSPyIntegrationService

    app.state.dspy_service = DSPyIntegrationService(
        settings, app.state.instructor_client
    )

    logger.info(f"[Lifespan Thread {thread_id}] Application startup completed")

    yield

    # Shutdown
    logger.info(
        f"[Lifespan Thread {thread_id}] Shutting down GenericRAG application..."
    )
    # Cleanup will be handled by the lifespan context


# Create FastAPI app
app = FastAPI(
    title="GenericRAG - RAG System",
    description="A Retrieval-Augmented Generation system with DSPy/GEPA optimization",
    version="1.0.0",
    lifespan=lifespan_manager,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, prefix="/api/v1", tags=["ingestion"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GenericRAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    settings = get_settings()

    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "qdrant": settings.qdrant.qdrant_url,
            "llm": settings.llm.student_model,
        },
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=True,
        log_level="info",
    )

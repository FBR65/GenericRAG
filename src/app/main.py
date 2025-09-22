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


def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    
    # Create logs directory if it doesn't exist
    import os
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger.add(
        os.path.join(log_dir, "app.log"),
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        colorize=False,
    )
    
    # Add console logging for development
    logger.add(
        lambda msg: print(msg, end=""),
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )


@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """Application lifespan manager"""
    import threading

    # Setup logging
    setup_logging()
    
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

    # Initialize BGE-M3 service
    from src.app.services.bge_m3_service import BGE_M3_Service
    
    app.state.bge_m3_service = None
    if hasattr(settings, 'bge_m3') and settings.bge_m3.model_name:
        try:
            app.state.bge_m3_service = BGE_M3_Service(settings)
            logger.info(f"[Lifespan Thread {thread_id}] BGE-M3 service initialized successfully")
        except Exception as e:
            logger.error(f"[Lifespan Thread {thread_id}] Failed to initialize BGE-M3 service: {e}")
    else:
        logger.info(f"[Lifespan Thread {thread_id}] BGE-M3 service disabled in settings")

    # Initialize Search Service with BGE-M3 support
    from src.app.services.search_service import SearchService
    
    app.state.search_service = SearchService(
        qdrant_client=app.state.qdrant_client,
        image_storage=app.state.image_storage,
        settings=settings,
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

    # Check BGE-M3 service status
    bge_m3_status = "disabled"
    if hasattr(settings, 'bge_m3') and settings.bge_m3.model_name:
        try:
            from src.app.services.bge_m3_service import BGE_M3_Service
            bge_m3_service = BGE_M3_Service(settings)
            bge_m3_status = "healthy" if bge_m3_service else "unavailable"
        except Exception:
            bge_m3_status = "error"

    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "qdrant": settings.qdrant.qdrant_url,
            "llm": settings.llm.student_model,
            "bge_m3": bge_m3_status,
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

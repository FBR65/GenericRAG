"""
FastAPI dependency injection
"""
import asyncio
from pathlib import Path
from typing import Annotated

import instructor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from fastapi import Depends
from instructor import AsyncInstructor
from qdrant_client import AsyncQdrantClient

from src.app.api.state import create_qdrant_client
from src.app.colpali.loaders import ColQwen2_5Loader
from src.app.services.dspy_gepa import DSPyGEPAService
from src.app.services.image_storage import LocalImageStorage
from src.app.settings import Settings, get_settings


# Settings
SettingsDep = Annotated[Settings, Depends(get_settings)]


# Qdrant client
async def get_qdrant_client(settings: SettingsDep) -> AsyncQdrantClient:
    """Get Qdrant client"""
    return create_qdrant_client(settings)


QdrantClientDep = Annotated[AsyncQdrantClient, Depends(get_qdrant_client)]


# ColPali model and processor
async def get_colpali_model_and_processor() -> tuple[ColQwen2_5, ColQwen2_5_Processor]:
    """Get ColPali model and processor"""
    import threading
    import asyncio
    
    thread_id = threading.get_ident()
    logger = get_settings().logger if hasattr(get_settings(), 'logger') else None
    
    if logger:
        logger.info(f"[Thread {thread_id}] get_colpali_model_and_processor called")
        logger.info(f"[Thread {thread_id}] Current event loop: {asyncio.get_event_loop()}")
        logger.info(f"[Thread {thread_id}] Active tasks: {len(asyncio.all_tasks())}")
    
    try:
        loader = ColQwen2_5Loader()
        model, processor = loader.load()
        
        if logger:
            logger.info(f"[Thread {thread_id}] ColPali model loaded successfully")
        
        return model, processor
    except Exception as e:
        if logger:
            logger.error(f"[Thread {thread_id}] Error in get_colpali_model_and_processor: {e}")
        raise


ColPaliModelDep = Annotated[ColQwen2_5, Depends(get_colpali_model_and_processor)]
ColPaliProcessorDep = Annotated[ColQwen2_5_Processor, Depends(get_colpali_model_and_processor)]


# Image storage
async def get_image_storage(settings: SettingsDep) -> LocalImageStorage:
    """Get local image storage"""
    return LocalImageStorage(
        storage_path=settings.storage.image_storage_path,
        temp_path=settings.storage.temp_storage_path,
    )


ImageStorageDep = Annotated[LocalImageStorage, Depends(get_image_storage)]


# Instructor client
async def get_instructor_client(settings: SettingsDep) -> AsyncInstructor:
    """Get Instructor client"""
    return instructor.AsyncInstructor(
        client_type="httpx",
        base_url=settings.llm.gemma_base_url,
        api_key=settings.llm.gemma_api_key,
    )


InstructorClientDep = Annotated[AsyncInstructor, Depends(get_instructor_client)]


# DSPy service
async def get_dspy_service(
    settings: SettingsDep,
    instructor_client: InstructorClientDep,
) -> DSPyGEPAService:
    """Get DSPy/GEPA service"""
    return DSPyGEPAService(settings, instructor_client)


DSPyServiceDep = Annotated[DSPyGEPAService, Depends(get_dspy_service)]


# Initialize Qdrant collection
async def initialize_qdrant_collection(
    qdrant_client: QdrantClientDep,
    settings: SettingsDep,
) -> None:
    """Initialize Qdrant collection"""
    from src.app.api.state import initialize_qdrant_collection as init_collection
    
    await init_collection(
        qdrant_client=qdrant_client,
        collection_name=settings.qdrant.collection_name,
        vector_size=768,  # ColQwen2.5 embedding size
    )


# Application startup dependencies
async def startup_dependencies(
    settings: SettingsDep,
    qdrant_client: QdrantClientDep,
) -> None:
    """Startup dependencies"""
    # Initialize Qdrant collection
    await initialize_qdrant_collection(qdrant_client, settings)
    
    # Initialize other services if needed
    logger = settings.storage.image_storage_path.parent / "logs"
    logger.mkdir(parents=True, exist_ok=True)


# Application shutdown dependencies
async def shutdown_dependencies() -> None:
    """Shutdown dependencies"""
    # Cleanup resources if needed
    pass
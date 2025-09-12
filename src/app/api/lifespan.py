"""
Application lifespan management
"""
import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from loguru import logger

from src.app.api.dependencies import startup_dependencies, shutdown_dependencies
from src.app.settings import Settings, get_settings


async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None
    """
    # Startup
    logger.info("Starting up GenericRAG application...")
    
    try:
        # Get settings
        settings = get_settings()
        
        # Create necessary directories
        directories = [
            settings.storage.image_storage_path,
            settings.storage.temp_storage_path,
            settings.storage.dspy_cache_path,
            settings.storage.image_storage_path.parent / "logs",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        
        # Initialize startup dependencies
        await startup_dependencies(settings, app.state.qdrant_client)
        
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down GenericRAG application...")
        
        try:
            # Cleanup resources
            await shutdown_dependencies()
            
            logger.info("Application shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}")
#!/usr/bin/env python3
"""
Startup script for the Generic RAG API server.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.api.main import app
from src.config import settings

def main():
    """Start the API server."""
    print(f"Starting {settings.system_name} API server...")
    print(f"API will be available at: http://{settings.api_host}:{settings.api_port}")
    print(f"Qdrant connection: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"LLM endpoint: {settings.llm_endpoint}")
    print(f"ColPali model: {settings.colpali_model_name}")
    print(f"Collection: {settings.qdrant_collection_name}")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
        workers=settings.api_workers
    )

if __name__ == "__main__":
    main()
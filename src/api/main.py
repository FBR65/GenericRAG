from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import logging
import asyncio
from pathlib import Path

from ..services.document_processor import DocumentProcessor
from ..config import settings
from ..utils.qdrant_client import QdrantManager

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.system_version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
qdrant_manager = QdrantManager(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key,
    collection_name=settings.qdrant_collection_name
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting {settings.system_name} API")
    logger.info(f"API will run on {settings.api_host}:{settings.api_port}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": settings.system_name, "version": settings.system_version}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        system_status = document_processor.get_system_status()
        return {
            "status": "healthy",
            "timestamp": system_status.get("timestamp"),
            "services": {
                "qdrant": "connected",
                "embedding_model": "loaded",
                "llm_service": "configured" if settings.llm_api_key else "not_configured"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a PDF document for processing."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Validate file size
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        await file.seek(0)  # Reset file pointer
        
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {settings.max_file_size} bytes"
            )
        
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        # Process the document
        result = await document_processor.process_document(file_path)
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_document(
    file_path: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Process a document from a specific file path."""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )
        
        result = await document_processor.process_document(file_path)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(
    query: str = Form(...),
    top_k: Optional[int] = Form(None)
):
    """Search for documents and generate an answer."""
    try:
        result = await document_processor.search_and_answer(query, top_k)
        return result
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete")
async def delete_document(filename: str = Form(...)):
    """Delete a document from the database."""
    try:
        success = qdrant_manager.delete_document(filename)
        
        if success:
            return {"message": f"Document '{filename}' deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-documents")
async def list_documents():
    """List all documents in the database."""
    try:
        documents = qdrant_manager.list_documents()
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document-status")
async def get_document_status(filename: str):
    """Get the status of a specific document."""
    try:
        status = document_processor.get_document_status(filename)
        return status
        
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-status")
async def get_system_status():
    """Get overall system status."""
    try:
        status = document_processor.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-upload")
async def batch_upload(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process multiple PDF documents."""
    try:
        # Validate files
        valid_files = []
        for file in files:
            if file.filename.lower().endswith('.pdf'):
                valid_files.append(file)
            else:
                logger.warning(f"Skipping non-PDF file: {file.filename}")
        
        if not valid_files:
            raise HTTPException(
                status_code=400,
                detail="No valid PDF files found"
            )
        
        # Save files
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_paths = []
        
        for file in valid_files:
            file_path = os.path.join(upload_dir, file.filename)
            contents = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(contents)
            file_paths.append(file_path)
        
        # Process files in batch
        results = await document_processor.process_batch(file_paths)
        
        # Clean up files
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return {
            "total_files": len(valid_files),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-collection")
async def clear_collection():
    """Clear all documents from the collection."""
    try:
        success = qdrant_manager.clear_collection()
        return {"message": "Collection cleared successfully" if success else "Failed to clear collection"}
        
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )
"""
PDF ingestion endpoints
"""
import asyncio
import uuid
from pathlib import Path
from typing import List, Optional

import pdf2image
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from loguru import logger
from PIL import Image
from qdrant_client.http import models as http_models

from src.app.api.dependencies import (
    ColPaliModelDep,
    ColPaliProcessorDep,
    ImageStorageDep,
    QdrantClientDep,
    SettingsDep,
)
from src.app.models.schemas import IngestResult, IngestResponse
from src.app.utils.colpali_utils import generate_embeddings, preprocess_images
from src.app.utils.qdrant_utils import (
    create_payload_filter,
    create_collection_if_not_exists,
    upsert_with_retry,
)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    qdrant_client: QdrantClientDep,
    colpali_model: ColPaliModelDep,
    colpali_processor: ColPaliProcessorDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    session_id: Optional[str] = None,
    file: UploadFile = File(...),
) -> IngestResponse:
    """
    Ingest a PDF file into the RAG system
    
    Args:
        file: PDF file to ingest
        session_id: Optional session identifier
        qdrant_client: Qdrant client
        colpali_model: ColPali model
        colpali_processor: ColPali processor
        image_storage: Image storage service
        settings: Application settings
        
    Returns:
        Ingest response with results
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    results = []
    
    try:
        # Read PDF file
        pdf_content = await file.read()
        
        # Save PDF temporarily
        temp_pdf_path = Path(settings.storage.temp_storage_path) / f"{uuid.uuid4()}.pdf"
        temp_pdf_path.write_bytes(pdf_content)
        
        # Convert PDF to images
        logger.info(f"Converting PDF to images: {file.filename}")
        images = pdf2image.convert_from_path(
            str(temp_pdf_path),
            dpi=300,
            fmt='jpeg',
            thread_count=4,
        )
        
        # Preprocess images
        processed_images = preprocess_images(images)
        
        if not processed_images:
            raise HTTPException(status_code=400, detail="No valid images could be extracted from PDF")
        
        # Store images locally
        logger.info(f"Storing {len(processed_images)} images locally")
        stored_paths = await image_storage.store_images(
            session_id=session_id,
            file_name=file.filename,
            images=processed_images,
        )
        
        # Generate embeddings using ColPali
        logger.info("Generating embeddings with ColPali")
        embeddings = generate_embeddings(
            model=colpali_model,
            processor=colpali_processor,
            images=processed_images,
            batch_size=4,
        )
        
        # Prepare points for Qdrant
        points = []
        for i, (embedding, stored_path) in enumerate(zip(embeddings, stored_paths)):
            point = {
                "id": i,
                "vector": embedding.tolist(),
                "payload": {
                    "session_id": session_id,
                    "document": file.filename,
                    "page": i + 1,
                    "image_path": stored_path,
                    "created_at": str(asyncio.get_event_loop().time()),
                },
            }
            points.append(point)
        
        # Upsert to Qdrant
        logger.info(f"Upserting {len(points)} points to Qdrant")
        await upsert_with_retry(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.collection_name,
            points=points,
            batch_size=100,
        )
        
        # Clean up temporary PDF
        temp_pdf_path.unlink()
        
        # Create success result
        result = IngestResult(
            filename=file.filename,
            num_pages=len(processed_images),
            status="success",
        )
        
        results.append(result)
        logger.info(f"Successfully ingested {file.filename}")
        
    except Exception as e:
        logger.error(f"Error ingesting PDF {file.filename}: {e}")
        
        # Create error result
        result = IngestResult(
            filename=file.filename,
            status="error",
            error=str(e),
        )
        
        results.append(result)
        
        # Clean up temporary files if they exist
        try:
            if 'temp_pdf_path' in locals():
                temp_pdf_path.unlink()
        except:
            pass
    
    return IngestResponse(results=results)


@router.post("/ingest-batch")
async def ingest_multiple_pdfs(
    qdrant_client: QdrantClientDep,
    colpali_model: ColPaliModelDep,
    colpali_processor: ColPaliProcessorDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    session_id: Optional[str] = None,
    files: List[UploadFile] = File(...),
) -> IngestResponse:
    """
    Ingest multiple PDF files in batch
    
    Args:
        files: List of PDF files to ingest
        session_id: Optional session identifier
        qdrant_client: Qdrant client
        colpali_model: ColPali model
        colpali_processor: ColPali processor
        image_storage: Image storage service
        settings: Application settings
        
    Returns:
        Ingest response with results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    results = []
    
    # Process each file
    for file in files:
        try:
            # Read PDF file
            pdf_content = await file.read()
            
            # Save PDF temporarily
            temp_pdf_path = Path(settings.storage.temp_storage_path) / f"{uuid.uuid4()}.pdf"
            temp_pdf_path.write_bytes(pdf_content)
            
            # Convert PDF to images
            logger.info(f"Converting PDF to images: {file.filename}")
            images = pdf2image.convert_from_path(
                str(temp_pdf_path),
                dpi=300,
                fmt='jpeg',
                thread_count=4,
            )
            
            # Preprocess images
            processed_images = preprocess_images(images)
            
            if not processed_images:
                raise HTTPException(status_code=400, detail="No valid images could be extracted from PDF")
            
            # Store images locally
            logger.info(f"Storing {len(processed_images)} images locally")
            stored_paths = await image_storage.store_images(
                session_id=session_id,
                file_name=file.filename,
                images=processed_images,
            )
            
            # Generate embeddings using ColPali
            logger.info("Generating embeddings with ColPali")
            embeddings = generate_embeddings(
                model=colpali_model,
                processor=colpali_processor,
                images=processed_images,
                batch_size=4,
            )
            
            # Prepare points for Qdrant
            points = []
            for i, (embedding, stored_path) in enumerate(zip(embeddings, stored_paths)):
                point = {
                    "id": len(results) * 1000 + i,  # Unique ID across files
                    "vector": embedding.tolist(),
                    "payload": {
                        "session_id": session_id,
                        "document": file.filename,
                        "page": i + 1,
                        "image_path": stored_path,
                        "created_at": str(asyncio.get_event_loop().time()),
                    },
                }
                points.append(point)
            
            # Upsert to Qdrant
            logger.info(f"Upserting {len(points)} points to Qdrant")
            await upsert_with_retry(
                qdrant_client=qdrant_client,
                collection_name=settings.qdrant.collection_name,
                points=points,
                batch_size=100,
            )
            
            # Clean up temporary PDF
            temp_pdf_path.unlink()
            
            # Create success result
            result = IngestResult(
                filename=file.filename,
                num_pages=len(processed_images),
                status="success",
            )
            
            results.append(result)
            logger.info(f"Successfully ingested {file.filename}")
            
        except Exception as e:
            logger.error(f"Error ingesting PDF {file.filename}: {e}")
            
            # Create error result
            result = IngestResult(
                filename=file.filename,
                status="error",
                error=str(e),
            )
            
            results.append(result)
            
            # Clean up temporary files if they exist
            try:
                if 'temp_pdf_path' in locals():
                    temp_pdf_path.unlink()
            except:
                pass
    
    return IngestResponse(results=results)


@router.get("/sessions/{session_id}/documents")
async def get_session_documents(
    qdrant_client: QdrantClientDep,
    settings: SettingsDep,
    session_id: str,
) -> List[str]:
    """
    Get list of documents in a session
    
    Args:
        session_id: Session identifier
        qdrant_client: Qdrant client
        settings: Application settings
        
    Returns:
        List of document names
    """
    try:
        # Search for documents in the session
        response = await qdrant_client.scroll(
            collection_name=settings.qdrant.collection_name,
            scroll_filter=create_payload_filter(session_id=session_id),
            limit=1000,
            with_payload=True,
        )
        
        # Extract unique document names
        documents = set()
        for point in response[0]:
            documents.add(point.payload.get("document", ""))
        
        return list(documents)
        
    except Exception as e:
        logger.error(f"Error getting session documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(
    qdrant_client: QdrantClientDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    session_id: str,
) -> dict:
    """
    Delete a session and all its data
    
    Args:
        session_id: Session identifier
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings
        
    Returns:
        Deletion result
    """
    try:
        # Get all image paths for the session
        response = await qdrant_client.scroll(
            collection_name=settings.qdrant.collection_name,
            scroll_filter=create_payload_filter(session_id=session_id),
            limit=1000,
            with_payload=True,
        )
        
        image_paths = []
        for point in response[0]:
            image_path = point.payload.get("image_path")
            if image_path:
                image_paths.append(image_path)
        
        # Delete images from storage
        if image_paths:
            await image_storage.delete_images(image_paths)
        
        # Delete points from Qdrant
        await qdrant_client.delete(
            collection_name=settings.qdrant.collection_name,
            points_selector=http_models.Filter(
                must=[http_models.FieldCondition(key="session_id", match=http_models.MatchValue(value=session_id))]
            ),
        )
        
        return {
            "status": "success",
            "message": f"Session {session_id} deleted",
            "deleted_images": len(image_paths),
        }
        
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
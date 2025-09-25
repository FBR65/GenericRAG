"""
PDF ingestion endpoints
"""

import asyncio
import uuid
import json
from pathlib import Path
from typing import List, Optional

import pdf2image
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from loguru import logger
from PIL import Image
from qdrant_client.http import models as http_models

from src.app.api.dependencies import (
    ImageStorageDep,
    QdrantClientDep,
    SettingsDep,
    SearchServiceDep,
)
from src.app.models.schemas import (
    IngestResult,
    IngestResponse,
    ExtractionResult,
    BGE_M3_IngestRequest,
    BGE_M3_IngestResult,
    BGE_M3_EmbeddingResponse,
    BGE_M3_BatchEmbeddingResponse,
    BGE_M3_EmbeddingType
)
from src.app.services import PDFExtractor, TextPreprocessor
from src.app.services.image_embedding_service import ImageEmbeddingService
from src.app.services.search_service import SearchService
from src.app.utils.qdrant_utils import (
    create_payload_filter,
    create_collection_if_not_exists,
    upsert_with_retry,
    create_hybrid_point,
    upsert_hybrid_chunks_with_retry,
    create_image_collection_if_not_exists,
    upsert_image_embeddings_with_retry,
)

router = APIRouter()


@router.post("/ingest-bge-m3", response_model=IngestResponse)
async def ingest_bge_m3(
    qdrant_client: QdrantClientDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    request: BGE_M3_IngestRequest,
    search_service: SearchServiceDep,
    file: UploadFile = File(...),
) -> IngestResponse:
    """
    BGE-M3 specific ingestion endpoint with advanced embedding capabilities
    
    This endpoint provides specialized ingestion functionality using BGE-M3's
    all-in-one embedding model to generate dense, sparse, and multivector
    embeddings for PDF documents.

    Args:
        request: BGE-M3 specific ingestion parameters
        file: PDF file to ingest
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings

    Returns:
        BGE-M3 specific ingestion response with detailed embedding information
    """
    import time
    start_time = time.time()
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate session ID if not provided
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    results = []
    
    try:
        # Initialize search service for BGE-M3 access
        from src.app.services.search_service import SearchService
        search_service = SearchService(qdrant_client, image_storage, settings)
        
        if not search_service.bge_m3_service:
            raise HTTPException(
                status_code=503,
                detail="BGE-M3 service is not available"
            )
        
        logger.info(f"Processing BGE-M3 ingestion for: {file.filename}")
        
        # Read PDF file
        pdf_content = await file.read()

        # Save PDF temporarily
        temp_pdf_path = Path(settings.storage.temp_storage_path) / f"{uuid.uuid4()}.pdf"
        temp_pdf_path.write_bytes(pdf_content)

        # Initialize PDF extractor
        extractor = PDFExtractor(
            output_dir=str(
                Path(settings.storage.temp_storage_path) / "extracted_images"
            )
        )

        # Extract PDF data
        logger.info(f"Extracting PDF data: {file.filename}")
        extraction_result: ExtractionResult = extractor.extract_to_pydantic(
            str(temp_pdf_path)
        )

        # Get images for embedding generation
        logger.info(f"Extracting images for embedding generation: {file.filename}")
        images_for_embedding = extractor.get_images_for_embedding(str(temp_pdf_path))

        # Initialize image embedding service
        image_embedding_service = ImageEmbeddingService()

        # Process images for embedding generation
        image_embeddings = []
        for image_info in images_for_embedding:
            try:
                logger.info(f"Processing image for embedding: {image_info['filename']}")

                # Generate CLIP embedding
                embedding = image_embedding_service.generate_clip_embedding_from_path(
                    image_info["file_path"]
                )

                if embedding and image_embedding_service.validate_embedding(embedding):
                    # Create image metadata
                    image_metadata = image_embedding_service.create_image_metadata(
                        image_path=image_info["file_path"],
                        page_number=image_info["page_number"],
                        document_id=Path(file.filename).stem,
                        chunk_id=f"page_{image_info['page_number']}_img_{len(image_embeddings)}",
                    )
                    image_metadata["created_at"] = str(asyncio.get_event_loop().time())
                    image_metadata["bge_m3_used"] = True

                    image_embeddings.append(
                        {
                            "id": f"img_{request.session_id}_{len(image_embeddings)}",
                            "vector": embedding,
                            "payload": image_metadata,
                        }
                    )
                    logger.info(
                        f"Successfully generated embedding for {image_info['filename']}"
                    )
                else:
                    logger.warning(
                        f"Failed to generate embedding for {image_info['filename']}"
                    )

            except Exception as e:
                logger.error(f"Error processing image {image_info['filename']}: {e}")
                continue

        # Store extracted images locally
        logger.info(
            f"Storing extracted images for {len(extraction_result.pages)} pages"
        )
        stored_paths = []
        for page in extraction_result.pages:
            for element in page.elements:
                if element.type == "image" and element.file_path:
                    # Move extracted images to proper storage location
                    original_path = Path(element.file_path)
                    if original_path.exists():
                        new_path = (
                            Path(settings.storage.temp_storage_path)
                            / "extracted_images"
                            / original_path.name
                        )
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        original_path.rename(new_path)
                        element.file_path = str(new_path)
                        stored_paths.append(str(new_path))

        # Initialize text preprocessor with settings
        text_preprocessor = TextPreprocessor(settings=settings)

        # Create semantic chunks from extracted elements
        logger.info("Creating semantic chunks from extracted elements")
        chunks = text_preprocessor.create_chunks(extraction_result)

        # Generate BGE-M3 embeddings for each chunk
        logger.info("Generating BGE-M3 embeddings for chunks")
        processed_chunks = []
        
        try:
            # Verarbeite Chunks mit BGE-M3
            processed_chunks = await text_preprocessor.process_chunks_with_bge_m3_embeddings(
                chunks,
                include_dense=request.include_dense,
                include_sparse=request.include_sparse,
                include_multivector=request.include_multivector,
                batch_size=request.batch_size,
                cache_embeddings=request.cache_embeddings,
                session_id=request.session_id
            )
            
            # Zähle generierte Embeddings
            embeddings_generated = {"dense": 0, "sparse": 0, "multivector": 0}
            cache_hits = 0
            
            for chunk in processed_chunks:
                if "dense_vector" in chunk:
                    embeddings_generated["dense"] += 1
                if "sparse_vector" in chunk:
                    embeddings_generated["sparse"] += 1
                if "multivector" in chunk:
                    embeddings_generated["multivector"] += 1
                    
            logger.info(f"BGE-M3 embeddings generated: {embeddings_generated}")
            
        except Exception as e:
            logger.error(f"Error generating BGE-M3 embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"BGE-M3 embedding generation failed: {str(e)}")

        # Prepare chunks for hybrid storage
        hybrid_chunks = []
        for chunk in processed_chunks:
            # Prepare payload with hierarchical structure
            payload = {
                "session_id": request.session_id,
                "document": file.filename,
                "page": chunk["page_number"],
                "element_type": chunk["type"],
                "bbox": chunk["bbox"],
                "content": chunk["content"],
                "created_at": str(asyncio.get_event_loop().time()),
                "element_count": chunk.get("element_count", 1),
                "bge_m3_used": True,
                "embedding_types": list(embeddings_generated.keys()),
            }

            # Add specific metadata based on element type
            if chunk["type"] == "table":
                payload["table_content"] = chunk["content"]
            elif chunk["type"] == "text":
                payload["text_content"] = chunk["content"]
            elif chunk["type"] == "image":
                payload["image_path"] = chunk.get("file_path")

            hybrid_chunk = {
                "id": len(hybrid_chunks),
                "dense_vector": chunk.get("dense_vector", [0.0] * 1024),
                "sparse_vector": chunk.get("sparse_vector", {}),
                "payload": payload,
            }
            hybrid_chunks.append(hybrid_chunk)

        # Save hierarchical JSON structure
        json_output_path = (
            Path(settings.storage.temp_storage_path)
            / f"{uuid.uuid4()}_hierarchical.json"
        )
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(extraction_result.model_dump(), f, ensure_ascii=False, indent=2)

        # Store JSON path in Qdrant for reference
        json_chunk = {
            "id": len(hybrid_chunks),
            "dense_vector": [0.0] * 1024,  # Placeholder for JSON metadata
            "sparse_vector": {},  # Empty sparse vector for JSON
            "payload": {
                "session_id": request.session_id,
                "document": file.filename,
                "json_path": str(json_output_path),
                "created_at": str(asyncio.get_event_loop().time()),
                "element_type": "json_metadata",
                "bge_m3_used": True,
            },
        }
        hybrid_chunks.append(json_chunk)

        # Create hybrid collection if not exists
        await create_collection_if_not_exists(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.collection_name,
            dense_vector_size=settings.qdrant.dense_dimension,
            sparse_vector_size=settings.qdrant.sparse_max_features,
        )

        # Create image collection if not exists
        await create_image_collection_if_not_exists(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.image_collection_name,
            vector_size=settings.qdrant.clip_dimension,
        )

        # Upsert hybrid chunks to Qdrant
        logger.info(f"Upserting {len(hybrid_chunks)} hybrid chunks to Qdrant")
        await upsert_hybrid_chunks_with_retry(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.collection_name,
            chunks=hybrid_chunks,
            batch_size=100,
        )

        # Upsert image embeddings to Qdrant
        if image_embeddings:
            logger.info(f"Upserting {len(image_embeddings)} image embeddings to Qdrant")
            await upsert_image_embeddings_with_retry(
                qdrant_client=qdrant_client,
                collection_name=settings.qdrant.image_collection_name,
                embeddings=image_embeddings,
                batch_size=50,
            )
            logger.info(f"Successfully stored {len(image_embeddings)} image embeddings")

        # Clean up temporary PDF
        temp_pdf_path.unlink()

        # Create success result
        processing_time = time.time() - start_time
        
        result = BGE_M3_IngestResult(
            filename=file.filename,
            num_pages=extraction_result.total_pages,
            status="success",
            embeddings_generated=embeddings_generated,
            processing_time=processing_time,
            cache_hits=cache_hits,
            session_id=request.session_id,
            bge_m3_metadata={
                "embedding_types_used": list(embeddings_generated.keys()),
                "total_chunks_processed": len(processed_chunks),
                "batch_size_used": request.batch_size,
                "cache_enabled": request.cache_embeddings,
            }
        )

        results.append(result)
        logger.info(
            f"Successfully ingested {file.filename} with BGE-M3 embeddings: {embeddings_generated}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting PDF {file.filename} with BGE-M3: {e}")
        processing_time = time.time() - start_time

        # Create error result
        result = BGE_M3_IngestResult(
            filename=file.filename,
            status="error",
            error=str(e),
            embeddings_generated={},
            processing_time=processing_time,
            cache_hits=0,
            session_id=request.session_id,
            bge_m3_metadata={}
        )

        results.append(result)

        # Clean up temporary files if they exist
        try:
            if "temp_pdf_path" in locals():
                temp_pdf_path.unlink()
        except:
            pass

    return IngestResponse(results=results)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    qdrant_client: QdrantClientDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    search_service: SearchServiceDep,
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    use_bge_m3: bool = False,
    embedding_types: str = "all",
    include_dense: bool = True,
    include_sparse: bool = True,
    include_multivector: bool = True,
    batch_size: int = 32,
    cache_embeddings: bool = True,
) -> IngestResponse:
    """
    Ingest a PDF file into the RAG system with BGE-M3 support

    Args:
        file: PDF file to ingest
        session_id: Optional session identifier
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings
        use_bge_m3: Whether to use BGE-M3 for embedding generation
        embedding_types: Types of embeddings to generate (dense, sparse, multivector, all)
        include_dense: Whether to generate dense embeddings
        include_sparse: Whether to generate sparse embeddings
        include_multivector: Whether to generate multivector embeddings
        batch_size: Batch size for processing
        cache_embeddings: Whether to cache generated embeddings

    Returns:
        Ingest response with results including BGE-M3 support
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    results = []
    
    # BGE-M3 spezifische Variablen
    bge_m3_used = False
    embeddings_generated = {"dense": 0, "sparse": 0, "multivector": 0}
    cache_hits = 0
    processing_start_time = asyncio.get_event_loop().time()

    try:
        # Read PDF file
        pdf_content = await file.read()

        # Save PDF temporarily
        temp_pdf_path = Path(settings.storage.temp_storage_path) / f"{uuid.uuid4()}.pdf"
        temp_pdf_path.write_bytes(pdf_content)

        # Initialize PDF extractor
        extractor = PDFExtractor(
            output_dir=str(
                Path(settings.storage.temp_storage_path) / "extracted_images"
            )
        )

        # Extract PDF data using the new PDF processor
        logger.info(f"Extracting PDF data: {file.filename}")
        extraction_result: ExtractionResult = extractor.extract_to_pydantic(
            str(temp_pdf_path)
        )

        # Get images for embedding generation
        logger.info(f"Extracting images for embedding generation: {file.filename}")
        images_for_embedding = extractor.get_images_for_embedding(str(temp_pdf_path))

        # Initialize image embedding service
        image_embedding_service = ImageEmbeddingService()

        # Process images for embedding generation
        image_embeddings = []
        for image_info in images_for_embedding:
            try:
                logger.info(f"Processing image for embedding: {image_info['filename']}")

                # Generate CLIP embedding
                embedding = image_embedding_service.generate_clip_embedding_from_path(
                    image_info["file_path"]
                )

                if embedding and image_embedding_service.validate_embedding(embedding):
                    # Create image metadata
                    image_metadata = image_embedding_service.create_image_metadata(
                        image_path=image_info["file_path"],
                        page_number=image_info["page_number"],
                        document_id=Path(file.filename).stem,
                        chunk_id=f"page_{image_info['page_number']}_img_{len(image_embeddings)}",
                    )
                    image_metadata["created_at"] = str(asyncio.get_event_loop().time())

                    image_embeddings.append(
                        {
                            "id": f"img_{session_id}_{len(image_embeddings)}",
                            "vector": embedding,
                            "payload": image_metadata,
                        }
                    )
                    logger.info(
                        f"Successfully generated embedding for {image_info['filename']}"
                    )
                else:
                    logger.warning(
                        f"Failed to generate embedding for {image_info['filename']}"
                    )

            except Exception as e:
                logger.error(f"Error processing image {image_info['filename']}: {e}")
                continue

        # Store extracted images locally
        logger.info(
            f"Storing extracted images for {len(extraction_result.pages)} pages"
        )
        stored_paths = []
        for page in extraction_result.pages:
            for element in page.elements:
                if element.type == "image" and element.file_path:
                    # Move extracted images to proper storage location
                    original_path = Path(element.file_path)
                    if original_path.exists():
                        new_path = (
                            Path(settings.storage.temp_storage_path)
                            / "extracted_images"
                            / original_path.name
                        )
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        original_path.rename(new_path)
                        element.file_path = str(new_path)
                        stored_paths.append(str(new_path))

        # Initialize text preprocessor with settings
        text_preprocessor = TextPreprocessor(settings=settings)

        # Create semantic chunks from extracted elements
        logger.info("Creating semantic chunks from extracted elements")
        chunks = text_preprocessor.create_chunks(extraction_result)

        # Generate embeddings for each chunk
        logger.info("Generating embeddings for chunks")
        
        if use_bge_m3:
            logger.info("Using BGE-M3 for embedding generation")
            try:
                # Verwende BGE-M3 für Embedding-Generierung
                processed_chunks = await text_preprocessor.process_chunks_with_bge_m3_embeddings(
                    chunks,
                    include_dense=include_dense,
                    include_sparse=include_sparse,
                    include_multivector=include_multivector,
                    batch_size=batch_size,
                    cache_embeddings=cache_embeddings,
                    session_id=session_id
                )
                bge_m3_used = True
                
                # Zähle generierte Embeddings
                for chunk in processed_chunks:
                    if "dense_vector" in chunk:
                        embeddings_generated["dense"] += 1
                    if "sparse_vector" in chunk:
                        embeddings_generated["sparse"] += 1
                    if "multivector" in chunk:
                        embeddings_generated["multivector"] += 1
                        
            except Exception as e:
                logger.error(f"Error generating BGE-M3 embeddings: {e}")
                # Fallback zur ursprünglichen Methode
                bge_m3_used = False
                logger.info("Falling back to standard embedding generation")
                processed_chunks = await text_preprocessor.process_chunks_with_embeddings(
                    chunks
                )
        else:
            # Verwende Standard Embedding-Generierung
            processed_chunks = await text_preprocessor.process_chunks_with_embeddings(
                chunks
            )

        # Prepare chunks for hybrid storage
        hybrid_chunks = []
        for chunk in processed_chunks:
            # Prepare payload with hierarchical structure
            payload = {
                "session_id": session_id,
                "document": file.filename,
                "page": chunk["page_number"],
                "element_type": chunk["type"],
                "bbox": chunk["bbox"],
                "content": chunk["content"],
                "created_at": str(asyncio.get_event_loop().time()),
                "element_count": chunk.get("element_count", 1),
            }

            # Add specific metadata based on element type
            if chunk["type"] == "table":
                payload["table_content"] = chunk["content"]
            elif chunk["type"] == "text":
                payload["text_content"] = chunk["content"]
            elif chunk["type"] == "image":
                payload["image_path"] = chunk.get("file_path")

            hybrid_chunk = {
                "id": len(hybrid_chunks),
                "dense_vector": chunk.get("dense_vector", [0.0] * 1024),
                "sparse_vector": chunk.get("sparse_vector", {}),
                "payload": payload,
            }
            hybrid_chunks.append(hybrid_chunk)

        # Save hierarchical JSON structure
        json_output_path = (
            Path(settings.storage.temp_storage_path)
            / f"{uuid.uuid4()}_hierarchical.json"
        )
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(extraction_result.model_dump(), f, ensure_ascii=False, indent=2)

        # Store JSON path in Qdrant for reference
        json_chunk = {
            "id": len(hybrid_chunks),
            "dense_vector": [0.0] * 1024,  # Placeholder for JSON metadata
            "sparse_vector": {},  # Empty sparse vector for JSON
            "payload": {
                "session_id": session_id,
                "document": file.filename,
                "json_path": str(json_output_path),
                "created_at": str(asyncio.get_event_loop().time()),
                "element_type": "json_metadata",
            },
        }
        hybrid_chunks.append(json_chunk)

        # Create hybrid collection if not exists
        await create_collection_if_not_exists(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.collection_name,
            dense_vector_size=settings.qdrant.dense_dimension,
            sparse_vector_size=settings.qdrant.sparse_max_features,
        )

        # Create image collection if not exists
        await create_image_collection_if_not_exists(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.image_collection_name,
            vector_size=settings.qdrant.clip_dimension,
        )

        # Upsert hybrid chunks to Qdrant
        logger.info(f"Upserting {len(hybrid_chunks)} hybrid chunks to Qdrant")
        await upsert_hybrid_chunks_with_retry(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.collection_name,
            chunks=hybrid_chunks,
            batch_size=100,
        )

        # Upsert image embeddings to Qdrant
        if image_embeddings:
            logger.info(f"Upserting {len(image_embeddings)} image embeddings to Qdrant")
            await upsert_image_embeddings_with_retry(
                qdrant_client=qdrant_client,
                collection_name=settings.qdrant.image_collection_name,
                embeddings=image_embeddings,
                batch_size=50,
            )
            logger.info(f"Successfully stored {len(image_embeddings)} image embeddings")

        # Clean up temporary PDF
        temp_pdf_path.unlink()

        # Create success result
        processing_time = asyncio.get_event_loop().time() - processing_start_time
        
        result = IngestResult(
            filename=file.filename,
            num_pages=extraction_result.total_pages,
            status="success",
            embeddings_generated=embeddings_generated if bge_m3_used else {},
            processing_time=processing_time if bge_m3_used else None,
            cache_hits=cache_hits if bge_m3_used else 0,
            bge_m3_used=bge_m3_used,
            embedding_types=list(embeddings_generated.keys()) if bge_m3_used else []
        )

        results.append(result)
        logger.info(
            f"Successfully ingested {file.filename} with {len(hybrid_chunks)} extracted elements"
            f"{' using BGE-M3' if bge_m3_used else ''}"
        )

    except Exception as e:
        logger.error(f"Error ingesting PDF {file.filename}: {e}")

        # Create error result
        processing_time = asyncio.get_event_loop().time() - processing_start_time
        
        result = IngestResult(
            filename=file.filename,
            status="error",
            error=str(e),
            embeddings_generated=embeddings_generated if bge_m3_used else {},
            processing_time=processing_time if bge_m3_used else None,
            cache_hits=cache_hits if bge_m3_used else 0,
            bge_m3_used=bge_m3_used,
            embedding_types=list(embeddings_generated.keys()) if bge_m3_used else []
        )

        results.append(result)

        # Clean up temporary files if they exist
        try:
            if "temp_pdf_path" in locals():
                temp_pdf_path.unlink()
        except:
            pass

    return IngestResponse(results=results)


@router.post("/ingest-batch")
async def ingest_multiple_pdfs(
    qdrant_client: QdrantClientDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    search_service: SearchServiceDep,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
    use_bge_m3: bool = False,
    embedding_types: str = "all",
    include_dense: bool = True,
    include_sparse: bool = True,
    include_multivector: bool = True,
    batch_size: int = 32,
    cache_embeddings: bool = True,
    max_concurrent: int = 3,
    progress_callback: Optional[str] = None,
) -> IngestResponse:
    """
    Ingest multiple PDF files in batch with BGE-M3 support

    Args:
        files: List of PDF files to ingest
        session_id: Optional session identifier
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings
        use_bge_m3: Whether to use BGE-M3 for embedding generation
        embedding_types: Types of embeddings to generate (dense, sparse, multivector, all)
        include_dense: Whether to generate dense embeddings
        include_sparse: Whether to generate sparse embeddings
        include_multivector: Whether to generate multivector embeddings
        batch_size: Batch size for processing
        cache_embeddings: Whether to cache generated embeddings
        max_concurrent: Maximum number of concurrent processing tasks
        progress_callback: Optional callback URL for progress updates

    Returns:
        Batch ingestion response with detailed results including BGE-M3 support
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    results = []
    
    # BGE-M3 spezifische Variablen
    bge_m3_used = False
    total_embeddings_generated = {"dense": 0, "sparse": 0, "multivector": 0}
    total_cache_hits = 0
    batch_start_time = asyncio.get_event_loop().time()
    
    # Filter nur PDF-Dateien
    pdf_files = [f for f in files if f.filename.lower().endswith(".pdf")]
    if len(pdf_files) != len(files):
        logger.warning(f"Filtered {len(files) - len(pdf_files)} non-PDF files")

    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF files provided")

    logger.info(f"Starting batch ingestion of {len(pdf_files)} PDF files with BGE-M3: {use_bge_m3}")

    # Verarbeite Dateien parallel mit Begrenzung
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_file(file: UploadFile) -> IngestResult:
        async with semaphore:
            try:
                file_start_time = asyncio.get_event_loop().time()
                
                # Erstelle temporäre Session-ID für jede Datei
                file_session_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
                
                # Rufe den einzelnen Ingest-Endpoint auf
                from src.app.api.endpoints.ingest import ingest_pdf
                
                # Erstelle eine Kopie der Request-Parameter
                request_data = {
                    "query": "",  # Dummy query
                    "session_id": file_session_id,
                    "use_bge_m3": use_bge_m3,
                    "embedding_types": embedding_types,
                    "include_dense": include_dense,
                    "include_sparse": include_sparse,
                    "include_multivector": include_multivector,
                    "batch_size": batch_size,
                    "cache_embeddings": cache_embeddings,
                }
                
                # Verwende die gleiche Logik wie im einzelnen Ingest-Endpoint
                if not file.filename.lower().endswith(".pdf"):
                    return IngestResult(
                        filename=file.filename,
                        status="error",
                        error="Only PDF files are supported"
                    )

                # Read PDF file
                pdf_content = await file.read()

                # Save PDF temporarily
                temp_pdf_path = Path(settings.storage.temp_storage_path) / f"{uuid.uuid4()}.pdf"
                temp_pdf_path.write_bytes(pdf_content)

                # Initialize PDF extractor
                extractor = PDFExtractor(
                    output_dir=str(
                        Path(settings.storage.temp_storage_path) / "extracted_images"
                    )
                )

                # Extract PDF data
                logger.info(f"Extracting PDF data: {file.filename}")
                extraction_result: ExtractionResult = extractor.extract_to_pydantic(
                    str(temp_pdf_path)
                )

                # Get images for embedding generation
                logger.info(f"Extracting images for embedding generation: {file.filename}")
                images_for_embedding = extractor.get_images_for_embedding(str(temp_pdf_path))

                # Initialize image embedding service
                image_embedding_service = ImageEmbeddingService()

                # Process images for embedding generation
                image_embeddings = []
                for image_info in images_for_embedding:
                    try:
                        logger.info(f"Processing image for embedding: {image_info['filename']}")

                        # Generate CLIP embedding
                        embedding = image_embedding_service.generate_clip_embedding_from_path(
                            image_info["file_path"]
                        )

                        if embedding and image_embedding_service.validate_embedding(embedding):
                            # Create image metadata
                            image_metadata = image_embedding_service.create_image_metadata(
                                image_path=image_info["file_path"],
                                page_number=image_info["page_number"],
                                document_id=Path(file.filename).stem,
                                chunk_id=f"page_{image_info['page_number']}_img_{len(image_embeddings)}",
                            )
                            image_metadata["created_at"] = str(asyncio.get_event_loop().time())

                            image_embeddings.append(
                                {
                                    "id": f"img_{file_session_id}_{len(image_embeddings)}",
                                    "vector": embedding,
                                    "payload": image_metadata,
                                }
                            )
                            logger.info(
                                f"Successfully generated embedding for {image_info['filename']}"
                            )
                        else:
                            logger.warning(
                                f"Failed to generate embedding for {image_info['filename']}"
                            )

                    except Exception as e:
                        logger.error(f"Error processing image {image_info['filename']}: {e}")
                        continue

                # Store extracted images locally
                logger.info(
                    f"Storing extracted images for {len(extraction_result.pages)} pages"
                )
                stored_paths = []
                for page in extraction_result.pages:
                    for element in page.elements:
                        if element.type == "image" and element.file_path:
                            # Move extracted images to proper storage location
                            original_path = Path(element.file_path)
                            if original_path.exists():
                                new_path = (
                                    Path(settings.storage.temp_storage_path)
                                    / "extracted_images"
                                    / original_path.name
                                )
                                new_path.parent.mkdir(parents=True, exist_ok=True)
                                original_path.rename(new_path)
                                element.file_path = str(new_path)
                                stored_paths.append(str(new_path))

                # Initialize text preprocessor with settings
                text_preprocessor = TextPreprocessor(settings=settings)

                # Create semantic chunks from extracted elements
                logger.info("Creating semantic chunks from extracted elements")
                chunks = text_preprocessor.create_chunks(extraction_result)

                # Generate embeddings for each chunk
                logger.info("Generating embeddings for chunks")
                
                if use_bge_m3:
                    logger.info("Using BGE-M3 for embedding generation")
                    try:
                        # Verwende BGE-M3 für Embedding-Generierung
                        processed_chunks = await text_preprocessor.process_chunks_with_bge_m3_embeddings(
                            chunks,
                            include_dense=include_dense,
                            include_sparse=include_sparse,
                            include_multivector=include_multivector,
                            batch_size=batch_size,
                            cache_embeddings=cache_embeddings,
                            session_id=file_session_id
                        )
                        bge_m3_used = True
                        
                        # Zähle generierte Embeddings
                        embeddings_generated = {"dense": 0, "sparse": 0, "multivector": 0}
                        cache_hits = 0
                        
                        for chunk in processed_chunks:
                            if "dense_vector" in chunk:
                                embeddings_generated["dense"] += 1
                            if "sparse_vector" in chunk:
                                embeddings_generated["sparse"] += 1
                            if "multivector" in chunk:
                                embeddings_generated["multivector"] += 1
                                
                        logger.info(f"BGE-M3 embeddings generated for {file.filename}: {embeddings_generated}")
                        
                    except Exception as e:
                        logger.error(f"Error generating BGE-M3 embeddings: {e}")
                        # Fallback zur ursprünglichen Methode
                        bge_m3_used = False
                        logger.info("Falling back to standard embedding generation")
                        processed_chunks = await text_preprocessor.process_chunks_with_embeddings(
                            chunks
                        )
                else:
                    # Verwende Standard Embedding-Generierung
                    processed_chunks = await text_preprocessor.process_chunks_with_embeddings(
                        chunks
                    )

                # Prepare chunks for hybrid storage
                hybrid_chunks = []
                for chunk in processed_chunks:
                    # Prepare payload with hierarchical structure
                    payload = {
                        "session_id": file_session_id,
                        "document": file.filename,
                        "page": chunk["page_number"],
                        "element_type": chunk["type"],
                        "bbox": chunk["bbox"],
                        "content": chunk["content"],
                        "created_at": str(asyncio.get_event_loop().time()),
                        "element_count": chunk.get("element_count", 1),
                    }

                    # Add specific metadata based on element type
                    if chunk["type"] == "table":
                        payload["table_content"] = chunk["content"]
                    elif chunk["type"] == "text":
                        payload["text_content"] = chunk["content"]
                    elif chunk["type"] == "image":
                        payload["image_path"] = chunk.get("file_path")

                    hybrid_chunk = {
                        "id": len(hybrid_chunks),
                        "dense_vector": chunk.get("dense_vector", [0.0] * 1024),
                        "sparse_vector": chunk.get("sparse_vector", {}),
                        "payload": payload,
                    }
                    hybrid_chunks.append(hybrid_chunk)

                # Save hierarchical JSON structure
                json_output_path = (
                    Path(settings.storage.temp_storage_path)
                    / f"{uuid.uuid4()}_hierarchical.json"
                )
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(extraction_result.model_dump(), f, ensure_ascii=False, indent=2)

                # Store JSON path in Qdrant for reference
                json_chunk = {
                    "id": len(hybrid_chunks),
                    "dense_vector": [0.0] * 1024,  # Placeholder for JSON metadata
                    "sparse_vector": {},  # Empty sparse vector for JSON
                    "payload": {
                        "session_id": file_session_id,
                        "document": file.filename,
                        "json_path": str(json_output_path),
                        "created_at": str(asyncio.get_event_loop().time()),
                        "element_type": "json_metadata",
                    },
                }
                hybrid_chunks.append(json_chunk)

                # Create hybrid collection if not exists
                await create_collection_if_not_exists(
                    qdrant_client=qdrant_client,
                    collection_name=settings.qdrant.collection_name,
                    dense_vector_size=settings.qdrant.dense_dimension,
                    sparse_vector_size=settings.qdrant.sparse_max_features,
                )

                # Create image collection if not exists
                await create_image_collection_if_not_exists(
                    qdrant_client=qdrant_client,
                    collection_name=settings.qdrant.image_collection_name,
                    vector_size=settings.qdrant.clip_dimension,
                )

                # Upsert hybrid chunks to Qdrant
                logger.info(f"Upserting {len(hybrid_chunks)} hybrid chunks to Qdrant")
                await upsert_hybrid_chunks_with_retry(
                    qdrant_client=qdrant_client,
                    collection_name=settings.qdrant.collection_name,
                    chunks=hybrid_chunks,
                    batch_size=100,
                )

                # Upsert image embeddings to Qdrant
                if image_embeddings:
                    logger.info(f"Upserting {len(image_embeddings)} image embeddings to Qdrant")
                    await upsert_image_embeddings_with_retry(
                        qdrant_client=qdrant_client,
                        collection_name=settings.qdrant.image_collection_name,
                        embeddings=image_embeddings,
                        batch_size=50,
                    )
                    logger.info(f"Successfully stored {len(image_embeddings)} image embeddings")

                # Clean up temporary PDF
                temp_pdf_path.unlink()

                # Create success result
                processing_time = asyncio.get_event_loop().time() - file_start_time
                
                result = IngestResult(
                    filename=file.filename,
                    num_pages=extraction_result.total_pages,
                    status="success",
                    embeddings_generated=embeddings_generated if bge_m3_used else {},
                    processing_time=processing_time if bge_m3_used else None,
                    cache_hits=cache_hits if bge_m3_used else 0,
                    bge_m3_used=bge_m3_used,
                    embedding_types=list(embeddings_generated.keys()) if bge_m3_used else []
                )

                # Aktualisiere Gesamtstatistiken
                if bge_m3_used:
                    for key, value in embeddings_generated.items():
                        total_embeddings_generated[key] += value
                    total_cache_hits += cache_hits

                logger.info(
                    f"Successfully ingested {file.filename} with {len(hybrid_chunks)} extracted elements"
                    f"{' using BGE-M3' if bge_m3_used else ''}"
                )
                
                return result

            except Exception as e:
                logger.error(f"Error ingesting PDF {file.filename}: {e}")
                processing_time = asyncio.get_event_loop().time() - file_start_time

                # Create error result
                result = IngestResult(
                    filename=file.filename,
                    status="error",
                    error=str(e),
                    embeddings_generated={},
                    processing_time=processing_time if bge_m3_used else None,
                    cache_hits=0,
                    bge_m3_used=bge_m3_used,
                    embedding_types=[]
                )

                return result

            finally:
                # Clean up temporary files if they exist
                try:
                    if "temp_pdf_path" in locals():
                        temp_pdf_path.unlink()
                except:
                    pass

    # Verarbeite alle Dateien parallel
    tasks = [process_single_file(pdf_file) for pdf_file in pdf_files]
    file_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Verarbeite Ergebnisse
    for i, result in enumerate(file_results):
        if isinstance(result, Exception):
            logger.error(f"Error processing file {pdf_files[i].filename}: {result}")
            error_result = IngestResult(
                filename=pdf_files[i].filename,
                status="error",
                error=str(result),
                embeddings_generated={},
                processing_time=None,
                cache_hits=0,
                bge_m3_used=bge_m3_used,
                embedding_types=[]
            )
            results.append(error_result)
        else:
            results.append(result)

    # Berechne Gesamtstatistiken
    batch_processing_time = asyncio.get_event_loop().time() - batch_start_time
    successful_files = sum(1 for r in results if r.status == "success")
    
    logger.info(f"Batch ingestion completed: {successful_files}/{len(pdf_files)} files successful")
    logger.info(f"Total BGE-M3 embeddings generated: {total_embeddings_generated}")
    logger.info(f"Total cache hits: {total_cache_hits}")
    logger.info(f"Total batch processing time: {batch_processing_time:.2f}s")

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
                must=[
                    http_models.FieldCondition(
                        key="session_id", match=http_models.MatchValue(value=session_id)
                    )
                ]
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

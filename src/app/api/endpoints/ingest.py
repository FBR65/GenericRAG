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
)
from src.app.models.schemas import IngestResult, IngestResponse, ExtractionResult
from src.app.services import PDFExtractor, TextPreprocessor
from src.app.services.image_embedding_service import ImageEmbeddingService
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


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    qdrant_client: QdrantClientDep,
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
        image_storage: Image storage service
        settings: Application settings

    Returns:
        Ingest response with results
    """
    if not file.filename.lower().endswith(".pdf"):
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

        # Initialize text preprocessor with sparse configuration
        text_preprocessor = TextPreprocessor(
            sparse_max_features=settings.qdrant.sparse_max_features
        )

        # Create semantic chunks from extracted elements
        logger.info("Creating semantic chunks from extracted elements")
        chunks = text_preprocessor.create_chunks(extraction_result)

        # Generate embeddings for each chunk
        logger.info("Generating embeddings for chunks")
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
            json.dump(extraction_result.dict(), f, ensure_ascii=False, indent=2)

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
        result = IngestResult(
            filename=file.filename,
            num_pages=extraction_result.total_pages,
            status="success",
        )

        results.append(result)
        logger.info(
            f"Successfully ingested {file.filename} with {len(hybrid_chunks)} extracted elements"
        )

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
    session_id: Optional[str] = None,
    files: List[UploadFile] = File(...),
) -> IngestResponse:
    """
    Ingest multiple PDF files in batch

    Args:
        files: List of PDF files to ingest
        session_id: Optional session identifier
        qdrant_client: Qdrant client
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
            temp_pdf_path = (
                Path(settings.storage.temp_storage_path) / f"{uuid.uuid4()}.pdf"
            )
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
            images_for_embedding = extractor.get_images_for_embedding(
                str(temp_pdf_path)
            )

            # Initialize image embedding service
            image_embedding_service = ImageEmbeddingService()

            # Process images for embedding generation
            image_embeddings = []
            for image_info in images_for_embedding:
                try:
                    logger.info(
                        f"Processing image for embedding: {image_info['filename']}"
                    )

                    # Generate CLIP embedding
                    embedding = (
                        image_embedding_service.generate_clip_embedding_from_path(
                            image_info["file_path"]
                        )
                    )

                    if embedding and image_embedding_service.validate_embedding(
                        embedding
                    ):
                        # Create image metadata
                        image_metadata = image_embedding_service.create_image_metadata(
                            image_path=image_info["file_path"],
                            page_number=image_info["page_number"],
                            document_id=Path(file.filename).stem,
                            chunk_id=f"page_{image_info['page_number']}_img_{len(image_embeddings)}",
                        )
                        image_metadata["created_at"] = str(
                            asyncio.get_event_loop().time()
                        )

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
                    logger.error(
                        f"Error processing image {image_info['filename']}: {e}"
                    )
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

            # Initialize text preprocessor with sparse configuration
            text_preprocessor = TextPreprocessor(
                sparse_max_features=settings.qdrant.sparse_max_features
            )

            # Create semantic chunks from extracted elements
            logger.info("Creating semantic chunks from extracted elements")
            chunks = text_preprocessor.create_chunks(extraction_result)

            # Generate embeddings for each chunk
            logger.info("Generating embeddings for chunks")
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
                json.dump(extraction_result.dict(), f, ensure_ascii=False, indent=2)

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
                logger.info(
                    f"Upserting {len(image_embeddings)} image embeddings to Qdrant"
                )
                await upsert_image_embeddings_with_retry(
                    qdrant_client=qdrant_client,
                    collection_name=settings.qdrant.image_collection_name,
                    embeddings=image_embeddings,
                    batch_size=50,
                )
                logger.info(
                    f"Successfully stored {len(image_embeddings)} image embeddings"
                )

            # Clean up temporary PDF
            temp_pdf_path.unlink()

            # Create success result
            result = IngestResult(
                filename=file.filename,
                num_pages=extraction_result.total_pages,
                status="success",
            )

            results.append(result)
            logger.info(
                f"Successfully ingested {file.filename} with {len(hybrid_chunks)} extracted elements"
            )

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
                if "temp_pdf_path" in locals():
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

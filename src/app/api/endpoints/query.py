"""
Query endpoints for the RAG system
"""

import asyncio
import json
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from PIL import Image

from src.app.api.dependencies import (
    ImageStorageDep,
    QdrantClientDep,
    SettingsDep,
)
from src.app.models.schemas import QueryResponse, SearchResult, VLMResponse
from src.app.services.vlm_service import VLMService
from src.app.utils.qdrant_utils import (
    search_with_retry,
    create_payload_filter,
    hybrid_search_with_metadata,
    search_images_with_text_context,
    combine_and_rank_results,
)
from src.app.services.search_service import SearchService

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    query: str,
    qdrant_client: QdrantClientDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    top_k: int = Query(5, ge=1, le=20),
    score_threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0),
    session_id: Optional[str] = None,
    search_strategy: str = Query("hybrid", regex="^(text_only|image_only|hybrid)$"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    metadata_filters: Optional[Dict[str, Any]] = None,
    include_images: bool = Query(True),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
) -> QueryResponse:
    """
    Query the RAG system with hybrid search support

    Args:
        query: User query
        session_id: Optional session identifier
        top_k: Number of results to return
        score_threshold: Minimum score threshold
        search_strategy: Search strategy ("text_only", "image_only", "hybrid")
        alpha: Weight for dense vs sparse search (0.0-1.0)
        metadata_filters: Additional metadata filters (bbox, page_number, type)
        include_images: Whether to include image results
        page: Page number for pagination
        page_size: Number of results per page
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings

    Returns:
        Query response with results
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Initialize search service
        search_service = SearchService(qdrant_client, image_storage, settings)

        # Generate query embeddings (placeholder - would use embedding model in real implementation)
        logger.info(f"Processing query: {query}")
        # TODO: Replace with actual embedding generation
        query_embedding = [0.0] * 768  # Placeholder embedding
        sparse_vector = {}  # Placeholder sparse vector

        # Perform hybrid search
        logger.info(f"Performing {search_strategy} search for query: {query}")
        search_results = await search_service.perform_hybrid_search(
            query=query,
            query_embedding=query_embedding,
            sparse_vector=sparse_vector,
            search_strategy=search_strategy,
            alpha=alpha,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filters=metadata_filters,
            include_images=include_images,
            session_id=session_id,
            page=page,
            page_size=page_size,
        )

        # Generate response using VLM
        logger.info("Generating response using VLM")
        vlm_service = VLMService()

        import time

        start_time = time.time()

        try:
            response_text = await vlm_service.generate_response_with_vlm(
                query=query,
                search_results=search_results,
                use_images=True,
                max_context_length=4000,
            )

            processing_time = time.time() - start_time

            # Create VLM response info
            vlm_info = VLMResponse(
                model_used="gemma3:latest",
                context_length=4000,
                processing_time=processing_time,
                images_used=True,
                sources_referenced=[result.id for result in search_results[:5]],
            )

        except Exception as e:
            logger.error(f"Error generating VLM response: {e}")
            # Fallback response
            response_text = f"⚠️ I encountered an error while processing your request with the Vision Language Model. Please try again or rephrase your question. Error: {str(e)}"

            # Create error VLM response info
            vlm_info = VLMResponse(
                model_used="gemma3:latest",
                context_length=4000,
                processing_time=time.time() - start_time,
                images_used=False,
                sources_referenced=[result.id for result in search_results[:5]],
            )

        return QueryResponse(
            query=query,
            session_id=session_id,
            results=search_results,
            response=response_text,
            total_results=len(search_results),
            search_strategy=search_strategy,
            metadata_filters=metadata_filters,
            page=page,
            page_size=page_size,
            vlm_info=vlm_info,
            response_type="vlm",
        )

    except Exception as e:
        logger.error(f"Error querying RAG system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-stream")
async def query_rag_stream(
    query: str,
    qdrant_client: QdrantClientDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    top_k: int = Query(5, ge=1, le=20),
    score_threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0),
    session_id: Optional[str] = None,
    search_strategy: str = Query("hybrid", regex="^(text_only|image_only|hybrid)$"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    metadata_filters: Optional[Dict[str, Any]] = None,
    include_images: bool = Query(True),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
):
    """
    Query the RAG system with streaming response and hybrid search support

    Args:
        query: User query
        session_id: Optional session identifier
        top_k: Number of results to return
        score_threshold: Minimum score threshold
        search_strategy: Search strategy ("text_only", "image_only", "hybrid")
        alpha: Weight for dense vs sparse search (0.0-1.0)
        metadata_filters: Additional metadata filters (bbox, page_number, type)
        include_images: Whether to include image results
        page: Page number for pagination
        page_size: Number of results per page
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings

    Yields:
        Server-sent events with streaming response
    """
    if not query.strip():
        yield f"data: {json.dumps({'error': 'Query cannot be empty'})}\n\n"
        return

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Initialize search service
        search_service = SearchService(qdrant_client, image_storage, settings)

        # Generate query embeddings (placeholder - would use embedding model in real implementation)
        yield 'data: {"status": "processing", "message": "Processing query..."}\n\n'

        # TODO: Replace with actual embedding generation
        query_embedding = [0.0] * 768  # Placeholder embedding
        sparse_vector = {}  # Placeholder sparse vector

        # Perform hybrid search
        yield 'data: {"status": "searching", "message": "Performing hybrid search..."}\n\n'

        search_results = await search_service.perform_hybrid_search(
            query=query,
            query_embedding=query_embedding,
            sparse_vector=sparse_vector,
            search_strategy=search_strategy,
            alpha=alpha,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filters=metadata_filters,
            include_images=include_images,
            session_id=session_id,
            page=page,
            page_size=page_size,
        )

        # Generate response using VLM with streaming
        yield 'data: {"status": "generating", "message": "Generating response with VLM..."}\n\n'

        # Generate VLM response
        vlm_service = VLMService()
        import time

        start_time = time.time()

        try:
            response_text = await vlm_service.generate_response_with_vlm(
                query=query,
                search_results=search_results,
                use_images=True,
                max_context_length=4000,
            )

            processing_time = time.time() - start_time

            # Add processing info to response
            response_text = f"🤖 **VLM Response** (Processing time: {processing_time:.2f}s)\n\n{response_text}"

        except Exception as e:
            logger.error(f"Error generating VLM response: {e}")
            # Fallback response
            response_text = f"⚠️ I encountered an error while processing your request with the Vision Language Model. Please try again or rephrase your question. Error: {str(e)}"

        # Stream the response in chunks
        chunks = _split_response_into_chunks(response_text)
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "type": "text",
                "content": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

        # Send final response
        final_response = {
            "status": "completed",
            "query": query,
            "session_id": session_id,
            "total_results": len(search_results),
            "search_strategy": search_strategy,
            "metadata_filters": metadata_filters,
            "page": page,
            "page_size": page_size,
        }

        yield f"data: {json.dumps(final_response)}\n\n"

    except Exception as e:
        logger.error(f"Error streaming query: {e}")
        error_response = {
            "status": "error",
            "query": query,
            "error": str(e),
            "session_id": session_id,
        }
        yield f"data: {json.dumps(error_response)}\n\n"


@router.get("/sessions/{session_id}/results")
async def get_session_results(
    qdrant_client: QdrantClientDep,
    settings: SettingsDep,
    session_id: str,
) -> List[SearchResult]:
    """
    Get all search results for a session

    Args:
        session_id: Session identifier
        qdrant_client: Qdrant client
        settings: Application settings

    Returns:
        List of search results
    """
    try:
        # Search for all results in the session
        response = await search_with_retry(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.collection_name,
            query_vector=[0.0] * 768,  # Dummy vector for all results
            limit=1000,
            query_filter=create_payload_filter(session_id=session_id),
        )

        # Process results
        search_results = []
        for point in response.points:
            payload = point.payload or {}

            # Create search result without loading image
            search_result = SearchResult(
                id=point.id,
                score=point.score,
                document=payload.get("document", ""),
                page=payload.get("page", 0),
                image=None,  # Don't load images for this endpoint
                metadata={
                    "session_id": payload.get("session_id"),
                    "created_at": payload.get("created_at"),
                    "image_path": payload.get("image_path"),
                },
            )

            search_results.append(search_result)

        # Sort by score (descending)
        search_results.sort(key=lambda x: x.score, reverse=True)

        return search_results

    except Exception as e:
        logger.error(f"Error getting session results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_response_with_dspy(
    query: str,
    search_results: List[SearchResult],
    settings: SettingsDep,
) -> str:
    """
    Generate response using DSPy/GEPA

    Args:
        query: User query
        search_results: Search results
        settings: Application settings

    Returns:
        Generated response text
    """
    try:
        # Get DSPy service from app state
        from src.app.services.dspy_integration import DSPyIntegrationService

        # This would be called from the main app context
        # For now, return a simple response
        if not search_results:
            return "No relevant documents found for your query."

        # Format context from search results
        context = []
        for result in search_results[:3]:  # Use top 3 results
            context.append(
                f"Document: {result.document}, Page: {result.page}, Score: {result.score:.3f}"
            )

        context_text = "\n".join(context)

        # Simple response generation (would be replaced with DSPy)
        response = f"""Based on the search results for your query "{query}":

{context_text}

This is a placeholder response. In the full implementation, this would be generated using DSPy/GEPA optimization with the google/gemma-3-27b-it model."""

        return response

    except Exception as e:
        logger.error(f"Error generating response with DSPy: {e}")
        return f"Error generating response: {str(e)}"


async def _generate_response_with_dspy_stream(
    query: str,
    search_results: List[SearchResult],
    settings: SettingsDep,
):
    """
    Generate response using DSPy/GEPA with streaming

    Args:
        query: User query
        search_results: Search results
        settings: Application settings

    Yields:
        Response chunks
    """
    try:
        # Get DSPy service from app state
        from src.app.services.dspy_integration import DSPyIntegrationService

        # This would be called from the main app context
        # For now, return a simple streaming response
        if not search_results:
            yield {
                "type": "text",
                "content": "No relevant documents found for your query.",
            }
            return

        # Format context from search results
        context = []
        for result in search_results[:3]:  # Use top 3 results
            context.append(
                f"Document: {result.document}, Page: {result.page}, Score: {result.score:.3f}"
            )

        context_text = "\n".join(context)

        # Simple streaming response (would be replaced with DSPy)
        yield {
            "type": "text",
            "content": f'Based on the search results for your query "{query}":\n\n',
        }
        yield {"type": "text", "content": context_text}
        yield {
            "type": "text",
            "content": "\n\nThis is a placeholder response. In the full implementation, this would be generated using DSPy/GEPA optimization with the google/gemma-3-27b-it model.",
        }

    except Exception as e:
        logger.error(f"Error generating streaming response with DSPy: {e}")
        yield {"type": "error", "content": f"Error generating response: {str(e)}"}


def _split_response_into_chunks(response_text: str, chunk_size: int = 200) -> List[str]:
    """
    Split response text into chunks for streaming

    Args:
        response_text: Response text to split
        chunk_size: Maximum size of each chunk

    Returns:
        List of text chunks
    """
    try:
        # Split by sentences first for better readability
        sentences = response_text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        # Add remaining content
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    except Exception as e:
        logger.error(f"Error splitting response into chunks: {e}")
        # Fallback to simple splitting
        return [
            response_text[i : i + chunk_size]
            for i in range(0, len(response_text), chunk_size)
        ]

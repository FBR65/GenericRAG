"""
Query endpoints for the RAG system
"""
import asyncio
import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from PIL import Image

from app.api.dependencies import (
    ColPaliModelDep,
    ColPaliProcessorDep,
    ImageStorageDep,
    QdrantClientDep,
    SettingsDep,
)
from app.models.schemas import QueryResponse, SearchResult
from app.utils.colpali_utils import process_query
from app.utils.qdrant_utils import search_with_retry, create_payload_filter

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    query: str,
    session_id: Optional[str] = None,
    top_k: int = Query(5, ge=1, le=20),
    score_threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0),
    qdrant_client: QdrantClientDep = Depends(),
    colpali_model: ColPaliModelDep = Depends(),
    colpali_processor: ColPaliProcessorDep = Depends(),
    image_storage: ImageStorageDep = Depends(),
    settings: SettingsDep = Depends(),
) -> QueryResponse:
    """
    Query the RAG system
    
    Args:
        query: User query
        session_id: Optional session identifier
        top_k: Number of results to return
        score_threshold: Minimum score threshold
        qdrant_client: Qdrant client
        colpali_model: ColPali model
        colpali_processor: ColPali processor
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
        # Generate query embedding using ColPali
        logger.info(f"Generating query embedding for: {query}")
        query_embedding = process_query(
            model=colpali_model,
            processor=colpali_processor,
            query=query,
        )
        
        # Search in Qdrant
        logger.info(f"Searching for top {top_k} results")
        response = await search_with_retry(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=create_payload_filter(session_id=session_id),
            score_threshold=score_threshold,
        )
        
        # Process results
        search_results = []
        for point in response.points:
            payload = point.payload or {}
            
            # Load image from storage
            image_path = payload.get("image_path")
            image = None
            if image_path:
                try:
                    images = await image_storage.load_images([image_path])
                    if images:
                        image = images[0]
                except Exception as e:
                    logger.warning(f"Could not load image {image_path}: {e}")
            
            # Create search result
            search_result = SearchResult(
                id=point.id,
                score=point.score,
                document=payload.get("document", ""),
                page=payload.get("page", 0),
                image=image,
                metadata={
                    "session_id": payload.get("session_id"),
                    "created_at": payload.get("created_at"),
                },
            )
            
            search_results.append(search_result)
        
        # Sort by score (descending)
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        # Generate response using DSPy/GEPA
        logger.info("Generating response using DSPy/GEPA")
        response_text = await _generate_response_with_dspy(
            query=query,
            search_results=search_results,
            settings=settings,
        )
        
        return QueryResponse(
            query=query,
            session_id=session_id,
            results=search_results,
            response=response_text,
            total_results=len(search_results),
        )
        
    except Exception as e:
        logger.error(f"Error querying RAG system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-stream")
async def query_rag_stream(
    query: str,
    session_id: Optional[str] = None,
    top_k: int = Query(5, ge=1, le=20),
    score_threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0),
    qdrant_client: QdrantClientDep = Depends(),
    colpali_model: ColPaliModelDep = Depends(),
    colpali_processor: ColPaliProcessorDep = Depends(),
    image_storage: ImageStorageDep = Depends(),
    settings: SettingsDep = Depends(),
):
    """
    Query the RAG system with streaming response
    
    Args:
        query: User query
        session_id: Optional session identifier
        top_k: Number of results to return
        score_threshold: Minimum score threshold
        qdrant_client: Qdrant client
        colpali_model: ColPali model
        colpali_processor: ColPali processor
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
        # Generate query embedding using ColPali
        yield "data: {\"status\": \"processing\", \"message\": \"Processing query...\"}\n\n"
        
        query_embedding = process_query(
            model=colpali_model,
            processor=colpali_processor,
            query=query,
        )
        
        # Search in Qdrant
        yield "data: {\"status\": \"searching\", \"message\": \"Searching documents...\"}\n\n"
        
        response = await search_with_retry(
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=create_payload_filter(session_id=session_id),
            score_threshold=score_threshold,
        )
        
        # Process results
        search_results = []
        for point in response.points:
            payload = point.payload or {}
            
            # Load image from storage
            image_path = payload.get("image_path")
            image = None
            if image_path:
                try:
                    images = await image_storage.load_images([image_path])
                    if images:
                        image = images[0]
                except Exception as e:
                    logger.warning(f"Could not load image {image_path}: {e}")
            
            # Create search result
            search_result = SearchResult(
                id=point.id,
                score=point.score,
                document=payload.get("document", ""),
                page=payload.get("page", 0),
                image=image,
                metadata={
                    "session_id": payload.get("session_id"),
                    "created_at": payload.get("created_at"),
                },
            )
            
            search_results.append(search_result)
        
        # Sort by score (descending)
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        # Generate response using DSPy/GEPA with streaming
        yield "data: {\"status\": \"generating\", \"message\": \"Generating response...\"}\n\n"
        
        async for chunk in _generate_response_with_dspy_stream(
            query=query,
            search_results=search_results,
            settings=settings,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send final response
        final_response = {
            "status": "completed",
            "query": query,
            "session_id": session_id,
            "total_results": len(search_results),
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
    session_id: str,
    qdrant_client: QdrantClientDep = Depends(),
    settings: SettingsDep = Depends(),
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
        from app.services.dspy_integration import DSPyIntegrationService
        
        # This would be called from the main app context
        # For now, return a simple response
        if not search_results:
            return "No relevant documents found for your query."
        
        # Format context from search results
        context = []
        for result in search_results[:3]:  # Use top 3 results
            context.append(f"Document: {result.document}, Page: {result.page}, Score: {result.score:.3f}")
        
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
        from app.services.dspy_integration import DSPyIntegrationService
        
        # This would be called from the main app context
        # For now, return a simple streaming response
        if not search_results:
            yield {"type": "text", "content": "No relevant documents found for your query."}
            return
        
        # Format context from search results
        context = []
        for result in search_results[:3]:  # Use top 3 results
            context.append(f"Document: {result.document}, Page: {result.page}, Score: {result.score:.3f}")
        
        context_text = "\n".join(context)
        
        # Simple streaming response (would be replaced with DSPy)
        yield {"type": "text", "content": f"Based on the search results for your query \"{query}\":\n\n"}
        yield {"type": "text", "content": context_text}
        yield {"type": "text", "content": "\n\nThis is a placeholder response. In the full implementation, this would be generated using DSPy/GEPA optimization with the google/gemma-3-27b-it model."}
        
    except Exception as e:
        logger.error(f"Error generating streaming response with DSPy: {e}")
        yield {"type": "error", "content": f"Error generating response: {str(e)}"}
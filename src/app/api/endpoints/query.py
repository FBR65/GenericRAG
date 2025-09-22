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
    SearchServiceDep,
)
from src.app.models.schemas import (
    QueryResponse,
    SearchResult,
    VLMResponse,
    BGE_M3_QueryRequest,
    BGE_M3_QueryResponse,
    BGE_M3_SearchResult,
    BGE_M3_SearchResultItem,
    BGE_M3_SearchMode,
    BGE_M3_MultivectorStrategy
)
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


@router.post("/query-bge-m3", response_model=BGE_M3_QueryResponse)
async def query_bge_m3(
    request: BGE_M3_QueryRequest,
    qdrant_client: QdrantClientDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    search_service: SearchServiceDep,
) -> BGE_M3_QueryResponse:
    """
    BGE-M3 specific query endpoint with advanced search capabilities
    
    This endpoint provides specialized search functionality using BGE-M3's
    all-in-one embedding model with support for dense, sparse, multivector,
    and hybrid search modes.

    Args:
        request: BGE-M3 specific query parameters
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings

    Returns:
        BGE-M3 specific query response with detailed search results
    """
    import time
    start_time = time.time()
    
    try:
        # Initialize search service
        search_service = SearchService(qdrant_client, image_storage, settings)
        
        if not search_service.bge_m3_service:
            raise HTTPException(
                status_code=503,
                detail="BGE-M3 service is not available"
            )
        
        logger.info(f"Processing BGE-M3 query: {request.query} with mode: {request.search_mode}")
        
        # Generiere BGE-M3 Embeddings
        embeddings_start = time.time()
        bge_m3_embeddings = await search_service.get_bge_m3_embeddings(request.query)
        embeddings_time = time.time() - embeddings_start
        
        # Validiere Embeddings
        if not bge_m3_embeddings or not any(bge_m3_embeddings.values()):
            raise HTTPException(
                status_code=400,
                detail="Failed to generate BGE-M3 embeddings"
            )
        
        # Führe BGE-M3 Suche durch
        search_start = time.time()
        
        if request.search_mode == BGE_M3_SearchMode.HYBRID:
            # Hybride Suche mit allen drei Modi
            search_results = await search_service.bge_m3_hybrid_search(
                query=request.query,
                search_strategy="hybrid",
                alpha=request.alpha,
                beta=request.beta,
                gamma=request.gamma,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filters=request.metadata_filters,
                include_images=request.include_images,
                session_id=request.session_id,
                page=request.page,
                page_size=request.page_size,
            )
        elif request.search_mode == BGE_M3_SearchMode.DENSE:
            # Nur Dense Suche
            search_results = await search_service.bge_m3_dense_search(
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filters=request.metadata_filters,
                session_id=request.session_id,
                page=request.page,
                page_size=request.page_size,
            )
        elif request.search_mode == BGE_M3_SearchMode.SPARSE:
            # Nur Sparse Suche
            search_results = await search_service.bge_m3_sparse_search(
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filters=request.metadata_filters,
                session_id=request.session_id,
                page=request.page,
                page_size=request.page_size,
            )
        elif request.search_mode == BGE_M3_SearchMode.MULTIVECTOR:
            # Nur Multi-Vector Suche
            search_results = await search_service.bge_m3_multivector_search(
                query=request.query,
                strategy=request.multivector_strategy,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filters=request.metadata_filters,
                session_id=request.session_id,
                page=request.page,
                page_size=request.page_size,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported search mode: {request.search_mode}"
            )
        
        search_time = time.time() - search_start
        total_time = time.time() - start_time
        
        # Formatiere BGE-M3 spezifische Ergebnisse
        bge_m3_results = BGE_M3_SearchResult(
            items=[
                BGE_M3_SearchResultItem(
                    id=result.id,
                    score=result.score,
                    document=result.document,
                    page=result.page,
                    image=result.image,
                    metadata=result.metadata,
                    search_type=result.metadata.get("search_type", "bge_m3"),
                    vector_types=bge_m3_embeddings.get("vector_types", []),
                    confidence=result.metadata.get("confidence", "low"),
                    bge_m3_metadata=result.metadata.get("bge_m3_metadata", {})
                ) for result in search_results
            ],
            total=len(search_results),
            query=request.query,
            search_mode=request.search_mode,
            embedding_info={
                "dense_vector_length": len(bge_m3_embeddings.get("dense", [])),
                "sparse_vector_length": len(bge_m3_embeddings.get("sparse", {})),
                "multivector_count": len(bge_m3_embeddings.get("multivector", [])),
                "vector_types": list(bge_m3_embeddings.keys()),
                "cache_hit": bge_m3_embeddings.get("cache_hit", False)
            },
            processing_time=total_time,
            cache_hit=bge_m3_embeddings.get("cache_hit", False)
        )
        
        # Generiere Antwort
        response_text = ""
        vlm_info = None
        
        if request.use_vlm:
            logger.info("Generating response using VLM with BGE-M3 results")
            vlm_service = VLMService()
            
            try:
                vlm_response = await vlm_service.generate_response_with_vlm(
                    query=request.query,
                    search_results=search_results,
                    use_images=request.use_images,
                    max_context_length=4000,
                )
                
                response_text = vlm_response.response
                vlm_info = vlm_response
                
            except Exception as e:
                logger.error(f"Error generating VLM response: {e}")
                response_text = f"⚠️ I encountered an error while processing your request with the Vision Language Model. Please try again or rephrase your question. Error: {str(e)}"
                
                vlm_info = VLMResponse(
                    response=response_text,
                    confidence_score=0.0,
                    processing_time=time.time() - start_time,
                    model_used="gemma3:latest",
                    context_length=4000,
                    images_used=False,
                    sources_referenced=[result.id for result in search_results[:5]],
                )
        else:
            # Generiere einfache Antwort ohne VLM
            if search_results:
                response_text = (
                    f"Found {len(search_results)} BGE-M3 results for your query '{request.query}' "
                    f"using {request.search_mode} mode."
                )
            else:
                response_text = f"No BGE-M3 results found for your query '{request.query}'."
        
        # Erstelle BGE-M3 Query Response
        response = BGE_M3_QueryResponse(
            query=request.query,
            session_id=request.session_id or str(uuid.uuid4()),
            results=bge_m3_results,
            response=response_text,
            total_results=len(search_results),
            search_mode=request.search_mode,
            metadata_filters=request.metadata_filters,
            page=request.page,
            page_size=request.page_size,
            vlm_info=vlm_info,
            response_type="vlm" if request.use_vlm else "simple",
            vlm_used=request.use_vlm,
            image_context_included=request.include_images,
            processing_time=total_time,
            cache_hit=bge_m3_embeddings.get("cache_hit", False),
            embedding_generation_time=embeddings_time,
            search_time=search_time
        )
        
        logger.info(f"BGE-M3 query completed in {total_time:.2f}s with {len(search_results)} results")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in BGE-M3 query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: dict,
    qdrant_client: QdrantClientDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    search_service: SearchServiceDep,
) -> QueryResponse:
    """
    Query the RAG system with hybrid search support and BGE-M3 integration

    Args:
        request: JSON request containing query and parameters
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings

    Returns:
        Query response with results including BGE-M3 support
    """
    # Extract parameters from request
    query = request.get("query", "")
    session_id = request.get("session_id", str(uuid.uuid4()))
    top_k = request.get("top_k", 5)
    score_threshold = request.get("score_threshold", 0.5)
    search_strategy = request.get("search_strategy", "hybrid")
    alpha = request.get("alpha", 0.5)
    metadata_filters = request.get("metadata_filters", None)
    include_images = request.get("include_images", True)
    page = request.get("page", 1)
    page_size = request.get("page_size", 10)
    use_vlm = request.get("use_vlm", False)
    use_images = request.get("use_images", True)
    
    # BGE-M3 spezifische Parameter
    use_bge_m3 = request.get("use_bge_m3", False)
    search_mode = request.get("search_mode", "hybrid")
    beta = request.get("beta", 0.3)
    gamma = request.get("gamma", 0.2)
    multivector_strategy = request.get("multivector_strategy", "max_sim")

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Initialize search service
        search_service = SearchService(qdrant_client, image_storage, settings)

        # Generate query embeddings
        logger.info(f"Processing query: {query}")
        query_embedding = [0.0] * 768  # Placeholder embedding
        sparse_vector = {}  # Placeholder sparse vector
        bge_m3_embeddings = None
        bge_m3_used = False
        
        # BGE-M3 Unterstützung
        if use_bge_m3 and search_service.bge_m3_service:
            logger.info(f"Using BGE-M3 for query processing with mode: {search_mode}")
            try:
                # Generiere BGE-M3 Embeddings
                bge_m3_embeddings = await search_service.get_bge_m3_embeddings(query)
                query_embedding = bge_m3_embeddings.get("dense", [0.0] * 768)
                sparse_vector = bge_m3_embeddings.get("sparse", {})
                bge_m3_used = True
                
                logger.info(f"BGE-M3 embeddings generated: dense={len(query_embedding)}, "
                           f"sparse={len(sparse_vector)}, multivector={len(bge_m3_embeddings.get('multivector', []))}")
            except Exception as e:
                logger.error(f"Error generating BGE-M3 embeddings: {e}")
                # Fallback zur ursprünglichen Methode
                bge_m3_used = False

        # Perform hybrid search
        logger.info(f"Performing {search_strategy} search for query: {query}")
        
        if bge_m3_used and bge_m3_embeddings:
            # Verwende BGE-M3 erweiterte Suche
            search_results = await search_service.bge_m3_hybrid_search(
                query=query,
                search_strategy=search_mode,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filters=metadata_filters,
                include_images=include_images,
                session_id=session_id,
                page=page,
                page_size=page_size,
            )
        else:
            # Verwende Standard hybride Suche
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

        # Generate response using VLM if requested
        response_text = ""
        vlm_info = None

        if use_vlm:
            logger.info("Generating response using VLM")
            vlm_service = VLMService()

            import time

            start_time = time.time()

            try:
                vlm_response = await vlm_service.generate_response_with_vlm(
                    query=query,
                    search_results=search_results,
                    use_images=use_images,
                    max_context_length=4000,
                )

                processing_time = time.time() - start_time
                response_text = vlm_response.response

                # Use the VLM response from the service
                vlm_info = vlm_response

            except Exception as e:
                logger.error(f"Error generating VLM response: {e}")
                # Fallback response
                response_text = f"⚠️ I encountered an error while processing your request with the Vision Language Model. Please try again or rephrase your question. Error: {str(e)}"

                # Create error VLM response info
                vlm_info = VLMResponse(
                    response=response_text,
                    confidence_score=0.0,
                    processing_time=time.time() - start_time,
                    model_used="gemma3:latest",
                    context_length=4000,
                    images_used=False,
                    sources_referenced=[result.id for result in search_results[:5]],
                )
        else:
            # Generate simple response without VLM
            if search_results:
                response_text = (
                    f"Found {len(search_results)} results for your query '{query}'."
                )
            else:
                response_text = f"No results found for your query '{query}'."

        # Erweitere QueryResponse mit BGE-M3 spezifischen Informationen
        response = QueryResponse(
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
            response_type="vlm" if use_vlm else "simple",
            vlm_used=use_vlm,
            image_context_included=use_images,
            bge_m3_used=bge_m3_used,
            bge_m3_search_mode=search_mode if bge_m3_used else None,
            embedding_info={
                "dense_vector_length": len(query_embedding),
                "sparse_vector_length": len(sparse_vector),
                "multivector_available": bge_m3_embeddings and len(bge_m3_embeddings.get("multivector", [])) > 0 if bge_m3_used else False,
                "cache_hit": bge_m3_embeddings.get("cache_hit", False) if bge_m3_embeddings else False
            } if bge_m3_used else None
        )
        
        logger.info(f"Query completed successfully with BGE-M3: {bge_m3_used}")
        return response

    except Exception as e:
        logger.error(f"Error querying RAG system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-stream")
async def query_rag_stream(
    request: dict,
    qdrant_client: QdrantClientDep,
    image_storage: ImageStorageDep,
    settings: SettingsDep,
    search_service: SearchServiceDep,
):
    """
    Query the RAG system with streaming response and hybrid search support with BGE-M3 integration

    Args:
        request: JSON request containing query and parameters
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings

    Yields:
        Server-sent events with streaming response including BGE-M3 support
    """
    # Extract parameters from request
    query = request.get("query", "")
    session_id = request.get("session_id", str(uuid.uuid4()))
    top_k = request.get("top_k", 5)
    score_threshold = request.get("score_threshold", 0.5)
    search_strategy = request.get("search_strategy", "hybrid")
    alpha = request.get("alpha", 0.5)
    metadata_filters = request.get("metadata_filters", None)
    include_images = request.get("include_images", True)
    page = request.get("page", 1)
    page_size = request.get("page_size", 10)
    use_vlm = request.get("use_vlm", False)
    use_images = request.get("use_images", True)
    
    # BGE-M3 spezifische Parameter
    use_bge_m3 = request.get("use_bge_m3", False)
    search_mode = request.get("search_mode", "hybrid")
    beta = request.get("beta", 0.3)
    gamma = request.get("gamma", 0.2)
    multivector_strategy = request.get("multivector_strategy", "max_sim")

    if not query.strip():
        yield f"data: {json.dumps({'error': 'Query cannot be empty'})}\n\n"
        return

    try:
        # Initialize search service
        search_service = SearchService(qdrant_client, image_storage, settings)

        # Generate query embeddings
        yield 'data: {"status": "processing", "message": "Processing query..."}\n\n'
        
        query_embedding = [0.0] * 768  # Placeholder embedding
        sparse_vector = {}  # Placeholder sparse vector
        bge_m3_embeddings = None
        bge_m3_used = False
        
        # BGE-M3 Unterstützung
        if use_bge_m3 and search_service.bge_m3_service:
            yield 'data: {"status": "bge_m3", "message": "Generating BGE-M3 embeddings..."}\n\n'
            try:
                # Generiere BGE-M3 Embeddings
                bge_m3_embeddings = await search_service.get_bge_m3_embeddings(query)
                query_embedding = bge_m3_embeddings.get("dense", [0.0] * 768)
                sparse_vector = bge_m3_embeddings.get("sparse", {})
                bge_m3_used = True
                
                yield 'data: {"status": "bge_m3_complete", "message": "BGE-M3 embeddings generated successfully"}\n\n'
            except Exception as e:
                logger.error(f"Error generating BGE-M3 embeddings: {e}")
                # Fallback zur ursprünglichen Methode
                bge_m3_used = False
                yield 'data: {"status": "fallback", "message": "Falling back to standard embeddings"}\n\n'

        # Perform hybrid search
        yield 'data: {"status": "searching", "message": "Performing hybrid search..."}\n\n'
        
        if bge_m3_used and bge_m3_embeddings:
            # Verwende BGE-M3 erweiterte Suche
            search_results = await search_service.bge_m3_hybrid_search(
                query=query,
                search_strategy=search_mode,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filters=metadata_filters,
                include_images=include_images,
                session_id=session_id,
                page=page,
                page_size=page_size,
            )
        else:
            # Verwende Standard hybride Suche
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

        # Generate response using VLM with streaming if requested
        response_text = ""
        if use_vlm:
            yield 'data: {"status": "generating", "message": "Generating response with VLM..."}\n\n'

            # Generate VLM response
            vlm_service = VLMService()
            import time

            start_time = time.time()

            try:
                response_text = await vlm_service.generate_response_with_vlm(
                    query=query,
                    search_results=search_results,
                    use_images=use_images,
                    max_context_length=4000,
                )

                processing_time = time.time() - start_time

                # Add processing info to response
                response_text = f"🤖 **VLM Response** (Processing time: {processing_time:.2f}s)\n\n{response_text}"

            except Exception as e:
                logger.error(f"Error generating VLM response: {e}")
                # Fallback response
                response_text = f"⚠️ I encountered an error while processing your request with the Vision Language Model. Please try again or rephrase your question. Error: {str(e)}"
        else:
            # Generate simple response without VLM
            if search_results:
                response_text = (
                    f"Found {len(search_results)} results for your query '{query}'."
                )
            else:
                response_text = f"No results found for your query '{query}'."

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
            "vlm_used": use_vlm,
            "image_context_included": use_images,
            "bge_m3_used": bge_m3_used,
            "bge_m3_search_mode": search_mode if bge_m3_used else None,
            "embedding_info": {
                "dense_vector_length": len(query_embedding),
                "sparse_vector_length": len(sparse_vector),
                "multivector_available": bge_m3_embeddings and len(bge_m3_embeddings.get("multivector", [])) > 0 if bge_m3_used else False,
                "cache_hit": bge_m3_embeddings.get("cache_hit", False) if bge_m3_embeddings else False
            } if bge_m3_used else None
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

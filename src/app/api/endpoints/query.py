"""
Query endpoints for the RAG system
"""

import json
import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from src.app.api.dependencies import (
    QdrantClientDep,
    SettingsDep,
)
from src.app.models.schemas import (
    BGE_M3_QueryRequest,
    BGE_M3_QueryResponse,
    BGE_M3_SearchResult,
    BGE_M3_SearchResultItem,
    BGE_M3_SearchMode,
    BGE_M3_MultivectorStrategy
)

router = APIRouter()


@router.post("/query", response_model=BGE_M3_QueryResponse)
async def query_bge_m3(
    request: BGE_M3_QueryRequest,
    qdrant_client: QdrantClientDep,
    settings: SettingsDep,
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
        # Import SearchService directly
        from src.app.services.search_service import SearchService
        
        # Initialize search service
        search_service = SearchService(settings, qdrant_client=qdrant_client)
        
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
            search_results = await search_service.search_text(
                query=request.query,
                use_bge_m3=True,
                search_strategy="dense_only",
                alpha=1.0,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filters=request.metadata_filters,
                session_id=request.session_id,
                page=request.page,
                page_size=request.page_size,
            )
        elif request.search_mode == BGE_M3_SearchMode.SPARSE:
            # Nur Sparse Suche
            search_results = await search_service.search_text(
                query=request.query,
                use_bge_m3=True,
                search_strategy="sparse_only",
                alpha=0.0,
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
        
        # Generiere einfache Antwort
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
            response_type="simple",
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









"""
Qdrant utility functions with BGE-M3 support
"""

import asyncio
import time
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http import models as http_models
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from src.app.services.bge_m3_service import BGE_M3_Service

# Import BGE-M3 Service for enhanced functionality (lazy import to avoid circular dependency)
def get_bge_m3_service():
    """Lazy import of BGE-M3 Service to avoid circular dependencies"""
    from src.app.services.bge_m3_service import BGE_M3_Service
    return BGE_M3_Service


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def upsert_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    points: List[models.PointStruct],
    batch_size: int = 100,
) -> None:
    """
    Upsert points to Qdrant with retry logic and batching

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        points: List of points to upsert
        batch_size: Batch size for upserting
    """
    logger.info(f"Upserting {len(points)} points to collection '{collection_name}'")

    # Process in batches
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        batch_num = i // batch_size + 1

        try:
            await qdrant_client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            logger.debug(
                f"Upserted batch {batch_num}/{(len(points) + batch_size - 1) // batch_size}"
            )

        except Exception as e:
            logger.error(f"Error upserting batch {batch_num}: {e}")
            raise

    logger.info(f"Successfully upserted {len(points)} points")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def search_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    search_params: Optional[models.SearchParams] = None,
    score_threshold: Optional[float] = None,
) -> models.QueryResponse:
    """
    Search in Qdrant with retry logic

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        query_vector: Query vector
        limit: Number of results to return
        query_filter: Filter to apply
        search_params: Search parameters
        score_threshold: Minimum score threshold

    Returns:
        Query response
    """
    logger.debug(f"Searching in collection '{collection_name}' with limit {limit}")

    try:
        response = await qdrant_client.query_points(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=True,
            score_threshold=score_threshold,
        )

        logger.debug(f"Found {len(response.points)} results")

        # Apply score threshold filtering if specified
        if score_threshold is not None:
            filtered_points = [
                point
                for point in response.points
                if hasattr(point, "score") and point.score >= score_threshold
            ]
            logger.debug(
                f"Filtered to {len(filtered_points)} results with score >= {score_threshold}"
            )
            response.points = filtered_points

        return response

    except Exception as e:
        logger.error(f"Error searching in collection '{collection_name}': {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def hybrid_search_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    dense_vector: List[float],
    sparse_vector: Dict[str, float],
    alpha: float = 0.5,
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
    enable_multivector_reranking: bool = False,
    multivector_query: Optional[List[List[float]]] = None,
    multivector_weight: float = 0.3,
) -> models.QueryResponse:
    """
    Hybrid search in Qdrant with dense and sparse vectors
    Enhanced with BGE-M3 support including multi-vector reranking

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        dense_vector: Dense query vector
        sparse_vector: Sparse query vector as dict {index: value}
        alpha: Weight for dense vs sparse search (0.0 = pure sparse, 1.0 = pure dense)
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold
        enable_multivector_reranking: Enable multi-vector reranking for BGE-M3
        multivector_query: Optional multi-vector query for reranking
        multivector_weight: Weight for multivector reranking (0.0-1.0)

    Returns:
        Query response with optional multivector reranking
    """
    logger.debug(
        f"Performing hybrid search in collection '{collection_name}' with alpha={alpha}"
    )

    try:
        # Execute dense vector search
        dense_query = models.QueryRequest(
            query=models.Vector(
                name="dense",
                vector=dense_vector,
            ),
            limit=limit * 2,  # Get more results for reranking
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        dense_response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=dense_query.query,
            limit=limit * 2,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        # Execute sparse vector search
        sparse_response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                name="sparse",
                vector=sparse_vector,
            ),
            limit=limit * 2,  # Get more results for reranking
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        # Combine results using weighted scoring
        combined_results = {}

        # Process dense results
        for point in dense_response.points:
            if point.id not in combined_results:
                combined_results[point.id] = {
                    "id": point.id,
                    "score": 0.0,
                    "payload": point.payload,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "multivector_score": 0.0,
                }
            combined_results[point.id]["dense_score"] = point.score
            combined_results[point.id]["score"] += point.score * alpha

        # Process sparse results
        for point in sparse_response.points:
            if point.id not in combined_results:
                combined_results[point.id] = {
                    "id": point.id,
                    "score": 0.0,
                    "payload": point.payload,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "multivector_score": 0.0,
                }
            combined_results[point.id]["sparse_score"] = point.score
            combined_results[point.id]["score"] += point.score * (1 - alpha)

        # Apply multi-vector reranking if enabled
        if enable_multivector_reranking and multivector_query is not None:
            logger.debug("Applying multi-vector reranking")
            
            # Execute multivector search for reranking
            multivector_response = await qdrant_client.query_points(
                collection_name=collection_name,
                query=models.Vector(
                    name="multivector",
                    vector=multivector_query[0] if multivector_query else [],  # Use first vector as query
                ),
                limit=limit * 3,  # Get more candidates for reranking
                query_filter=query_filter,
                with_payload=True,
                score_threshold=score_threshold,
            )

            # Apply multivector reranking scores
            for point in multivector_response.points:
                if point.id in combined_results:
                    # Normalize multivector score (MAX_SIM returns values between 0 and 1)
                    multivector_score = point.score * multivector_weight
                    combined_results[point.id]["multivector_score"] = multivector_score
                    combined_results[point.id]["score"] += multivector_score

        # Sort by combined score and limit results
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["score"], reverse=True
        )[:limit]

        # Convert to QueryResponse format
        final_response = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=result["id"],
                    version=1,  # Add required version field
                    score=result["score"],
                    payload=result["payload"],
                    embedding=result.get("embedding", []),
                    metadata=result.get("metadata", {}),
                    document=result.get("document", "")
                )
                for result in sorted_results
            ]
        )

        logger.debug(f"Hybrid search found {len(final_response.points)} results")

        return final_response

    except Exception as e:
        logger.error(
            f"Error performing hybrid search in collection '{collection_name}': {e}"
        )
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def bge_m3_hybrid_search_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    dense_vector: List[float],
    sparse_vector: Dict[str, float],
    multivector_query: Optional[List[List[float]]] = None,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
) -> models.QueryResponse:
    """
    BGE-M3 specific hybrid search with three-phase approach:
    1. Pre-filtering with dense and sparse vectors
    2. Multi-vector reranking as final phase
    3. Optimized for BGE-M3 embedding formats

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        dense_vector: Dense query vector
        sparse_vector: Sparse query vector as dict {index: value}
        multivector_query: Optional multi-vector query for reranking
        alpha: Weight for dense search (0.0-1.0)
        beta: Weight for sparse search (0.0-1.0)
        gamma: Weight for multivector reranking (0.0-1.0)
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold

    Returns:
        Query response with three-phase BGE-M3 optimization
    """
    logger.debug(
        f"Performing BGE-M3 hybrid search in collection '{collection_name}' with "
        f"alpha={alpha}, beta={beta}, gamma={gamma}"
    )

    try:
        # Phase 1: Pre-filtering with dense and sparse vectors
        logger.debug("Phase 1: Pre-filtering with dense and sparse vectors")
        
        # Execute dense vector search with higher limit for pre-filtering
        dense_response = await qdrant_client.query_points(
            collection_name=collection_name,
            query_vector=dense_vector,
            limit=limit * 4,  # Get more candidates for pre-filtering
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        # Execute sparse vector search with higher limit for pre-filtering
        sparse_response = await qdrant_client.query_points(
            collection_name=collection_name,
            query_vector=sparse_vector,
            limit=limit * 4,  # Get more candidates for pre-filtering
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        # Combine pre-filtering results
        pre_filtered_results = {}
        
        # Process dense results
        for point in dense_response.points:
            if point.id not in pre_filtered_results:
                pre_filtered_results[point.id] = {
                    "id": point.id,
                    "score": 0.0,
                    "payload": point.payload,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "multivector_score": 0.0,
                }
            pre_filtered_results[point.id]["dense_score"] = point.score
            pre_filtered_results[point.id]["score"] += point.score * alpha

        # Process sparse results
        for point in sparse_response.points:
            if point.id not in pre_filtered_results:
                pre_filtered_results[point.id] = {
                    "id": point.id,
                    "score": 0.0,
                    "payload": point.payload,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "multivector_score": 0.0,
                }
            pre_filtered_results[point.id]["sparse_score"] = point.score
            pre_filtered_results[point.id]["score"] += point.score * beta

        # Sort pre-filtered results and keep top candidates
        pre_filtered_sorted = sorted(
            pre_filtered_results.values(), key=lambda x: x["score"], reverse=True
        )[:limit * 2]  # Keep top 2x candidates for multivector reranking

        logger.debug(f"Phase 1 completed: {len(pre_filtered_sorted)} candidates pre-filtered")

        # Phase 2: Multi-vector reranking (if multivector query is provided)
        if multivector_query is not None:
            logger.debug("Phase 2: Multi-vector reranking")
            
            # Extract candidate IDs for multivector search
            candidate_ids = [result["id"] for result in pre_filtered_sorted]
            
            # Execute multivector search on candidates
            multivector_response = await qdrant_client.query_points(
                collection_name=collection_name,
                query_vector=multivector_query[0] if multivector_query else [],  # Use first vector as query
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchAny(any=candidate_ids),
                        )
                    ]
                ) if candidate_ids else query_filter,
                limit=len(candidate_ids),
                with_payload=True,
                score_threshold=score_threshold,
            )

            # Apply multivector reranking scores
            for point in multivector_response.points:
                if point.id in pre_filtered_results:
                    # Normalize multivector score (MAX_SIM returns values between 0 and 1)
                    multivector_score = point.score * gamma
                    pre_filtered_results[point.id]["multivector_score"] = multivector_score
                    pre_filtered_results[point.id]["score"] += multivector_score

            logger.debug(f"Phase 2 completed: {len(multivector_response.points)} candidates reranked")
        else:
            # If no multivector query, apply gamma weight to existing scores
            for result in pre_filtered_sorted:
                result["score"] *= (1 + gamma)  # Boost scores if no reranking

        # Phase 3: Final ranking and result selection
        logger.debug("Phase 3: Final ranking and result selection")
        
        # Sort by final combined score
        final_sorted = sorted(
            pre_filtered_results.values(), key=lambda x: x["score"], reverse=True
        )[:limit]

        # Create a simple response object that mimics QueryResponse
        class SimpleQueryResponse:
            def __init__(self, points):
                self.points = points
        
        # Convert to QueryResponse format
        final_response = SimpleQueryResponse(
            points=[
                models.ScoredPoint(
                    id=result["id"],
                    version=1,  # Add required version field
                    score=result["score"],
                    payload=result["payload"],
                    vector=result.get("vector", {}),
                    metadata=result.get("metadata", {}),
                    document=result.get("document", "")
                )
                for result in final_sorted
            ]
        )

        logger.debug(f"BGE-M3 hybrid search found {len(final_response.points)} results")

        return final_response

    except Exception as e:
        logger.error(
            f"Error performing BGE-M3 hybrid search in collection '{collection_name}': {e}"
        )
        raise


async def bge_m3_hybrid_search_with_metadata(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_embeddings: Dict[str, Any],
    search_strategy: str = "hybrid",
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    limit: int = 10,
    metadata_filters: Optional[Dict] = None,
    score_threshold: Optional[float] = None,
    enable_multivector: bool = False,
) -> models.QueryResponse:
    """
    BGE-M3 hybrid search with metadata support
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        query_embeddings: Query embeddings dictionary
        search_strategy: Search strategy ("dense", "sparse", "multivector", "hybrid")
        alpha: Weight for dense search
        beta: Weight for sparse search
        gamma: Weight for multivector search
        limit: Number of results to return
        metadata_filters: Metadata filters to apply
        score_threshold: Minimum score threshold
        enable_multivector: Enable multivector search
        
    Returns:
        Query response
    """
    try:
        # Convert metadata filters to Qdrant filter
        query_filter = None
        if metadata_filters:
            query_filter = create_payload_filter(**metadata_filters)
        
        # Determine which embeddings to use based on search strategy
        dense_vector = query_embeddings.get("dense", [])
        sparse_vector = query_embeddings.get("sparse", {})
        multivector_query = query_embeddings.get("multivector", None)
        
        # Call the main hybrid search function
        return await bge_m3_hybrid_search_with_retry(
            qdrant_client,
            collection_name,
            dense_vector,
            sparse_vector,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            enable_multivector_reranking=enable_multivector,
            multivector_query=multivector_query,
        )
        
    except Exception as e:
        logger.error(f"Error in BGE-M3 hybrid search with metadata: {e}")
        raise


def create_hybrid_point(
    point_id: int,
    dense_vector: List[float],
    sparse_vector: Dict[str, float],
    payload: Dict[str, Any],
    multivector_vector: Optional[List[List[float]]] = None,
) -> models.PointStruct:
    """
    Create a point with dense, sparse, and optionally multi-vector vectors

    Args:
        point_id: ID of the point
        dense_vector: Dense vector
        sparse_vector: Sparse vector as dict {index: value}
        payload: Payload data
        multivector_vector: Optional multi-vector (ColBERT) embedding

    Returns:
        PointStruct with all vector types
    """
    vector_config = {
        "dense": dense_vector,
        "sparse": sparse_vector,
    }
    
    if multivector_vector is not None:
        vector_config["multivector"] = multivector_vector

    return models.PointStruct(
        id=point_id,
        vector=vector_config,
        payload=payload,
    )


async def upsert_hybrid_chunks_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    batch_size: int = 100,
    enable_multivector: bool = False,
) -> None:
    """
    Upsert chunks with dense, sparse, and optionally multi-vector vectors to Qdrant with retry logic and batching

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        chunks: List of chunks with dense_vector, sparse_vector, payload, and optionally multivector_vector
        batch_size: Batch size for upserting
        enable_multivector: Enable multi-vector support for chunks that have multivector_vector
    """
    logger.info(
        f"Upserting {len(chunks)} hybrid chunks to collection '{collection_name}'"
    )

    # Create PointStruct objects
    points = []
    for chunk in chunks:
        multivector_vector = chunk.get("multivector_vector") if enable_multivector else None
        
        point = create_hybrid_point(
            point_id=chunk["id"],
            dense_vector=chunk["dense_vector"],
            sparse_vector=chunk["sparse_vector"],
            payload=chunk["payload"],
            multivector_vector=multivector_vector,
        )
        points.append(point)

    # Use existing upsert function
    await upsert_with_retry(qdrant_client, collection_name, points, batch_size)


async def create_collection_if_not_exists(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    dense_vector_size: int = 1024,
    sparse_vector_size: int = 1000,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """
    Create a collection with dense and sparse vector support if it doesn't exist

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        dense_vector_size: Size of the dense vectors
        sparse_vector_size: Size of the sparse vectors
        distance: Distance metric for dense vectors
    """
    try:
        # Check if collection exists
        try:
            collections = await qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
        except Exception as e:
            logger.warning(f"Error checking collections: {e}")
            # If we can't check collections, assume it doesn't exist
            collection_names = []
        
        # Check if collection exists using get_collection
        try:
            await qdrant_client.get_collection(collection_name)
            logger.info(f"BGE-M3 collection '{collection_name}' already exists")
            return True
        except Exception:
            logger.info(f"Creating BGE-M3 optimized collection '{collection_name}'")

        if collection_name not in collection_names:
            logger.info(
                f"Creating collection '{collection_name}' with dense and sparse vectors"
            )

            # Configure dense vectors
            dense_config = models.VectorParams(
                size=dense_vector_size,
                distance=distance,
            )

            # Configure sparse vectors
            sparse_config = models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )

            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": dense_config,
                    "sparse": sparse_config,
                },
            )

            logger.info(
                f"Successfully created collection '{collection_name}' with hybrid search support"
            )
        else:
            logger.info(f"Collection '{collection_name}' already exists")

    except Exception as e:
        logger.error(f"Error creating collection '{collection_name}': {e}")
        raise


async def create_hybrid_collection_if_not_exists(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    dense_vector_size: int = 1024,
    sparse_vector_size: int = 1000,
    distance: models.Distance = models.Distance.COSINE,
    enable_multivector: bool = False,
    multivector_count: int = 16,
    multivector_dimension: int = 128,
    sparse_index_params: Optional[models.SparseIndexParams] = None,
) -> None:
    """
    Create a collection optimized for hybrid search (dense + sparse) if it doesn't exist
    Enhanced with BGE-M3 support including multi-vector capabilities

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        dense_vector_size: Size of the dense vectors
        sparse_vector_size: Size of the sparse vectors
        distance: Distance metric for dense vectors
        enable_multivector: Enable multi-vector (ColBERT) support
        multivector_count: Number of vectors in multi-vector embedding
        multivector_dimension: Dimension of each vector in multi-vector embedding
        sparse_index_params: Custom sparse index parameters for BGE-M3 optimization
    """
    try:
        # Check if collection exists
        collections = await qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if collection_name not in collection_names:
            logger.info(f"Creating hybrid collection '{collection_name}' with BGE-M3 support")

            # Configure dense vectors
            dense_config = models.VectorParams(
                size=dense_vector_size,
                distance=distance,
            )

            # Configure sparse vectors with BGE-M3 optimization
            if sparse_index_params is None:
                # Default sparse index optimized for BGE-M3
                sparse_index_params = models.SparseIndexParams(
                    on_disk=False,
                )

            sparse_config = models.SparseVectorParams(
                index=sparse_index_params,
            )

            # Configure multi-vector vectors (ColBERT) with COSINE comparator
            multivector_config = None
            if enable_multivector:
                logger.info(f"Enabling multi-vector support with {multivector_count} vectors of {multivector_dimension} dimensions")
                multivector_config = models.VectorParams(
                    size=multivector_dimension,
                    distance=models.Distance.COSINE,
                )

            # Build vectors configuration
            vectors_config = {
                "dense": dense_config,
                "sparse": sparse_config,
            }
            
            if enable_multivector:
                vectors_config["multivector"] = multivector_config

            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )

            logger.info(f"Successfully created hybrid collection '{collection_name}' with BGE-M3 support")
        else:
            logger.info(f"Collection '{collection_name}' already exists")

    except Exception as e:
        logger.error(f"Error creating hybrid collection '{collection_name}': {e}")
        raise


async def create_bge_m3_collection_if_not_exists(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    dense_vector_size: int = 1024,
    sparse_vector_size: int = 1000,
    multivector_count: int = 16,
    multivector_dimension: int = 128,
    distance: models.Distance = models.Distance.COSINE,
    sparse_index_params: Optional[models.SparseIndexParams] = None,
    config: Optional[Dict] = None,
) -> bool:
    """
    Create a collection specifically optimized for BGE-M3 with all three vector types:
    Dense, Sparse, and Multi-Vector (ColBERT)

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        dense_vector_size: Size of the dense vectors
        sparse_vector_size: Size of the sparse vectors
        multivector_count: Number of vectors in multi-vector embedding
        multivector_dimension: Dimension of each vector in multi-vector embedding
        distance: Distance metric for dense vectors
        sparse_index_params: Custom sparse index parameters for BGE-M3 optimization
        config: Additional configuration parameters
    """
    try:
        # Check if collection exists by trying to get it
        try:
            await qdrant_client.get_collection(collection_name)
            logger.info(f"BGE-M3 collection '{collection_name}' already exists")
            return True
        except Exception:
            logger.info(f"Creating BGE-M3 optimized collection '{collection_name}'")

        # If we get here, collection doesn't exist, so create it
        collections_response = await qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections_response.collections]

        if collection_name not in collection_names:
            # Use custom config if provided
            if config:
                logger.info(f"Using custom configuration for collection '{collection_name}'")
                # Convert the config to the proper format
                vectors_config = {}
                
                if "vectors" in config:
                    for vector_type, vector_config in config["vectors"].items():
                        if vector_type == "dense":
                            vectors_config["dense"] = models.VectorParams(
                                size=vector_config.get("size", dense_vector_size),
                                distance=vector_config.get("distance", distance)
                            )
                        elif vector_type == "sparse":
                            vectors_config["sparse"] = models.SparseVectorParams(
                                index=models.SparseIndexParams(
                                    on_disk=vector_config.get("on_disk", False),
                                    full_scan_threshold=vector_config.get("full_scan_threshold")
                                )
                            )
                        elif vector_type == "multivector" or vector_type == "multi_vector":
                            vectors_config["multivector"] = models.VectorParams(
                                size=vector_config.get("size", multivector_dimension),
                                distance=vector_config.get("distance", distance)
                            )
                
                # Extract additional parameters
                create_kwargs = {
                    "collection_name": collection_name,
                    "vectors_config": vectors_config
                }
                
                # Add additional parameters if they exist
                if "shards" in config:
                    create_kwargs["shards"] = config["shards"]
                
                await qdrant_client.create_collection(**create_kwargs)
            else:
                # Use default BGE-M3 configuration
                # Configure dense vectors
                dense_config = models.VectorParams(
                    size=dense_vector_size,
                    distance=distance,
                )

                # Configure sparse vectors with BGE-M3 optimization
                if sparse_index_params is None:
                    # Optimized sparse index for BGE-M3
                    sparse_index_params = models.SparseIndexParams(
                        on_disk=False,
                    )

                sparse_config = models.SparseVectorParams(
                    index=sparse_index_params,
                )

                # Configure multi-vector vectors (ColBERT) with COSINE comparator
                multivector_config = models.VectorParams(
                    size=multivector_dimension,
                    distance=models.Distance.COSINE,
                )

                # Build vectors configuration for BGE-M3
                vectors_config = {
                    "dense": dense_config,
                    "sparse": sparse_config,
                    "multivector": multivector_config,
                }

                await qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                )

            logger.info(f"Successfully created BGE-M3 collection '{collection_name}' with all three vector types")
            return True
        else:
            logger.info(f"BGE-M3 collection '{collection_name}' already exists")
            return True

    except Exception as e:
        logger.error(f"Error creating BGE-M3 collection '{collection_name}': {e}")
        return False


async def delete_collection(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
) -> None:
    """
    Delete a collection

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
    """
    try:
        logger.info(f"Deleting collection '{collection_name}'")

        await qdrant_client.delete_collection(collection_name=collection_name)

        logger.info(f"Successfully deleted collection '{collection_name}'")

    except Exception as e:
        logger.error(f"Error deleting collection '{collection_name}': {e}")
        raise


async def get_collection_info(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
) -> Dict[str, Any]:
    """
    Get information about a collection

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection

    Returns:
        Dictionary with collection information
    """
    try:
        collection_info = await qdrant_client.get_collection(
            collection_name=collection_name
        )

        return {
            "name": collection_name,
            "status": collection_info.status,
            "vectors_count": collection_info.config.params.vectors.size,
            "distance": collection_info.config.params.vectors.distance,
            "points_count": collection_info.points_count,
        }

    except Exception as e:
        logger.error(f"Error getting collection info for '{collection_name}': {e}")
        raise


async def cleanup_old_points(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    max_age_days: int = 30,
    batch_size: int = 1000,
) -> int:
    """
    Clean up old points from a collection

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        max_age_days: Maximum age in days
        batch_size: Batch size for deletion

    Returns:
        Number of deleted points
    """
    try:
        cutoff_timestamp = int(time.time()) - (max_age_days * 24 * 60 * 60)

        # This is a simplified implementation - in production, you might want to
        # store timestamps in the payload and use them for filtering
        logger.info(
            f"Cleaning up points older than {max_age_days} days from '{collection_name}'"
        )

        # Note: This is a placeholder implementation
        # In a real scenario, you would need to implement proper timestamp-based filtering
        # and deletion logic

        logger.info("Cleanup completed (placeholder implementation)")
        return 0

    except Exception as e:
        logger.error(f"Error cleaning up collection '{collection_name}': {e}")
        raise


def create_payload_filter(
    session_id: Optional[str] = None,
    **kwargs
) -> Optional[models.Filter]:
    """
    Create a payload filter for Qdrant queries
    
    Args:
        session_id: Session ID to filter by
        **kwargs: Additional filter criteria
        
    Returns:
        Filter object or None if no filters needed
    """
    try:
        filter_conditions = []
        
        # Add session ID filter if provided
        if session_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            )
        
        # Add additional filters from kwargs
        for key, value in kwargs.items():
            if key == "bbox" and isinstance(value, list) and len(value) == 4:
                filter_conditions.append(
                    models.FieldCondition(
                        key="bbox",
                        match=models.MatchValue(value=value),
                    )
                )
            elif key == "page" and isinstance(value, int):
                filter_conditions.append(
                    models.FieldCondition(
                        key="page",
                        match=models.MatchValue(value=value),
                    )
                )
            elif key == "element_type" and isinstance(value, str):
                filter_conditions.append(
                    models.FieldCondition(
                        key="element_type",
                        match=models.MatchValue(value=value),
                    )
                )
            elif key == "element_type" and isinstance(value, list):
                filter_conditions.append(
                    models.FieldCondition(
                        key="element_type",
                        match=models.MatchAny(any=value),
                    )
                )
            elif key == "must" and isinstance(value, list):
                # Allow direct filter conditions
                filter_conditions.extend(value)
            elif isinstance(value, dict) and "key" in value and "match" in value:
                # Support direct field condition format
                filter_conditions.append(
                    models.FieldCondition(
                        key=value["key"],
                        match=value["match"],
                    )
                )
        
        # Create filter if we have conditions
        if filter_conditions:
            return models.Filter(must=filter_conditions)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error creating payload filter: {e}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def create_image_collection_if_not_exists(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    vector_size: int = 512,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """
    Create a collection optimized for image embeddings if it doesn't exist

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        vector_size: Size of the image vectors
        distance: Distance metric for image vectors
    """
    try:
        # Check if collection exists
        collections = await qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if collection_name not in collection_names:
            logger.info(f"Creating image collection '{collection_name}'")

            # Configure image vectors
            image_config = models.VectorParams(
                size=vector_size,
                distance=distance,
            )

            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=image_config,
            )

            logger.info(f"Successfully created image collection '{collection_name}'")
        else:
            logger.info(f"Image collection '{collection_name}' already exists")

    except Exception as e:
        logger.error(f"Error creating image collection '{collection_name}': {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def upsert_image_embeddings_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    embeddings: List[Dict[str, Any]],
    batch_size: int = 50,
) -> None:
    """
    Upsert image embeddings to Qdrant with retry logic and batching

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        embeddings: List of embeddings with id, vector, and payload
        batch_size: Batch size for upserting
    """
    logger.info(
        f"Upserting {len(embeddings)} image embeddings to collection '{collection_name}'"
    )

    # Create PointStruct objects
    points = []
    for embedding in embeddings:
        point = models.PointStruct(
            id=embedding["id"],
            vector=embedding["vector"],
            payload=embedding["payload"],
        )
        points.append(point)

    # Use existing upsert function
    await upsert_with_retry(qdrant_client, collection_name, points, batch_size)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def search_image_embeddings(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
) -> models.QueryResponse:
    """
    Search for similar image embeddings

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        query_vector: Query vector
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold

    Returns:
        Query response with similar images
    """
    logger.debug(
        f"Searching image embeddings in collection '{collection_name}' with limit {limit}"
    )

    try:
        query_request = models.QueryRequest(
            query=models.Vector(
                name="image",
                vector=query_vector,
            ),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=query_request.query,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        logger.debug(f"Found {len(response.points)} similar images")
        return response

    except Exception as e:
        logger.error(
            f"Error searching image embeddings in collection '{collection_name}': {e}"
        )
        raise


async def find_related_text_chunks(
    qdrant_client: AsyncQdrantClient,
    image_metadata: Dict[str, Any],
    text_collection_name: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Find related text chunks for a given image based on shared metadata

    Args:
        qdrant_client: Qdrant client
        image_metadata: Image metadata containing page_number, document_id, etc.
        text_collection_name: Name of the text collection
        limit: Number of related chunks to return

    Returns:
        List of related text chunks
    """
    try:
        # Create filter based on shared metadata
        conditions = []

        # Filter by document ID
        if "document_id" in image_metadata:
            conditions.append(
                models.FieldCondition(
                    key="document",
                    match=models.MatchValue(value=image_metadata["document_id"]),
                )
            )

        # Filter by page number
        if "page_number" in image_metadata:
            conditions.append(
                models.FieldCondition(
                    key="page",
                    match=models.MatchValue(value=image_metadata["page_number"]),
                )
            )

        # Filter by element type (text or table)
        conditions.append(
            models.FieldCondition(
                key="element_type",
                match=models.MatchAny(any=["text", "table"]),
            )
        )

        query_filter = models.Filter(must=conditions) if conditions else None

        # Search for related text chunks
        response = await qdrant_client.scroll(
            collection_name=text_collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        related_chunks = []
        for point in response[0]:
            related_chunks.append(
                {
                    "id": point.id,
                    "score": 1.0,  # Perfect match based on metadata
                    "payload": point.payload,
                }
            )

        logger.info(f"Found {len(related_chunks)} related text chunks for image")
        return related_chunks

    except Exception as e:
        logger.error(f"Error finding related text chunks: {e}")
        return []


async def create_image_text_connections(
    qdrant_client: AsyncQdrantClient,
    image_collection_name: str,
    text_collection_name: str,
    session_id: str,
    document_id: str,
) -> Dict[str, List[str]]:
    """
    Create connections between image and text embeddings based on shared metadata

    Args:
        qdrant_client: Qdrant client
        image_collection_name: Name of the image collection
        text_collection_name: Name of the text collection
        session_id: Session identifier
        document_id: Document identifier

    Returns:
        Dictionary mapping image IDs to related text chunk IDs
    """
    try:
        logger.info(
            f"Creating connections between images and text for document '{document_id}'"
        )

        # Get all images for the document
        image_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                ),
            ]
        )

        image_response = await qdrant_client.scroll(
            collection_name=image_collection_name,
            scroll_filter=image_filter,
            limit=1000,
            with_payload=True,
        )

        connections = {}

        for image_point in image_response[0]:
            image_metadata = image_point.payload

            # Find related text chunks
            related_chunks = await find_related_text_chunks(
                qdrant_client=qdrant_client,
                image_metadata=image_metadata,
                text_collection_name=text_collection_name,
                limit=5,
            )

            if related_chunks:
                connections[image_point.id] = [chunk["id"] for chunk in related_chunks]
                logger.info(
                    f"Connected image {image_point.id} to {len(related_chunks)} text chunks"
                )

        logger.info(f"Created {len(connections)} image-text connections")
        return connections

    except Exception as e:
        logger.error(f"Error creating image-text connections: {e}")
        return {}


async def get_image_with_context(
    qdrant_client: AsyncQdrantClient,
    image_id: str,
    image_collection_name: str,
    text_collection_name: str,
    limit: int = 3,
) -> Dict[str, Any]:
    """
    Get image embedding with related text context

    Args:
        qdrant_client: Qdrant client
        image_id: ID of the image
        image_collection_name: Name of the image collection
        text_collection_name: Name of the text collection
        limit: Number of related text chunks to return

    Returns:
        Dictionary containing image data and related text context
    """
    try:
        # Get the image embedding
        image_response = await qdrant_client.retrieve(
            collection_name=image_collection_name,
            ids=[image_id],
            with_payload=True,
        )

        if not image_response.points:
            return {"error": "Image not found"}

        image_point = image_response.points[0]
        image_metadata = image_point.payload

        # Find related text chunks
        related_chunks = await find_related_text_chunks(
            qdrant_client=qdrant_client,
            image_metadata=image_metadata,
            text_collection_name=text_collection_name,
            limit=limit,
        )

        return {
            "image_id": image_id,
            "image_metadata": image_metadata,
            "related_text": related_chunks,
            "context_count": len(related_chunks),
        }

    except Exception as e:
        logger.error(f"Error getting image with context: {e}")
        return {"error": str(e)}


async def hybrid_search_with_metadata(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    dense_vector: List[float],
    sparse_vector: Dict[str, float],
    alpha: float = 0.5,
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
) -> models.QueryResponse:
    """
    Hybrid search with metadata filtering support

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        dense_vector: Dense query vector
        sparse_vector: Sparse query vector as dict {index: value}
        alpha: Weight for dense vs sparse search (0.0 = pure sparse, 1.0 = pure dense)
        limit: Number of results to return
        query_filter: Base filter to apply
        score_threshold: Minimum score threshold
        metadata_filters: Additional metadata filters (bbox, page_number, type)

    Returns:
        Query response with hybrid search results
    """
    logger.debug(
        f"Performing hybrid search with metadata in collection '{collection_name}'"
    )

    try:
        # Combine base filter with metadata filters
        combined_filter = query_filter

        if metadata_filters:
            metadata_conditions = []

            # Bbox filter
            if "bbox" in metadata_filters:
                bbox = metadata_filters["bbox"]
                if isinstance(bbox, list) and len(bbox) == 4:
                    metadata_conditions.append(
                        models.FieldCondition(
                            key="bbox",
                            match=models.MatchValue(value=bbox),
                        )
                    )

            # Page number filter
            if "page_number" in metadata_filters:
                page_number = metadata_filters["page_number"]
                metadata_conditions.append(
                    models.FieldCondition(
                        key="page",
                        match=models.MatchValue(value=page_number),
                    )
                )

            # Type filter
            if "type" in metadata_filters:
                element_type = metadata_filters["type"]
                if isinstance(element_type, str):
                    metadata_conditions.append(
                        models.FieldCondition(
                            key="element_type",
                            match=models.MatchValue(value=element_type),
                        )
                    )
                elif isinstance(element_type, list):
                    metadata_conditions.append(
                        models.FieldCondition(
                            key="element_type",
                            match=models.MatchAny(any=element_type),
                        )
                    )

            # Add metadata conditions to existing filter
            if metadata_conditions:
                if combined_filter and combined_filter.must:
                    combined_filter.must.extend(metadata_conditions)
                else:
                    combined_filter = models.Filter(must=metadata_conditions)

        # Use existing hybrid search function
        return await hybrid_search_with_retry(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            alpha=alpha,
            limit=limit,
            query_filter=combined_filter,
            score_threshold=score_threshold,
        )

    except Exception as e:
        logger.error(f"Error in hybrid search with metadata: {e}")
        raise


# Helper functions for BGE-M3 search operations
async def search_dense_embeddings(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
) -> models.QueryResponse:
    """
    Search for dense embeddings
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        query_vector: Query vector
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold
        
    Returns:
        Query response with dense search results
    """
    try:
        response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=models.Vector(
                name="dense",
                vector=query_vector,
            ),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )
        return response
    except Exception as e:
        logger.error(f"Error searching dense embeddings: {e}")
        raise


async def search_sparse_embeddings(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    sparse_vector: Dict[str, float],
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
) -> models.QueryResponse:
    """
    Search for sparse embeddings
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        sparse_vector: Sparse query vector
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold
        
    Returns:
        Query response with sparse search results
    """
    try:
        response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                name="sparse",
                vector=sparse_vector,
            ),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )
        return response
    except Exception as e:
        logger.error(f"Error searching sparse embeddings: {e}")
        raise


async def search_multivector_embeddings(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
) -> models.QueryResponse:
    """
    Search for multivector embeddings (ColBERT)
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        query_vector: Query vector
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold
        
    Returns:
        Query response with multivector search results
    """
    try:
        response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=models.Vector(
                name="multivector",
                vector=query_vector,
            ),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )
        return response
    except Exception as e:
        logger.error(f"Error searching multivector embeddings: {e}")
        raise


async def search_images_with_text_context(
    qdrant_client: AsyncQdrantClient,
    image_collection_name: str,
    text_collection_name: str,
    query_vector: List[float],
    text_query: str,
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
) -> models.QueryResponse:
    """
    Search for images with text context

    Args:
        qdrant_client: Qdrant client
        image_collection_name: Name of the image collection
        text_collection_name: Name of the text collection
        query_vector: Image query vector
        text_query: Text query for context
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold

    Returns:
        Query response with images and related text context
    """
    logger.debug(f"Searching images with text context in '{image_collection_name}'")

    try:
        # Search for similar images
        image_response = await search_image_embeddings(
            qdrant_client=qdrant_client,
            collection_name=image_collection_name,
            query_vector=query_vector,
            limit=limit * 2,  # Get more images to allow for filtering
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        # Enhance image results with text context
        enhanced_results = []

        for point in image_response.points:
            payload = point.payload or {}

            # Find related text chunks
            related_chunks = await find_related_text_chunks(
                qdrant_client=qdrant_client,
                image_metadata=payload,
                text_collection_name=text_collection_name,
                limit=3,
            )

            # Create enhanced result
            enhanced_result = {
                "id": point.id,
                "score": point.score,
                "payload": {
                    **payload,
                    "related_text": [
                        {
                            "id": chunk["id"],
                            "content": chunk["payload"].get("document", ""),
                            "page": chunk["payload"].get("page", 0),
                            "score": chunk["score"],
                        }
                        for chunk in related_chunks
                    ],
                },
            }

            enhanced_results.append(enhanced_result)

        # Convert back to QueryResponse format
        final_response = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=result["id"],
                    score=result["score"],
                    payload=result["payload"],
                )
                for result in enhanced_results[:limit]
            ]
        )

        logger.debug(f"Found {len(final_response.points)} images with text context")
        return final_response

    except Exception as e:
        logger.error(f"Error searching images with text context: {e}")
        raise


def combine_and_rank_results(
    text_results: List[Dict[str, Any]],
    image_results: List[Dict[str, Any]],
    strategy: str = "hybrid",
    text_weight: float = 0.7,
    image_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Combine and rank results from text and image searches

    Args:
        text_results: Results from text search
        image_results: Results from image search
        strategy: Combination strategy ("hybrid", "text_only", "image_only", "weighted")
        text_weight: Weight for text results (0.0-1.0)
        image_weight: Weight for image results (0.0-1.0)

    Returns:
        Combined and ranked results
    """
    logger.debug(f"Combining and ranking results with strategy '{strategy}'")

    try:
        combined_results = {}

        # Process text results
        for result in text_results:
            result_id = f"text_{result['id']}"
            combined_results[result_id] = {
                "id": result_id,
                "original_id": result["id"],
                "type": "text",
                "score": result["score"],
                "payload": result["payload"],
                "combined_score": result["score"] * text_weight,
            }

        # Process image results
        for result in image_results:
            result_id = f"image_{result['id']}"
            combined_results[result_id] = {
                "id": result_id,
                "original_id": result["id"],
                "type": "image",
                "score": result["score"],
                "payload": result["payload"],
                "combined_score": result["score"] * image_weight,
            }

        # Apply combination strategy
        if strategy == "hybrid":
            # Hybrid: combine scores with weights
            pass  # Already done above
        elif strategy == "text_only":
            # Only text results
            combined_results = {
                k: v for k, v in combined_results.items() if v["type"] == "text"
            }
        elif strategy == "image_only":
            # Only image results
            combined_results = {
                k: v for k, v in combined_results.items() if v["type"] == "image"
            }
        elif strategy == "weighted":
            # Weighted combination (already applied)
            pass

        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["combined_score"], reverse=True
        )

        logger.debug(f"Combined and ranked {len(sorted_results)} results")
        return sorted_results

    except Exception as e:
        logger.error(f"Error combining and ranking results: {e}")
        return []


def convert_bge_m3_sparse_to_qdrant_format(
    bge_m3_sparse: Dict[str, float],
    max_dimension: int = 10000
) -> List[Dict[str, Any]]:
    """
    Convert BGE-M3 sparse output to Qdrant sparse vector format
    
    Args:
        bge_m3_sparse: BGE-M3 sparse embedding as dict {token_id: weight}
        max_dimension: Maximum dimension for sparse vector
        
    Returns:
        Qdrant sparse vector format as list of {"index": int, "value": float}
    """
    try:
        # Handle None or invalid input
        if bge_m3_sparse is None:
            return []
        
        # Convert token IDs to integer indices and format for Qdrant
        qdrant_sparse = []
        
        for token_id, weight in bge_m3_sparse.items():
            # Convert token_id to integer
            try:
                index = int(token_id)
            except (ValueError, TypeError):
                # Skip invalid indices
                continue
            
            # Filter out very small weights and limit dimension
            # Keep negative values as they are important for sparse representation
            if abs(weight) > 1e-6 and len(qdrant_sparse) < max_dimension:
                qdrant_sparse.append({"index": index, "value": weight})
        
        # Sort by index for consistency
        qdrant_sparse.sort(key=lambda x: x["index"])
        
        logger.debug(f"Converted BGE-M3 sparse to Qdrant format: {len(qdrant_sparse)} dimensions")
        return qdrant_sparse
        
    except Exception as e:
        logger.error(f"Error converting BGE-M3 sparse to Qdrant format: {e}")
        return []


async def prepare_bge_m3_query_embeddings(
    query_text: str,
    bge_m3_service: Optional["BGE_M3_Service"] = None,
    search_mode: str = "hybrid",
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    multivector_strategy: str = "max_sim",
    normalize: bool = False,
    dense_only: bool = False,
    sparse_only: bool = False,
    multivector_only: bool = False
) -> Dict[str, Any]:
    """
    Prepare query embeddings for BGE-M3 search
    
    Args:
        query_text: Query text to embed
        bge_m3_service: Optional BGE-M3 service for embedding generation
        search_mode: Search mode ("dense", "sparse", "multivector", "hybrid")
        alpha: Weight for dense embeddings
        beta: Weight for sparse embeddings
        gamma: Weight for multivector embeddings
        multivector_strategy: Strategy for multivector generation
        normalize: Whether to normalize embeddings
        dense_only: Only generate dense embeddings
        sparse_only: Only generate sparse embeddings
        multivector_only: Only generate multivector embeddings
        
    Returns:
        Dictionary containing prepared embeddings
    """
    try:
        # Validate query text
        if not query_text or not query_text.strip():
            logger.error("Query text cannot be empty")
            raise ValueError("Query text cannot be empty")
        
        # Validate search mode
        valid_modes = ["dense", "sparse", "multivector", "hybrid"]
        if search_mode not in valid_modes:
            logger.error(f"Unknown search mode: {search_mode}")
            raise ValueError(f"Unknown search mode: {search_mode}")
        
        embeddings = {}
        errors = []
        
        # Determine which embedding types to generate
        generate_dense = dense_only or (search_mode in ["dense", "hybrid"] and not sparse_only and not multivector_only)
        generate_sparse = sparse_only or (search_mode in ["sparse", "hybrid"] and not dense_only and not multivector_only)
        generate_multivector = multivector_only or (search_mode in ["multivector", "hybrid"] and not dense_only and not sparse_only)
        
        logger.debug(f"Preparing BGE-M3 query embeddings: dense={generate_dense}, "
                    f"sparse={generate_sparse}, multivector={generate_multivector}")
        
        if bge_m3_service:
            # Use BGE-M3 service for embedding generation
            if generate_dense:
                try:
                    embeddings["dense"] = await bge_m3_service.generate_dense_embedding(query_text)
                except Exception as e:
                    errors.append(("dense", str(e)))
            
            if generate_sparse:
                try:
                    embeddings["sparse"] = await bge_m3_service.generate_sparse_embedding(query_text)
                except Exception as e:
                    errors.append(("sparse", str(e)))
            
            if generate_multivector:
                try:
                    embeddings["multivector"] = await bge_m3_service.generate_multivector_embedding(query_text)
                except Exception as e:
                    errors.append(("multivector", str(e)))
        else:
            # Fallback to mock embeddings for testing
            logger.warning("BGE-M3 service not available, using mock embeddings")
            
            if generate_dense:
                embeddings["dense"] = [0.1] * 1024
            
            if generate_sparse:
                embeddings["sparse"] = {str(i): 0.5 + (i % 5) * 0.1 for i in range(0, 100, 10)}
            
            if generate_multivector:
                embeddings["multivector"] = [[0.1] * 128 for _ in range(16)]
        
        # Log any errors
        if errors:
            for mode, error in errors:
                logger.error(f"Error generating {mode} embedding: {error}")
        
        
        if not embeddings:
            logger.warning(f"No embeddings generated for search mode: {search_mode}")
            # Don't raise an error, return empty embeddings instead
            return {"embeddings": {}, "errors": errors}
        
        return {
            "embeddings": embeddings,
            "errors": errors,
            "query_text": query_text[:100] + "..." if len(query_text) > 100 else query_text
        }
        
    except ValueError as e:
        # Re-raise validation errors
        raise e
    except Exception as e:
        logger.error(f"Error preparing BGE-M3 query embeddings: {e}")
        return {
            "embeddings": {},
            "errors": [("all", str(e))],
            "query_text": query_text[:100] + "..." if len(query_text) > 100 else query_text
        }


def format_bge_m3_search_results(
    search_response: models.QueryResponse,
    include_scores: bool = True,
    include_payload: bool = True,
    format_type: str = "standard"
) -> List[Dict[str, Any]]:
    """
    Format search results in BGE-M3 context
    
    Args:
        search_response: Qdrant search response
        include_scores: Include score information
        include_payload: Include payload data
        format_type: Format type ("standard", "detailed", "compact")
        
    Returns:
        Formatted search results
    """
    try:
        formatted_results = []
        
        for point in search_response.points:
            result = {}
            
            # Basic information
            result["id"] = point.id
            
            # Score information
            if include_scores and hasattr(point, "score"):
                result["score"] = point.score
                result["confidence"] = _calculate_confidence_level(point.score)
            
            # Payload information
            if include_payload and point.payload:
                result["payload"] = point.payload
                
                # Add BGE-M3 specific metadata if available
                if "document" in point.payload:
                    result["document"] = point.payload["document"]
                if "page" in point.payload:
                    result["page"] = point.payload["page"]
                if "element_type" in point.payload:
                    result["element_type"] = point.payload["element_type"]
            
            # Format based on type
            if format_type == "detailed":
                result["detailed_info"] = {
                    "vector_types": _extract_vector_types(point.payload),
                    "metadata_completeness": _calculate_metadata_completeness(point.payload)
                }
            elif format_type == "compact":
                # Keep only essential information
                compact_result = {"id": result["id"]}
                if include_scores and "score" in result:
                    compact_result["score"] = result["score"]
                if "document" in result:
                    compact_result["document"] = result["document"]
                if "page" in result:
                    compact_result["page"] = result["page"]
                result = compact_result
            
            formatted_results.append(result)
        
        logger.debug(f"Formatted {len(formatted_results)} search results in {format_type} format")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error formatting BGE-M3 search results: {e}")
        return []


def _calculate_confidence_level(score: float) -> str:
    """Calculate confidence level based on score"""
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    else:
        return "low"


def _extract_vector_types(payload: Optional[Dict[str, Any]]) -> List[str]:
    """Extract available vector types from payload"""
    if not payload:
        return []
    
    vector_types = []
    if "dense_vector" in payload:
        vector_types.append("dense")
    if "sparse_vector" in payload:
        vector_types.append("sparse")
    if "multivector_vector" in payload:
        vector_types.append("multivector")
    
    return vector_types


def _calculate_metadata_completeness(payload: Optional[Dict[str, Any]]) -> float:
    """Calculate metadata completeness score (0.0-1.0)"""
    if not payload:
        return 0.0
    
    required_fields = ["document", "page", "element_type"]
    present_fields = sum(1 for field in required_fields if field in payload)
    
    return present_fields / len(required_fields)


# BGE-M3 specific utility functions
async def bge_m3_batch_search(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_embeddings: Dict[str, Any],
    search_strategy: str = "hybrid",
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
    enable_multivector: bool = True,
) -> models.QueryResponse:
    """
    Perform batch search with BGE-M3 embeddings
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        query_embeddings: Dictionary containing dense, sparse, and multivector embeddings
        search_strategy: Search strategy ("dense_only", "sparse_only", "multivector_only", "hybrid")
        alpha: Weight for dense search
        beta: Weight for sparse search
        gamma: Weight for multivector reranking
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold
        enable_multivector: Enable multivector reranking
        
    Returns:
        Query response with batch search results
    """
    try:
        logger.debug(f"Performing BGE-M3 batch search with strategy: {search_strategy}")
        
        # Prepare search based on strategy
        if search_strategy == "dense_only":
            dense_vector = query_embeddings.get("dense", [])
            sparse_vector = {}
            multivector_query = None
        elif search_strategy == "sparse_only":
            dense_vector = []
            sparse_vector = query_embeddings.get("sparse", {})
            multivector_query = None
        elif search_strategy == "multivector_only":
            dense_vector = []
            sparse_vector = {}
            multivector_query = query_embeddings.get("multivector", [])
        else:  # hybrid
            dense_vector = query_embeddings.get("dense", [])
            sparse_vector = query_embeddings.get("sparse", {})
            multivector_query = query_embeddings.get("multivector", [])
        
        # Perform BGE-M3 hybrid search
        response = await bge_m3_hybrid_search_with_retry(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            multivector_query=multivector_query,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )
        
        logger.debug(f"BGE-M3 batch search found {len(response.points)} results")
        return response
        
    except Exception as e:
        logger.error(f"Error in BGE-M3 batch search: {e}")
        raise


async def bge_m3_multivector_rerank(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    candidate_ids: List[str],
    multivector_query: List[List[float]],
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
) -> models.QueryResponse:
    """
    Perform multivector reranking on candidate results
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        candidate_ids: List of candidate IDs to rerank
        multivector_query: Multivector query for reranking
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold
        
    Returns:
        Query response with reranked results
    """
    try:
        logger.debug(f"Performing BGE-M3 multivector reranking on {len(candidate_ids)} candidates")
        
        # Create filter for candidate IDs
        candidate_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="id",
                    match=models.MatchAny(any=candidate_ids),
                )
            ]
        ) if candidate_ids else query_filter
        
        # Perform multivector search
        response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=models.Vector(
                name="multivector",
                vector=multivector_query[0] if multivector_query else [],
            ),
            limit=limit,
            query_filter=candidate_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )
        
        logger.debug(f"Multivector reranking completed: {len(response.points)} results")
        return response
        
    except Exception as e:
        logger.error(f"Error in BGE-M3 multivector reranking: {e}")
        raise


async def bge_m3_similarity_search(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_vector: List[float],
    vector_type: str = "dense",
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
    search_params: Optional[models.SearchParams] = None,
) -> models.QueryResponse:
    """
    Perform similarity search with specific vector type
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        query_vector: Query vector
        vector_type: Type of vector ("dense", "sparse", "multivector")
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold
        search_params: Search parameters
        
    Returns:
        Query response with similarity search results
    """
    try:
        logger.debug(f"Performing {vector_type} similarity search")
        
        if vector_type == "sparse":
            # Sparse vector search
            query = models.SparseVector(
                name="sparse",
                vector=query_vector,
            )
        else:
            # Dense or multivector search
            query = models.Vector(
                name=vector_type,
                vector=query_vector,
            )
        
        response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=query,
            limit=limit,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=True,
            score_threshold=score_threshold,
        )
        
        logger.debug(f"Similarity search found {len(response.points)} results")
        return response
        
    except Exception as e:
        logger.error(f"Error in {vector_type} similarity search: {e}")
        raise


async def bge_m3_hybrid_search_with_metadata(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_embeddings: Dict[str, Any],
    search_strategy: str = "hybrid",
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    limit: int = 10,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    enable_multivector: bool = True,
) -> models.QueryResponse:
    """
    Perform hybrid search with BGE-M3 embeddings and metadata filtering
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        query_embeddings: Dictionary containing dense, sparse, and multivector embeddings
        search_strategy: Search strategy ("dense_only", "sparse_only", "multivector_only", "hybrid")
        alpha: Weight for dense search
        beta: Weight for sparse search
        gamma: Weight for multivector reranking
        limit: Number of results to return
        query_filter: Base filter to apply
        score_threshold: Minimum score threshold
        metadata_filters: Additional metadata filters
        enable_multivector: Enable multivector reranking
        
    Returns:
        Query response with hybrid search results
    """
    try:
        logger.debug(f"Performing BGE-M3 hybrid search with metadata")
        
        # Combine query filter with metadata filters
        combined_filter = query_filter
        if metadata_filters:
            metadata_conditions = []
            
            # Add metadata conditions
            for key, value in metadata_filters.items():
                if key == "bbox" and isinstance(value, list) and len(value) == 4:
                    metadata_conditions.append(
                        models.FieldCondition(
                            key="bbox",
                            match=models.MatchValue(value=value),
                        )
                    )
                elif key == "page" and isinstance(value, int):
                    metadata_conditions.append(
                        models.FieldCondition(
                            key="page",
                            match=models.MatchValue(value=value),
                        )
                    )
                elif key == "element_type" and isinstance(value, str):
                    metadata_conditions.append(
                        models.FieldCondition(
                            key="element_type",
                            match=models.MatchValue(value=value),
                        )
                    )
                elif key == "element_type" and isinstance(value, list):
                    metadata_conditions.append(
                        models.FieldCondition(
                            key="element_type",
                            match=models.MatchAny(any=value),
                        )
                    )
            
            # Add metadata conditions to existing filter
            if metadata_conditions:
                if combined_filter and combined_filter.must:
                    combined_filter.must.extend(metadata_conditions)
                else:
                    combined_filter = models.Filter(must=metadata_conditions)
        
        # Perform batch search
        response = await bge_m3_batch_search(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            query_embeddings=query_embeddings,
            search_strategy=search_strategy,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            limit=limit,
            query_filter=combined_filter,
            score_threshold=score_threshold,
            enable_multivector=enable_multivector,
        )
        
        logger.debug(f"BGE-M3 hybrid search with metadata completed: {len(response.points)} results")
        return response
        
    except Exception as e:
        logger.error(f"Error in BGE-M3 hybrid search with metadata: {e}")
        raise


def validate_bge_m3_embeddings(embeddings: Dict[str, Any]) -> bool:
    """
    Validate BGE-M3 embeddings structure and content
    
    Args:
        embeddings: Dictionary containing embeddings
        
    Returns:
        True if embeddings are valid, False otherwise
    """
    try:
        required_types = ["dense", "sparse", "multivector"]
        
        for embedding_type in required_types:
            if embedding_type not in embeddings:
                logger.warning(f"Missing {embedding_type} embedding")
                continue
            
            embedding_data = embeddings[embedding_type]
            
            if embedding_type == "dense":
                # Validate dense vector
                if not isinstance(embedding_data, list):
                    logger.error(f"Dense embedding is not a list")
                    return False
                
                if len(embedding_data) != 1024:
                    logger.error(f"Dense embedding has wrong dimension: {len(embedding_data)}")
                    return False
                
                # Check for NaN or infinite values
                for value in embedding_data:
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        logger.error(f"Dense embedding contains invalid value: {value}")
                        return False
            
            elif embedding_type == "sparse":
                # Validate sparse vector
                if not isinstance(embedding_data, dict):
                    logger.error(f"Sparse embedding is not a dict")
                    return False
                
                # Check for valid indices and values
                for index, value in embedding_data.items():
                    if not isinstance(index, str):
                        logger.error(f"Sparse embedding index is not string: {index}")
                        return False
                    
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        logger.error(f"Sparse embedding contains invalid value: {value}")
                        return False
            
            elif embedding_type == "multivector":
                # Validate multivector
                if not isinstance(embedding_data, list):
                    logger.error(f"Multivector embedding is not a list")
                    return False
                
                if len(embedding_data) != 16:  # BGE-M3 default
                    logger.warning(f"Multivector has unexpected count: {len(embedding_data)}")
                
                # Check each vector in multivector
                for i, vector in enumerate(embedding_data):
                    if not isinstance(vector, list):
                        logger.error(f"Multivector vector {i} is not a list")
                        return False
                    
                    if len(vector) != 128:  # BGE-M3 default
                        logger.error(f"Multivector vector {i} has wrong dimension: {len(vector)}")
                        return False
                    
                    # Check for NaN or infinite values
                    for value in vector:
                        if not isinstance(value, (int, float)) or not math.isfinite(value):
                            logger.error(f"Multivector vector {i} contains invalid value: {value}")
                            return False
        
        logger.debug("BGE-M3 embeddings validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating BGE-M3 embeddings: {e}")
        return False


def optimize_bge_m3_query(
    query_text: str,
    max_length: int = 512,
    min_length: int = 10,
    truncate_method: str = "middle"
) -> str:
    """
    Optimize query text for BGE-M3 embedding generation
    
    Args:
        query_text: Original query text
        max_length: Maximum allowed length
        min_length: Minimum required length
        truncate_method: Method for truncation ("start", "middle", "end")
        
    Returns:
        Optimized query text
    """
    try:
        # Remove extra whitespace
        optimized_text = " ".join(query_text.split())
        
        # Check length requirements
        if len(optimized_text) < min_length:
            logger.warning(f"Query text too short: {len(optimized_text)} < {min_length}")
            return optimized_text
        
        # Truncate if necessary
        if len(optimized_text) > max_length:
            logger.debug(f"Truncating query text from {len(optimized_text)} to {max_length} characters")
            
            if truncate_method == "start":
                optimized_text = optimized_text[-max_length:]
            elif truncate_method == "middle":
                start_len = (max_length - 3) // 2
                end_len = max_length - 3 - start_len
                optimized_text = optimized_text[:start_len] + "..." + optimized_text[-end_len:]
            else:  # end
                optimized_text = optimized_text[:max_length]
        
        logger.debug(f"Optimized query text: {optimized_text[:100]}...")
        return optimized_text
        
    except Exception as e:
        logger.error(f"Error optimizing query text: {e}")
        return query_text


# Add math import for validation
import math


class BGE_M3_QdrantUtils:
    """BGE-M3 Qdrant Utils wrapper class for convenience"""
    
    def __init__(self, settings=None, qdrant_client=None):
        """Initialize with settings and optional qdrant client"""
        self.settings = settings
        self.qdrant_client = qdrant_client
        self.bge_m3_service = None
        self._initialize_bge_m3_service()
        
        # Ensure qdrant_client is not None
        if self.qdrant_client is None:
            raise ValueError("qdrant_client cannot be None")
    
    def _initialize_bge_m3_service(self):
        """Lazy initialization of BGE-M3 Service"""
        if self.bge_m3_service is None:
            BGE_M3_Service = get_bge_m3_service()
            if self.settings:
                self.bge_m3_service = BGE_M3_Service(self.settings)
            else:
                # Create minimal settings for BGE-M3 service if none provided
                from src.app.settings import Settings
                default_settings = Settings()
                self.bge_m3_service = BGE_M3_Service(default_settings)
    
    async def create_collection(self, collection_name: str, config: Optional[Dict] = None) -> bool:
        """Create BGE-M3 collection"""
        return await create_bge_m3_collection_if_not_exists(
            self.qdrant_client,
            collection_name,
            dense_vector_size=1024,
            sparse_vector_size=1000,
            config=config
        )
    
    async def hybrid_search(
        self,
        collection_name: str,
        query_embeddings: Dict[str, Any],
        search_mode: str = "hybrid",
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        top_k: int = 10,
        metadata_filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
        multivector_strategy: str = "max_sim",
        normalize: bool = False,
        enable_multivector: bool = False,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Perform hybrid search using BGE-M3 embeddings"""
        result = await bge_m3_hybrid_search_with_metadata(
            self.qdrant_client,
            collection_name,
            query_embeddings,
            search_strategy=search_mode,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            limit=top_k,
            metadata_filters=metadata_filters,
            score_threshold=score_threshold,
            enable_multivector=enable_multivector,
        )
        
        # Convert to expected format
        return {
            "results": [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                    "metadata": getattr(point, "metadata", {}),
                    "document": getattr(point, "document", "")
                }
                for point in result.points
            ]
        }
    
    async def prepare_query_embeddings(
        self,
        query_text: str,
        search_mode: str = "hybrid",
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        multivector_strategy: str = "max_sim",
        normalize: bool = False,
        dense_only: bool = False,
        sparse_only: bool = False,
        multivector_only: bool = False
    ) -> Dict[str, Any]:
        """Prepare query embeddings using BGE-M3 service"""
        return await prepare_bge_m3_query_embeddings(
            query_text,
            self.bge_m3_service,
            search_mode=search_mode,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            multivector_strategy=multivector_strategy,
            normalize=normalize,
            dense_only=dense_only,
            sparse_only=sparse_only,
            multivector_only=multivector_only
        )
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get collection information"""
        try:
            return await self.qdrant_client.get_collection(collection_name)
        except Exception:
            return None
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            await self.qdrant_client.delete_collection(collection_name)
            return True
        except Exception:
            return False
    
    async def get_collection_stats(self, collection_name: str) -> Optional[Dict]:
        """Get collection statistics"""
        try:
            return await self.qdrant_client.get_collection_stats(collection_name)
        except Exception:
            return None
    
    async def optimize_collection(self, collection_name: str) -> bool:
        """Optimize a collection"""
        try:
            await self.qdrant_client.optimize(collection_name)
            return True
        except Exception:
            return False
    
    async def health_check(self, collection_name: str) -> Dict[str, Any]:
        """Check collection health"""
        try:
            info = await self.get_collection_info(collection_name)
            if info:
                return {
                    "status": "healthy",
                    "collection_exists": True,
                    "info": info
                }
            else:
                return {
                    "status": "unhealthy",
                    "collection_exists": False,
                    "error": "Collection not found"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "collection_exists": False,
                "error": str(e)
            }
        except Exception:
            return {
                "status": "unhealthy",
                "collection_exists": False,
                "error": "Collection not found"
            }
    
    async def batch_upsert(
        self,
        collection_name: str,
        points: List[Dict],
        batch_size: int = 100,
        max_retries: int = 3
    ) -> bool:
        """Batch upsert points with retry"""
        try:
            await upsert_with_retry(
                self.qdrant_client,
                collection_name,
                points,
                batch_size=batch_size
            )
            return True
        except Exception:
            return False
    
    async def batch_delete(self, collection_name: str, point_ids: List[Any]) -> bool:
        """Batch delete points"""
        try:
            from qdrant_client.http import models as http_models
            
            await self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=http_models.Filter(
                    must=[
                        http_models.FieldCondition(
                            key="id",
                            match=http_models.MatchAny(any=point_ids)
                        )
                    ]
                )
            )
            return True
        except Exception:
            return False
    
    async def scroll_collection(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[int] = None,
        filters: Optional[Dict] = None,
        with_payload: bool = True,
        with_vectors: bool = True
    ) -> tuple:
        """Scroll through collection"""
        kwargs = {
            "collection_name": collection_name,
            "limit": limit,
            "with_payload": with_payload,
            "with_vectors": with_vectors
        }
        
        if offset is not None:
            kwargs["offset"] = offset
        
        if filters:
            kwargs["scroll_filter"] = create_payload_filter(**filters)
        
        result = await self.qdrant_client.scroll(**kwargs)
        return result

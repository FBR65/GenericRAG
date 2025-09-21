"""
Qdrant utility functions
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http import models as http_models
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


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
) -> models.QueryResponse:
    """
    Hybrid search in Qdrant with dense and sparse vectors

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        dense_vector: Dense query vector
        sparse_vector: Sparse query vector as dict {index: value}
        alpha: Weight for dense vs sparse search (0.0 = pure sparse, 1.0 = pure dense)
        limit: Number of results to return
        query_filter: Filter to apply
        score_threshold: Minimum score threshold

    Returns:
        Query response
    """
    logger.debug(
        f"Performing hybrid search in collection '{collection_name}' with alpha={alpha}"
    )

    try:
        # Create query for dense vectors
        dense_query = models.QueryRequest(
            query=models.Vector(
                name="dense",
                vector=dense_vector,
            ),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        # Create query for sparse vectors
        sparse_query = models.QueryRequest(
            query=models.SparseVector(
                name="sparse",
                vector=sparse_vector,
            ),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        # Execute both queries
        dense_response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=dense_query.query,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold,
        )

        sparse_response = await qdrant_client.query_points(
            collection_name=collection_name,
            query=sparse_query.query,
            limit=limit,
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
                }
            combined_results[point.id]["sparse_score"] = point.score
            combined_results[point.id]["score"] += point.score * (1 - alpha)

        # Sort by combined score and limit results
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["score"], reverse=True
        )[:limit]

        # Convert to QueryResponse format
        final_response = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=result["id"],
                    score=result["score"],
                    payload=result["payload"],
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


def create_hybrid_point(
    point_id: int,
    dense_vector: List[float],
    sparse_vector: Dict[str, float],
    payload: Dict[str, Any],
) -> models.PointStruct:
    """
    Create a point with both dense and sparse vectors

    Args:
        point_id: ID of the point
        dense_vector: Dense vector
        sparse_vector: Sparse vector as dict {index: value}
        payload: Payload data

    Returns:
        PointStruct with both vector types
    """
    return models.PointStruct(
        id=point_id,
        vector={
            "dense": dense_vector,
            "sparse": sparse_vector,
        },
        payload=payload,
    )


async def upsert_hybrid_chunks_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    batch_size: int = 100,
) -> None:
    """
    Upsert chunks with both dense and sparse vectors to Qdrant with retry logic and batching

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        chunks: List of chunks with dense_vector, sparse_vector, and payload
        batch_size: Batch size for upserting
    """
    logger.info(
        f"Upserting {len(chunks)} hybrid chunks to collection '{collection_name}'"
    )

    # Create PointStruct objects
    points = []
    for chunk in chunks:
        point = create_hybrid_point(
            point_id=chunk["id"],
            dense_vector=chunk["dense_vector"],
            sparse_vector=chunk["sparse_vector"],
            payload=chunk["payload"],
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
        collections = await qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

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
) -> None:
    """
    Create a collection optimized for hybrid search (dense + sparse) if it doesn't exist

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        dense_vector_size: Size of the dense vectors
        sparse_vector_size: Size of the sparse vectors
        distance: Distance metric for dense vectors
    """
    try:
        # Check if collection exists
        collections = await qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if collection_name not in collection_names:
            logger.info(f"Creating hybrid collection '{collection_name}'")

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

            logger.info(f"Successfully created hybrid collection '{collection_name}'")
        else:
            logger.info(f"Collection '{collection_name}' already exists")

    except Exception as e:
        logger.error(f"Error creating hybrid collection '{collection_name}': {e}")
        raise


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
    document: Optional[str] = None,
    page_range: Optional[tuple[int, int]] = None,
) -> Optional[models.Filter]:
    """
    Create a payload filter for Qdrant queries

    Args:
        session_id: Filter by session ID
        document: Filter by document name
        page_range: Filter by page range (min, max)

    Returns:
        Filter object or None
    """
    conditions = []

    if session_id:
        conditions.append(
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            )
        )

    if document:
        conditions.append(
            models.FieldCondition(
                key="document",
                match=models.MatchValue(value=document),
            )
        )

    if page_range:
        min_page, max_page = page_range
        conditions.append(
            models.FieldCondition(
                key="page",
                range=models.Range(
                    gte=min_page,
                    lte=max_page,
                ),
            )
        )

    return models.Filter(must=conditions) if conditions else models.Filter(must=[])


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
        response = await qdrant_client.query_points(
            collection_name=collection_name,
            query_vector=query_vector,
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

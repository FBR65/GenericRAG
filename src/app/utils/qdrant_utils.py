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
        batch = points[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            await qdrant_client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            logger.debug(f"Upserted batch {batch_num}/{(len(points) + batch_size - 1) // batch_size}")
            
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
        return response
        
    except Exception as e:
        logger.error(f"Error searching in collection '{collection_name}': {e}")
        raise


async def create_collection_if_not_exists(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    vector_size: int = 768,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """
    Create a collection if it doesn't exist
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        vector_size: Size of the vectors
        distance: Distance metric
    """
    try:
        # Check if collection exists
        collections = await qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if collection_name not in collection_names:
            logger.info(f"Creating collection '{collection_name}'")
            
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )
            
            logger.info(f"Successfully created collection '{collection_name}'")
        else:
            logger.info(f"Collection '{collection_name}' already exists")
            
    except Exception as e:
        logger.error(f"Error creating collection '{collection_name}': {e}")
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
        collection_info = await qdrant_client.get_collection(collection_name=collection_name)
        
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
        logger.info(f"Cleaning up points older than {max_age_days} days from '{collection_name}'")
        
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
    
    return models.Filter(must=conditions) if conditions else None
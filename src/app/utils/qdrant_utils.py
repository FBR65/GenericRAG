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
    
    # Debug: Check collection info before upsert
    try:
        collection_info = await qdrant_client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' info: {collection_info.config.params.vectors}")
        # Check if 'dense' vector exists
        if hasattr(collection_info.config.params.vectors, 'dense'):
            logger.info("Dense vector configuration exists")
        else:
            logger.error("Dense vector configuration missing from collection")
    except Exception as e:
        logger.error(f"Error getting collection info for '{collection_name}': {e}")

    # Process in batches
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        batch_num = i // batch_size + 1
        
        # Debug: Log first point in batch to check vector format
        if batch and i == 0:
            logger.debug(f"First point in batch: vector={batch[0].vector.keys()}, payload_keys={list(batch[0].payload.keys()) if batch[0].payload else {}}")

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
            # Debug: Log more details about the error
            logger.error(f"Error type: {type(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Error response: {e.response.text}")
            raise

    logger.info(f"Successfully upserted {len(points)} points")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def create_collection_if_not_exists(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    dense_vector_size: int = 1024,
    sparse_vector_size: int = 1000,
    distance: models.Distance = models.Distance.COSINE,
    force_recreate_if_missing_dense: bool = True,
) -> None:
    """
    Create a collection with dense and sparse vector support if it doesn't exist
    If it exists but is missing the dense vector configuration, recreate it by default

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        dense_vector_size: Size of the dense vectors
        sparse_vector_size: Size of the sparse vectors
        distance: Distance metric for dense vectors
        force_recreate_if_missing_dense: Force recreation if collection exists but lacks dense config
    """
    try:
        # Debug: Log collection creation parameters
        logger.debug(f"DEBUG: Creating collection '{collection_name}' with parameters:")
        logger.debug(f"DEBUG: dense_vector_size: {dense_vector_size}")
        logger.debug(f"DEBUG: sparse_vector_size: {sparse_vector_size}")
        logger.debug(f"DEBUG: distance: {distance}")
        logger.debug(f"DEBUG: force_recreate_if_missing_dense: {force_recreate_if_missing_dense}")
        
        # Check if collection exists
        try:
            collections = await qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            logger.info(f"Available collections: {collection_names}")
        except Exception as e:
            logger.warning(f"Error checking collections: {e}")
            # If we can't check collections, assume it doesn't exist
            collection_names = []
        
        # Check if collection exists using get_collection
        collection_exists = False
        collection_has_dense = False
        
        try:
            existing_collection = await qdrant_client.get_collection(collection_name)
            collection_exists = True
            logger.info(f"Collection '{collection_name}' already exists with config: {existing_collection.config.params.vectors}")
            
            # Debug: Log existing collection structure
            logger.debug(f"DEBUG: Existing collection vectors type: {type(existing_collection.config.params.vectors)}")
            logger.debug(f"DEBUG: Existing collection vectors attributes: {dir(existing_collection.config.params.vectors)}")
            
            # Check if dense vector exists
            if hasattr(existing_collection.config.params.vectors, 'dense'):
                logger.info("✓ Dense vector configuration exists in existing collection")
                collection_has_dense = True
                return
            else:
                logger.error("✗ Dense vector configuration missing from existing collection")
                logger.debug(f"DEBUG: Collection vectors structure: {existing_collection.config.params.vectors}")
                
                # If collection exists but lacks dense config, recreate it by default
                if force_recreate_if_missing_dense:
                    logger.info("Force recreating collection to fix missing dense vector configuration")
                    # Delete existing collection first
                    try:
                        await qdrant_client.delete_collection(collection_name)
                        logger.info(f"Deleted existing collection '{collection_name}'")
                    except Exception as del_error:
                        logger.error(f"Error deleting existing collection: {del_error}")
                        raise
                else:
                    logger.warning("Collection exists but lacks dense vector configuration. Set force_recreate_if_missing_dense=True to fix.")
                    return
                    
        except Exception as e:
            logger.info(f"Collection '{collection_name}' does not exist, creating it: {e}")

        # Create collection (either new or after deletion)
        logger.info(
            f"Creating collection '{collection_name}' with dense and sparse vectors"
        )

        # Configure dense vectors
        dense_config = models.VectorParams(
            size=dense_vector_size,
            distance=distance,
        )

        # Configure sparse vectors - Qdrant expects a different structure for sparse vectors
        # We need to use the correct structure for hybrid search
        sparse_config = models.VectorParams(
            size=sparse_vector_size,
            distance=models.Distance.DOT,  # Use DOT product for sparse vectors
            on_disk=False,
        )

        # Create vectors config with the correct structure
        vectors_config = {
            "dense": dense_config,
            "sparse": sparse_config,
        }
        
        # Debug: Log configuration details
        logger.debug(f"DEBUG: Dense config: {dense_config}")
        logger.debug(f"DEBUG: Sparse config: {sparse_config}")
        logger.debug(f"DEBUG: Vectors config: {vectors_config}")

        logger.info(f"Creating collection with vectors config: {vectors_config}")

        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )

        # Verify collection was created correctly
        try:
            new_collection = await qdrant_client.get_collection(collection_name)
            logger.info(f"Successfully created collection '{collection_name}' with config: {new_collection.config.params.vectors}")
            # Debug: Log new collection structure
            logger.debug(f"DEBUG: New collection vectors type: {type(new_collection.config.params.vectors)}")
            logger.debug(f"DEBUG: New collection vectors attributes: {dir(new_collection.config.params.vectors)}")
            
            # Verify dense vector exists
            if hasattr(new_collection.config.params.vectors, 'dense'):
                logger.info("✓ Dense vector configuration successfully created")
                logger.debug(f"DEBUG: Dense vector config: {new_collection.config.params.vectors.dense}")
            else:
                logger.error("✗ Dense vector configuration missing from newly created collection")
                logger.debug(f"DEBUG: New collection vectors structure: {new_collection.config.params.vectors}")
        except Exception as verify_error:
            logger.error(f"Error verifying created collection: {verify_error}")

    except Exception as e:
        logger.error(f"Error creating collection '{collection_name}': {e}")
        logger.debug(f"DEBUG: Error type: {type(e)}")
        logger.debug(f"DEBUG: Error details: {e}")
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
        logger.info(f"Hybrid collection check - available collections: {collection_names}")

        # Check if collection already exists and get its config
        try:
            existing_collection = await qdrant_client.get_collection(collection_name)
            logger.info(f"Hybrid collection '{collection_name}' already exists with config: {existing_collection.config.params.vectors}")
            # Verify dense vector exists
            if hasattr(existing_collection.config.params.vectors, 'dense'):
                logger.info("✓ Dense vector configuration exists in existing hybrid collection")
            else:
                logger.error("✗ Dense vector configuration missing from existing hybrid collection")
            return
        except Exception as e:
            logger.info(f"Hybrid collection '{collection_name}' does not exist, creating it: {e}")

        # Configure dense vectors
        dense_config = models.VectorParams(
            size=dense_vector_size,
            distance=distance,
        )

        # Configure sparse vectors with BGE-M3 optimization
        if sparse_index_params is None:
            sparse_index_params = models.SparseIndexParams(
                on_disk=False,
            )
        
        sparse_config = models.SparseVectorParams(
            index=sparse_index_params,
        )

        # Configure multi-vector vectors if enabled
        vectors_config = {
            "dense": dense_config,
            "sparse": sparse_config,
        }
        
        if enable_multivector:
            multivector_config = models.VectorParams(
                size=multivector_dimension,
                distance=distance,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                    max_sample_size=multivector_count,
                ),
            )
            vectors_config["multivector"] = multivector_config
            logger.info(f"Enabled multi-vector with {multivector_count} vectors of {multivector_dimension} dimensions each")

        logger.info(f"Creating hybrid collection with vectors config: {vectors_config}")
        logger.info(f"Dense config: {dense_config}")
        logger.info(f"Sparse config: {sparse_config}")

        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )

        # Verify collection was created correctly
        try:
            new_collection = await qdrant_client.get_collection(collection_name)
            logger.info(f"Successfully created hybrid collection '{collection_name}' with config: {new_collection.config.params.vectors}")
            # Verify dense vector exists
            if hasattr(new_collection.config.params.vectors, 'dense'):
                logger.info("✓ Dense vector configuration successfully created")
            else:
                logger.error("✗ Dense vector configuration missing from newly created hybrid collection")
        except Exception as verify_error:
            logger.error(f"Error verifying created hybrid collection: {verify_error}")

    except Exception as e:
        logger.error(f"Error creating hybrid collection '{collection_name}': {e}")
        raise


async def create_hybrid_chunks(
    chunks: List[Dict[str, Any]],
    bge_m3_service: "BGE_M3_Service",
    enable_multivector: bool = False,
    multivector_count: int = 16,
    multivector_dimension: int = 128,
) -> List[Dict[str, Any]]:
    """
    Create hybrid chunks with dense, sparse, and optionally multi-vector embeddings

    Args:
        chunks: List of chunks to process
        bge_m3_service: BGE-M3 service for embeddings
        enable_multivector: Enable multi-vector (ColBERT) support
        multivector_count: Number of vectors in multi-vector embedding
        multivector_dimension: Dimension of each vector in multi-vector embedding

    Returns:
        List of processed chunks with embeddings
    """
    logger.info(f"Processing {len(chunks)} chunks with BGE-M3")
    
    processed_chunks = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Get text content
            text_content = chunk.get("text_content", "")
            if not text_content:
                logger.warning(f"Chunk {i} has no text content, skipping")
                continue
            
            # Generate embeddings using BGE-M3
            embeddings = await bge_m3_service.generate_embeddings(text_content)
            
            if not embeddings:
                logger.warning(f"Failed to generate embeddings for chunk {i}")
                continue
            
            # Extract dense and sparse vectors
            dense_vector = embeddings.get("dense", [])
            sparse_vector = embeddings.get("sparse", {})
            multivector_vector = embeddings.get("multivector") if enable_multivector else None
            
            # Debug: Log embedding info
            if i < 3:  # Log first 3 chunks for debugging
                logger.debug(f"Chunk {i}: dense_vector length={len(dense_vector)}, sparse_vector keys={list(sparse_vector.keys())}")
                if multivector_vector:
                    logger.debug(f"Chunk {i}: multivector_vector shape={multivector_vector.shape if hasattr(multivector_vector, 'shape') else type(multivector_vector)}")
            
            # Create processed chunk
            processed_chunk = {
                "id": chunk["id"],
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
                "payload": {
                    **chunk.get("payload", {}),
                    "bge_m3_used": True,
                    "embedding_types": list(embeddings.keys()),
                }
            }
            
            # Add multivector if enabled and available
            if multivector_vector and enable_multivector:
                processed_chunk["multivector_vector"] = multivector_vector
            
            processed_chunks.append(processed_chunk)
            
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(processed_chunks)} chunks")
    return processed_chunks


async def upsert_hybrid_chunks_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    batch_size: int = 100,
    enable_multivector: bool = False,
    multivector_count: int = 16,
    multivector_dimension: int = 128,
) -> None:
    """
    Upsert hybrid chunks to Qdrant with retry logic
    Chunks should already be processed with embeddings before calling this function

    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        chunks: List of chunks to upsert (should already contain embeddings)
        batch_size: Batch size for upserting
        enable_multivector: Enable multi-vector (ColBERT) support
        multivector_count: Number of vectors in multi-vector embedding
        multivector_dimension: Dimension of each vector in multi-vector embedding
    """
    logger.info(f"Upserting {len(chunks)} hybrid chunks to collection '{collection_name}'")

    # Debug: Check first chunk structure
    if chunks:
        logger.debug(f"DEBUG: First chunk structure: {list(chunks[0].keys())}")
        logger.debug(f"DEBUG: First chunk dense_vector: {type(chunks[0].get('dense_vector', []))}, length: {len(chunks[0].get('dense_vector', []))}")
        logger.debug(f"DEBUG: First chunk sparse_vector: {type(chunks[0].get('sparse_vector', {}))}, keys: {list(chunks[0].get('sparse_vector', {}).keys())}")
        logger.debug(f"DEBUG: First chunk payload: {type(chunks[0].get('payload', {}))}, keys: {list(chunks[0].get('payload', {}).keys())}")

    # Create PointStruct objects
    points = []
    for i, chunk in enumerate(chunks):
        # Get vectors with fallback values
        dense_vector = chunk.get("dense_vector", [])
        sparse_vector = chunk.get("sparse_vector", {})
        multivector_vector = chunk.get("multivector_vector") if enable_multivector else None
        
        # Debug: Log vector info for first few chunks
        if i < 3:
            logger.debug(f"DEBUG: Chunk {i}: dense_vector length={len(dense_vector)}, sparse_vector keys={list(sparse_vector.keys())}")
            logger.debug(f"DEBUG: Chunk {i}: chunk type={chunk.get('type')}, page={chunk.get('page_number')}")
        
        try:
            point = create_hybrid_point(
                point_id=chunk["id"],
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                payload=chunk["payload"],
                multivector_vector=multivector_vector,
            )
            points.append(point)
            logger.debug(f"DEBUG: Successfully created point {chunk['id']}")
        except Exception as e:
            logger.error(f"DEBUG: Error creating point {chunk['id']}: {e}")
            logger.error(f"DEBUG: Problematic chunk data: {chunk}")
            # Skip problematic chunk but continue processing
            continue

    # Debug: Log final points structure
    if points:
        logger.debug(f"DEBUG: Created {len(points)} PointStruct objects")
        if points[0].vector:
            logger.debug(f"DEBUG: First point vector keys: {list(points[0].vector.keys())}")
            logger.debug(f"DEBUG: First point vector types: {[(k, type(v)) for k, v in points[0].vector.items()]}")
        else:
            logger.debug(f"DEBUG: First point has no vector")
        
        if points[0].payload:
            logger.debug(f"DEBUG: First point payload keys: {list(points[0].payload.keys())}")
        else:
            logger.debug(f"DEBUG: First point has no payload")
    else:
        logger.error(f"DEBUG: No points created from {len(chunks)} chunks")

    # Use existing upsert function
    if points:
        await upsert_with_retry(qdrant_client, collection_name, points, batch_size)
    else:
        logger.error("DEBUG: No points to upsert")


def create_hybrid_point(
    point_id: int,
    dense_vector: List[float],
    sparse_vector: Dict[str, float],
    payload: Dict[str, Any],
    multivector_vector: Optional[Any] = None,
) -> models.PointStruct:
    """
    Create a hybrid point with dense, sparse, and optionally multi-vector vectors

    Args:
        point_id: ID of the point
        dense_vector: Dense vector embedding
        sparse_vector: Sparse vector embedding
        payload: Payload data
        multivector_vector: Multi-vector embedding (optional)

    Returns:
        PointStruct object
    """
    # Debug: Log input parameters
    logger.debug(f"DEBUG: Creating point {point_id}")
    logger.debug(f"DEBUG: dense_vector type: {type(dense_vector)}, length: {len(dense_vector) if dense_vector else 0}")
    logger.debug(f"DEBUG: sparse_vector type: {type(sparse_vector)}, keys: {list(sparse_vector.keys()) if sparse_vector else []}")
    logger.debug(f"DEBUG: multivector_vector type: {type(multivector_vector)}")
    
    # Create vector dictionary
    vector = {}
    
    if dense_vector:
        vector["dense"] = dense_vector
        logger.debug(f"DEBUG: Added dense vector with {len(dense_vector)} dimensions")
    
    if sparse_vector:
        # Convert sparse vector to Qdrant format if needed
        sparse_vector_qdrant = []
        for index, value in sparse_vector.items():
            try:
                sparse_vector_qdrant.append([int(index), float(value)])
            except (ValueError, TypeError) as e:
                logger.warning(f"DEBUG: Invalid sparse vector entry {index}: {value}, error: {e}")
                continue
        
        # Sort by index for consistency
        sparse_vector_qdrant.sort(key=lambda x: x[0])
        vector["sparse"] = sparse_vector_qdrant
        logger.debug(f"DEBUG: Added sparse vector with {len(sparse_vector_qdrant)} entries")
    
    if multivector_vector:
        vector["multivector"] = multivector_vector
        logger.debug(f"DEBUG: Added multivector")
    
    # Debug: Log final vector structure
    logger.debug(f"DEBUG: Final vector structure: {vector}")
    logger.debug(f"DEBUG: Vector keys: {list(vector.keys())}")
    logger.debug(f"DEBUG: Point {point_id} payload keys: {list(payload.keys())}")
    
    # Validate vector structure before creating PointStruct
    try:
        # Test if vector structure is valid by creating a simple validation
        test_vector = vector.copy()
        logger.debug(f"DEBUG: Testing vector structure: {test_vector}")
        
        return models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )
    except Exception as e:
        logger.error(f"DEBUG: Error creating PointStruct: {e}")
        logger.error(f"DEBUG: Problematic vector: {vector}")
        raise



def create_payload_filter(**kwargs) -> models.Filter:
    """
    Create a Qdrant filter from payload key-value pairs
    
    Args:
        **kwargs: Key-value pairs to filter by
        
    Returns:
        Qdrant Filter object
    """
    conditions = []
    
    for key, value in kwargs.items():
        if value is not None:
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
    
    return models.Filter(must=conditions)


async def create_image_collection_if_not_exists(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    vector_size: int = 512,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """
    Create an image collection with CLIP vector support if it doesn't exist
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        vector_size: Size of the CLIP vectors
        distance: Distance metric for vectors
    """
    try:
        # Check if collection exists
        try:
            collections = await qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            logger.info(f"Available collections: {collection_names}")
        except Exception as e:
            logger.warning(f"Error checking collections: {e}")
            collection_names = []
        
        # Check if collection exists using get_collection
        try:
            existing_collection = await qdrant_client.get_collection(collection_name)
            logger.info(f"Image collection '{collection_name}' already exists")
            return
        except Exception as e:
            logger.info(f"Image collection '{collection_name}' does not exist, creating it: {e}")

        if collection_name not in collection_names:
            logger.info(f"Creating image collection '{collection_name}' with CLIP vectors")
            
            # Configure CLIP vectors
            vector_config = models.VectorParams(
                size=vector_size,
                distance=distance,
            )
            
            logger.info(f"Creating image collection with vector config: {vector_config}")
            
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"clip": vector_config},
            )
            
            logger.info(f"Successfully created image collection '{collection_name}'")

    except Exception as e:
        logger.error(f"Error creating image collection '{collection_name}': {e}")
        raise


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
        embeddings: List of embeddings to upsert
        batch_size: Batch size for upserting
    """
    logger.info(f"Upserting {len(embeddings)} image embeddings to collection '{collection_name}'")
    
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



def convert_bge_m3_sparse_to_qdrant_format(sparse_vector: Dict[str, float]) -> List[List[int]]:
    """
    Convert BGE-M3 sparse vector format to Qdrant format
    
    Args:
        sparse_vector: Sparse vector in BGE-M3 format {index: value}
        
    Returns:
        Sparse vector in Qdrant format [[index, value], ...]
    """
    # Convert sparse vector from dict {index: value} to list [index, value] pairs
    sparse_vector_list = []
    for index, value in sparse_vector.items():
        try:
            sparse_vector_list.append([int(index), float(value)])
        except (ValueError, TypeError):
            # Skip invalid entries
            continue
    # Sort by index for consistency
    sparse_vector_list.sort(key=lambda x: x[0])
    return sparse_vector_list

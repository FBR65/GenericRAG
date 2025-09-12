"""
State management for the application
"""
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as http_models
from src.app.settings import Settings


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """
    Create Qdrant client with configuration
    
    Args:
        settings: Application settings
        
    Returns:
        Configured Qdrant client
    """
    return AsyncQdrantClient(
        url=settings.qdrant.qdrant_url,
        api_key=settings.qdrant.qdrant_api_key or None,
        timeout=30,
        prefer_grpc=False,
    )


async def initialize_qdrant_collection(
    qdrant_client: AsyncQdrantClient,
    collection_name: str = "generic_rag_collection",
    vector_size: int = 768,
) -> None:
    """
    Initialize Qdrant collection with proper configuration
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        vector_size: Size of the vectors
    """
    try:
        # Check if collection exists
        collections = await qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if collection_name not in collection_names:
            # Create collection with basic configuration
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=http_models.VectorParams(
                    size=vector_size,
                    distance=http_models.Distance.COSINE,
                ),
            )
            
            print(f"Created collection '{collection_name}'")
        else:
            print(f"Collection '{collection_name}' already exists")
            
    except Exception as e:
        print(f"Error initializing Qdrant collection: {e}")
        raise
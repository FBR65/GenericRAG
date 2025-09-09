import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, host: str, port: int, api_key: Optional[str] = None, 
                 collection_name: str = "generic_docs"):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name
        self.client = None
        self._initialize_client()
        self._ensure_collection()
    
    def _initialize_client(self):
        """Initialize Qdrant client."""
        try:
            # Prepare client configuration
            client_config = {
                "host": self.host,
                "port": self.port,
            }
            
            # Add API key if provided
            if self.api_key:
                client_config["api_key"] = self.api_key
                # Note: Qdrant client automatically handles HTTPS when API key is provided
                # For explicit HTTPS control, you can add: https=True
            else:
                # For local development without API key
                client_config["https"] = False
            
            self.client = QdrantClient(**client_config)
            
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    
    def _ensure_collection(self):
        """Ensure the collection exists with cosine distance."""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1024,  # ColPali embedding size
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection '{self.collection_name}' with cosine distance")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {str(e)}")
            raise
    
    def store_document_page(self, filename: str, page_number: int, 
                           embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """
        Store a document page embedding in Qdrant.
        
        Args:
            filename: Name of the original PDF file
            page_number: Page number (1-based)
            embedding: List of embedding values
            metadata: Additional metadata
            
        Returns:
            Point ID
        """
        try:
            point_id = str(uuid.uuid4())
            
            # Prepare metadata
            default_metadata = {
                "filename": filename,
                "page_number": page_number,
                "timestamp": datetime.now().isoformat(),
                "document_id": f"{filename}_page_{page_number}"
            }
            
            if metadata:
                default_metadata.update(metadata)
            
            # Store the point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=default_metadata
                )]
            )
            
            logger.info(f"Stored embedding for {filename}, page {page_number}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to store document page: {str(e)}")
            raise
    
    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                    "distance": hit.distance
                })
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {str(e)}")
            raise
    
    def get_document_pages(self, filename: str) -> List[Dict[str, Any]]:
        """
        Get all pages for a specific document.
        
        Args:
            filename: Name of the document
            
        Returns:
            List of page information
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=1000,  # Adjust based on expected max pages
                with_payload=True
            )
            
            pages = []
            for point in results[0]:
                pages.append({
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector
                })
            
            # Sort by page number
            pages.sort(key=lambda x: x["payload"]["page_number"])
            
            logger.info(f"Retrieved {len(pages)} pages for {filename}")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to get document pages: {str(e)}")
            raise
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete all pages of a document.
        
        Args:
            filename: Name of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all points for this document
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=1000
            )
            
            if not results[0]:
                logger.warning(f"No pages found for document {filename}")
                return False
            
            # Delete all points
            point_ids = [point.id for point in results[0]]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            
            logger.info(f"Deleted {len(point_ids)} pages for {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            raise
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the collection.
        
        Returns:
            List of document information
        """
        try:
            # Get all unique filenames
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )
            
            documents = {}
            for point in results[0]:
                filename = point.payload["filename"]
                if filename not in documents:
                    documents[filename] = {
                        "filename": filename,
                        "page_count": 0,
                        "first_seen": point.payload["timestamp"],
                        "total_size": 0
                    }
                
                documents[filename]["page_count"] += 1
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = {
                "collection_name": self.collection_name,
                "vector_count": collection_info.status.points_count,
                "status": collection_info.status,
                "config": collection_info.config_params
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            raise
    
    def clear_collection(self) -> bool:
        """
        Clear all points from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info("Cleared collection and recreated it")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise
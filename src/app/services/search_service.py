"""
Search service for hybrid search functionality
"""

import asyncio
from typing import List, Optional, Dict, Any
from loguru import logger

from src.app.utils.qdrant_utils import (
    hybrid_search_with_metadata,
    search_images_with_text_context,
    combine_and_rank_results,
    create_payload_filter,
)
from src.app.models.schemas import SearchResult
from qdrant_client import AsyncQdrantClient
from PIL import Image


class SearchService:
    """Service for performing hybrid search operations"""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        image_storage,
        settings,
    ):
        """
        Initialize the search service

        Args:
            qdrant_client: Qdrant client instance
            image_storage: Image storage service
            settings: Application settings
        """
        self.qdrant_client = qdrant_client
        self.image_storage = image_storage
        self.settings = settings

        # Collection names
        self.text_collection_name = settings.qdrant.collection_name
        self.image_collection_name = f"{settings.qdrant.collection_name}_images"

        logger.info("Initialized SearchService")

    async def perform_hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        sparse_vector: Dict[str, float],
        search_strategy: str = "hybrid",
        alpha: float = 0.5,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        include_images: bool = True,
        session_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> List[SearchResult]:
        """
        Perform hybrid search with multiple strategies

        Args:
            query: Original query text
            query_embedding: Dense query embedding
            sparse_vector: Sparse query vector
            search_strategy: Search strategy ("text_only", "image_only", "hybrid")
            alpha: Weight for dense vs sparse search
            top_k: Number of results to return
            score_threshold: Minimum score threshold
            metadata_filters: Additional metadata filters
            include_images: Whether to include image results
            session_id: Session identifier for filtering
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            List of search results
        """
        logger.info(f"Performing {search_strategy} search for query: {query}")

        try:
            # Get base filter
            base_filter = create_payload_filter(session_id=session_id)

            # Perform text search
            text_results = []
            if search_strategy in ["text_only", "hybrid"]:
                logger.info("Performing text search")
                text_results = await self._search_text(
                    query_embedding=query_embedding,
                    sparse_vector=sparse_vector,
                    alpha=alpha,
                    limit=top_k * 2,  # Get more for combination
                    score_threshold=score_threshold,
                    query_filter=base_filter,
                    metadata_filters=metadata_filters,
                )

            # Perform image search
            image_results = []
            if search_strategy in ["image_only", "hybrid"] and include_images:
                logger.info("Performing image search")
                image_results = await self._search_images(
                    query_embedding=query_embedding,
                    text_query=query,
                    limit=top_k * 2,  # Get more for combination
                    score_threshold=score_threshold,
                    query_filter=base_filter,
                    metadata_filters=metadata_filters,
                )

            # Combine and rank results
            combined_results = await self._combine_and_rank_results(
                text_results=text_results,
                image_results=image_results,
                strategy=search_strategy,
                text_weight=0.7,
                image_weight=0.3,
            )

            # Apply pagination
            paginated_results = self._apply_pagination(
                combined_results, page, page_size
            )

            # Format results
            search_results = await self._format_search_results(paginated_results)

            logger.info(f"Found {len(search_results)} search results")
            return search_results

        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise

    async def _search_text(
        self,
        query_embedding: List[float],
        sparse_vector: Dict[str, float],
        alpha: float = 0.5,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Any] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search in text collection

        Args:
            query_embedding: Dense query embedding
            sparse_vector: Sparse query vector
            alpha: Weight for dense vs sparse search
            limit: Number of results to return
            score_threshold: Minimum score threshold
            query_filter: Base filter to apply
            metadata_filters: Additional metadata filters

        Returns:
            List of text search results
        """
        try:
            response = await hybrid_search_with_metadata(
                qdrant_client=self.qdrant_client,
                collection_name=self.text_collection_name,
                dense_vector=query_embedding,
                sparse_vector=sparse_vector,
                alpha=alpha,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                metadata_filters=metadata_filters,
            )

            # Convert to dictionary format
            results = []
            for point in response.points:
                results.append(
                    {
                        "id": point.id,
                        "score": point.score,
                        "payload": point.payload or {},
                        "search_type": "text",
                    }
                )

            logger.info(f"Found {len(results)} text results")
            return results

        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return []

    async def _search_images(
        self,
        query_embedding: List[float],
        text_query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Any] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search in image collection

        Args:
            query_embedding: Image query embedding
            text_query: Text query for context
            limit: Number of results to return
            score_threshold: Minimum score threshold
            query_filter: Base filter to apply
            metadata_filters: Additional metadata filters

        Returns:
            List of image search results
        """
        try:
            response = await search_images_with_text_context(
                qdrant_client=self.qdrant_client,
                image_collection_name=self.image_collection_name,
                text_collection_name=self.text_collection_name,
                query_vector=query_embedding,
                text_query=text_query,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

            # Convert to dictionary format
            results = []
            for point in response.points:
                results.append(
                    {
                        "id": point.id,
                        "score": point.score,
                        "payload": point.payload or {},
                        "search_type": "image",
                    }
                )

            logger.info(f"Found {len(results)} image results")
            return results

        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []

    async def _combine_and_rank_results(
        self,
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
            strategy: Combination strategy
            text_weight: Weight for text results
            image_weight: Weight for image results

        Returns:
            Combined and ranked results
        """
        try:
            combined_results = combine_and_rank_results(
                text_results=text_results,
                image_results=image_results,
                strategy=strategy,
                text_weight=text_weight,
                image_weight=image_weight,
            )

            logger.info(f"Combined and ranked {len(combined_results)} results")
            return combined_results

        except Exception as e:
            logger.error(f"Error combining and ranking results: {e}")
            return []

    def _apply_pagination(
        self,
        results: List[Dict[str, Any]],
        page: int,
        page_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Apply pagination to results

        Args:
            results: Full list of results
            page: Page number (1-based)
            page_size: Number of results per page

        Returns:
            Paginated results
        """
        try:
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_results = results[start_idx:end_idx]

            logger.info(
                f"Applied pagination: page {page}, size {page_size}, returning {len(paginated_results)} results"
            )
            return paginated_results

        except Exception as e:
            logger.error(f"Error applying pagination: {e}")
            return []

    async def _format_search_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[SearchResult]:
        """
        Format search results for API response

        Args:
            results: Raw search results

        Returns:
            Formatted search results
        """
        try:
            search_results = []

            for result in results:
                payload = result["payload"]

                # Load image if available
                image = None
                if result["search_type"] == "image" and "image_path" in payload:
                    try:
                        images = await self.image_storage.load_images(
                            [payload["image_path"]]
                        )
                        if images:
                            image = images[0]
                    except Exception as e:
                        logger.warning(
                            f"Could not load image {payload['image_path']}: {e}"
                        )

                # Create search result
                search_result = SearchResult(
                    id=result["id"],
                    score=result["score"],
                    document=payload.get("document", ""),
                    page=payload.get("page", 0),
                    image=image,
                    metadata={
                        "session_id": payload.get("session_id"),
                        "created_at": payload.get("created_at"),
                        "search_type": result["search_type"],
                        "combined_score": result.get("combined_score", result["score"]),
                        "related_text": payload.get("related_text", []),
                        "bbox": payload.get("bbox"),
                        "element_type": payload.get("element_type"),
                    },
                )

                search_results.append(search_result)

            logger.info(f"Formatted {len(search_results)} search results")
            return search_results

        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            return []

    async def filter_by_metadata(
        self,
        results: List[SearchResult],
        metadata_filters: Dict[str, Any],
    ) -> List[SearchResult]:
        """
        Filter search results by metadata

        Args:
            results: Search results to filter
            metadata_filters: Metadata filters to apply

        Returns:
            Filtered search results
        """
        try:
            filtered_results = []

            for result in results:
                include = True

                # Bbox filter
                if "bbox" in metadata_filters:
                    bbox_filter = metadata_filters["bbox"]
                    result_bbox = result.metadata.get("bbox")
                    if result_bbox and result_bbox != bbox_filter:
                        include = False

                # Page number filter
                if "page_number" in metadata_filters:
                    page_filter = metadata_filters["page_number"]
                    if result.page != page_filter:
                        include = False

                # Type filter
                if "type" in metadata_filters:
                    type_filter = metadata_filters["type"]
                    result_type = result.metadata.get("search_type")
                    if result_type and result_type not in type_filter:
                        include = False

                if include:
                    filtered_results.append(result)

            logger.info(f"Filtered {len(results)} results to {len(filtered_results)}")
            return filtered_results

        except Exception as e:
            logger.error(f"Error filtering by metadata: {e}")
            return results

    async def rank_results(
        self,
        results: List[SearchResult],
        strategy: str = "combined",
    ) -> List[SearchResult]:
        """
        Rank search results by different strategies

        Args:
            results: Search results to rank
            strategy: Ranking strategy ("combined", "score", "relevance")

        Returns:
            Ranked search results
        """
        try:
            if strategy == "combined":
                # Sort by combined score
                results.sort(
                    key=lambda x: x.metadata.get("combined_score", x.score),
                    reverse=True,
                )
            elif strategy == "score":
                # Sort by original score
                results.sort(key=lambda x: x.score, reverse=True)
            elif strategy == "relevance":
                # Sort by relevance (could be more complex in future)
                results.sort(key=lambda x: x.score, reverse=True)

            logger.info(f"Ranked {len(results)} results using strategy '{strategy}'")
            return results

        except Exception as e:
            logger.error(f"Error ranking results: {e}")
            return results

    async def format_search_results(
        self,
        results: List[SearchResult],
        include_metadata: bool = True,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        """
        Format search results for different output formats

        Args:
            results: Search results to format
            include_metadata: Whether to include metadata
            include_images: Whether to include images

        Returns:
            Formatted results dictionary
        """
        try:
            formatted_results = {
                "total_results": len(results),
                "results": [],
                "summary": {
                    "text_results": 0,
                    "image_results": 0,
                    "avg_score": 0.0,
                },
            }

            total_score = 0.0

            for result in results:
                result_dict = {
                    "id": result.id,
                    "score": result.score,
                    "document": result.document,
                    "page": result.page,
                }

                if include_metadata:
                    result_dict["metadata"] = result.metadata

                if include_images and result.image:
                    result_dict["image"] = result.image

                formatted_results["results"].append(result_dict)

                # Update summary
                if result.metadata.get("search_type") == "text":
                    formatted_results["summary"]["text_results"] += 1
                else:
                    formatted_results["summary"]["image_results"] += 1

                total_score += result.score

            # Calculate average score
            if results:
                formatted_results["summary"]["avg_score"] = total_score / len(results)

            logger.info(f"Formatted {len(results)} results for output")
            return formatted_results

        except Exception as e:
            logger.error(f"Error formatting search results for output: {e}")
            return {"total_results": 0, "results": [], "summary": {}}

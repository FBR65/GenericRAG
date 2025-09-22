"""
Search service for hybrid search functionality with BGE-M3 support
"""

import asyncio
import aiohttp
from typing import List, Optional, Dict, Any, Union
from loguru import logger

from src.app.utils.qdrant_utils import (
    hybrid_search_with_metadata,
    search_images_with_text_context,
    combine_and_rank_results,
    create_payload_filter,
    bge_m3_hybrid_search_with_retry,
    prepare_bge_m3_query_embeddings,
    format_bge_m3_search_results,
    convert_bge_m3_sparse_to_qdrant_format,
)
from src.app.models.schemas import SearchResult
from qdrant_client import AsyncQdrantClient
from PIL import Image
from qdrant_client import models

# Import BGE-M3 Service
from src.app.services.bge_m3_service import BGE_M3_Service


class SearchService:
    """Service for performing hybrid search operations with BGE-M3 support"""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        image_storage,
        settings,
    ):
        """
        Initialize the search service with BGE-M3 support

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

        # BGE-M3 Service initialization
        self.bge_m3_service = None
        self._initialize_bge_m3_service()

        logger.info("Initialized SearchService with BGE-M3 support")

    def _initialize_bge_m3_service(self):
        """Initialize BGE-M3 service if enabled in settings"""
        try:
            # Überprüfe, ob BGE-M3 in den Settings aktiviert ist
            if hasattr(self.settings, 'bge_m3') and self.settings.bge_m3.model_name:
                self.bge_m3_service = BGE_M3_Service(self.settings)
                logger.info("BGE-M3 service initialized successfully")
            else:
                logger.info("BGE-M3 service disabled in settings")
        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 service: {e}")
            self.bge_m3_service = None

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
        Search in text collection with BGE-M3 support

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
            # Konvertiere BGE-M3 Sparse Format zu Qdrant Format, wenn nötig
            qdrant_sparse_vector = sparse_vector
            if sparse_vector and isinstance(next(iter(sparse_vector.keys())), int):
                qdrant_sparse_vector = convert_bge_m3_sparse_to_qdrant_format(sparse_vector)
            
            response = await hybrid_search_with_metadata(
                qdrant_client=self.qdrant_client,
                collection_name=self.text_collection_name,
                dense_vector=query_embedding,
                sparse_vector=qdrant_sparse_vector,
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

    async def get_dense_embedding(self, text: str, use_bge_m3: bool = None) -> List[float]:
        """
        Generiert Dense Embeddings für einen Text unter Verwendung der konfigurierten Settings
        Unterstützt BGE-M3 als primäre Quelle, wenn verfügbar und aktiviert
        
        Args:
            text: Text für den Embedding-Vektor
            use_bge_m3: Optionaler Parameter zur expliziten Auswahl von BGE-M3
            
        Returns:
            Dense Embedding-Vektor als Liste von Floats
        """
        try:
            # Bestimme, ob BGE-M3 verwendet werden soll
            if use_bge_m3 is None:
                use_bge_m3 = self.bge_m3_service is not None
            
            # Verwende BGE-M3 Service, wenn verfügbar und aktiviert
            if use_bge_m3 and self.bge_m3_service:
                logger.info(f"Generating dense embedding using BGE-M3 for text: {text[:50]}...")
                return await self.bge_m3_service.generate_dense_embedding(text)
            
            # Fallback zur ursprünglichen API-Anfrage
            logger.info(f"Generating dense embedding using API for text: {text[:50]}...")
            payload = {
                "model": self.settings.qdrant.dense_model,
                "prompt": text,
                "options": {"temperature": 0, "num_ctx": 2048},
            }
            
            headers = {"Content-Type": "application/json"}
            
            # Füge API Key hinzu, wenn vorhanden
            if self.settings.qdrant.dense_api_key:
                headers["Authorization"] = f"Bearer {self.settings.qdrant.dense_api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.settings.qdrant.dense_base_url}/v1/embeddings",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Standardformat für Embeddings
                        if "data" in result and len(result["data"]) > 0:
                            return result["data"][0]["embedding"]
                        else:
                            logger.warning(
                                "Kein Embedding in der Antwort gefunden, verwende Dummy-Vektor"
                            )
                            # Fallback: generiere einen Dummy-Vektor basierend auf der Dimension
                            return [0.0] * self.settings.qdrant.dense_dimension
                    else:
                        logger.error(
                            f"Fehler bei der Embedding-Anfrage: {response.status}"
                        )
                        # Fallback: generiere einen Dummy-Vektor
                        return [0.0] * self.settings.qdrant.dense_dimension
                        
        except Exception as e:
            logger.error(f"Fehler bei der Embedding-Erzeugung: {e}")
            # Fallback: generiere einen Dummy-Vektor
            return [0.0] * self.settings.qdrant.dense_dimension

    async def get_bge_m3_embeddings(self, text: str) -> Dict[str, Any]:
        """
        Generiert alle drei Embedding-Typen (Dense, Sparse, Multi-Vector) mit BGE-M3
        
        Args:
            text: Text für die Embedding-Generierung
            
        Returns:
            Dictionary mit allen drei Embedding-Typen und Metadaten
        """
        try:
            if not self.bge_m3_service:
                logger.error("BGE-M3 service not available")
                return {
                    "dense": [],
                    "sparse": {},
                    "multivector": [],
                    "errors": ["BGE-M3 service not available"],
                    "success": False
                }
            
            logger.info(f"Generating all BGE-M3 embeddings for text: {text[:50]}...")
            
            # Nutze den BGE-M3 Service für effiziente Generierung aller drei Typen
            result = await self.bge_m3_service.generate_embeddings(text)
            
            # Formatieren der Ergebnisse
            embeddings = {
                "dense": result["embeddings"].get("dense", []),
                "sparse": result["embeddings"].get("sparse", {}),
                "multivector": result["embeddings"].get("multivector", []),
                "errors": result["errors"],
                "success": len(result["errors"]) == 0,
                "text": result["text"]
            }
            
            logger.info(f"Generated BGE-M3 embeddings: dense={len(embeddings['dense'])}, "
                       f"sparse={len(embeddings['sparse'])}, multivector={len(embeddings['multivector'])}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating BGE-M3 embeddings: {e}")
            return {
                "dense": [],
                "sparse": {},
                "multivector": [],
                "errors": [str(e)],
                "success": False,
                "text": text[:100] + "..." if len(text) > 100 else text
            }

    async def get_bge_m3_embeddings_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch-Verarbeitung für BGE-M3 Embeddings
        
        Args:
            texts: Liste von Texten für die Embedding-Generierung
            
        Returns:
            Liste von Embedding-Ergebnissen
        """
        try:
            if not self.bge_m3_service:
                logger.error("BGE-M3 service not available")
                return [{
                    "dense": [],
                    "sparse": {},
                    "multivector": [],
                    "errors": ["BGE-M3 service not available"],
                    "success": False,
                    "text": text[:100] + "..." if len(text) > 100 else text
                } for text in texts]
            
            logger.info(f"Processing batch of {len(texts)} texts for BGE-M3 embeddings")
            
            # Nutze den BGE-M3 Service für Batch-Verarbeitung
            results = await self.bge_m3_service.batch_generate_embeddings(texts)
            
            # Formatieren der Ergebnisse
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "dense": result["embeddings"].get("dense", []),
                    "sparse": result["embeddings"].get("sparse", {}),
                    "multivector": result["embeddings"].get("multivector", []),
                    "errors": result["errors"],
                    "success": len(result["errors"]) == 0,
                    "text": result["text"]
                })
            
            success_count = sum(1 for r in formatted_results if r["success"])
            logger.info(f"Batch processing completed: {success_count}/{len(texts)} successful")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in BGE-M3 batch processing: {e}")
            return [{
                "dense": [],
                "sparse": {},
                "multivector": [],
                "errors": [str(e)],
                "success": False,
                "text": text[:100] + "..." if len(text) > 100 else text
            } for text in texts]

    async def bge_m3_hybrid_search(
        self,
        query: str,
        search_strategy: str = "full_bge_m3",
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        include_images: bool = True,
        session_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> List[SearchResult]:
        """
        Führt hybride Suche mit BGE-M3 durch mit drei-Phasen-Suche
        
        Args:
            query: Suchanfrage
            search_strategy: Suchstrategie ("dense_only", "sparse_only", "full_bge_m3")
            alpha: Gewicht für Dense-Suche (0.0-1.0)
            beta: Gewicht für Sparse-Suche (0.0-1.0)
            gamma: Gewicht für Multi-Vector Reranking (0.0-1.0)
            top_k: Anzahl der Ergebnisse
            score_threshold: Mindest-Score-Schwelle
            metadata_filters: Metadaten-Filter
            include_images: Bilder in Ergebnissen einschließen
            session_id: Session-ID für Filterung
            page: Seitennummer für Paginierung
            page_size: Ergebnisse pro Seite
            
        Returns:
            Liste formatierter Suchergebnisse
        """
        try:
            logger.info(f"Performing BGE-M3 hybrid search for query: {query}")
            
            # Vorbereiten der BGE-M3 Embeddings
            if search_strategy == "dense_only":
                embeddings = await self.get_bge_m3_embeddings(query)
                dense_vector = embeddings["dense"]
                sparse_vector = {}
                multivector_query = None
            elif search_strategy == "sparse_only":
                embeddings = await self.get_bge_m3_embeddings(query)
                dense_vector = []
                sparse_vector = embeddings["sparse"]
                multivector_query = None
            else:  # full_bge_m3
                embeddings = await self.get_bge_m3_embeddings(query)
                dense_vector = embeddings["dense"]
                sparse_vector = embeddings["sparse"]
                multivector_query = embeddings["multivector"]
            
            # Erstellen des Basis-Filters
            base_filter = create_payload_filter(session_id=session_id)
            
            # Führe BGE-M3 hybride Suche durch
            search_response = await bge_m3_hybrid_search_with_retry(
                qdrant_client=self.qdrant_client,
                collection_name=self.text_collection_name,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                alpha=alpha,
                limit=top_k,
                query_filter=base_filter,
                score_threshold=score_threshold,
                enable_multivector_reranking=multivector_query is not None,
                multivector_query=multivector_query,
                multivector_weight=gamma,
            )
            
            # Formatieren der Ergebnisse
            formatted_results = format_bge_m3_search_results(
                search_response,
                include_scores=True,
                include_payload=True,
                format_type="detailed"
            )
            
            # Konvertiere in SearchResult-Format
            search_results = []
            for result in formatted_results:
                # Erstelle SearchResult-Objekt
                search_result = SearchResult(
                    id=result["id"],
                    score=result.get("score", 0.0),
                    document=result.get("payload", {}).get("document", ""),
                    page=result.get("payload", {}).get("page", 0),
                    image=None,  # Wird später hinzugefügt
                    metadata={
                        "session_id": result.get("payload", {}).get("session_id"),
                        "created_at": result.get("payload", {}).get("created_at"),
                        "search_type": "bge_m3_hybrid",
                        "confidence": result.get("confidence", "low"),
                        "vector_types": result.get("payload", {}).get("vector_types", []),
                        "element_type": result.get("payload", {}).get("element_type"),
                        "document": result.get("payload", {}).get("document"),
                        "page": result.get("payload", {}).get("page"),
                    },
                )
                search_results.append(search_result)
            
            # Wende Paginierung an
            paginated_results = self._apply_pagination(search_results, page, page_size)
            
            logger.info(f"BGE-M3 hybrid search found {len(paginated_results)} results")
            return paginated_results
            
        except Exception as e:
            logger.error(f"Error in BGE-M3 hybrid search: {e}")
            return []

    async def bge_m3_multivector_search(
        self,
        query: str,
        strategy: str = "max_sim",
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> List[SearchResult]:
        """
        Spezielle Suche für Multi-Vector Embeddings (ColBERT)
        
        Args:
            query: Suchanfrage
            strategy: Suchstrategie ("max_sim", "mean_sim", "max_mean")
            top_k: Anzahl der Ergebnisse
            score_threshold: Mindest-Score-Schwelle
            metadata_filters: Metadaten-Filter
            session_id: Session-ID für Filterung
            page: Seitennummer für Paginierung
            page_size: Ergebnisse pro Seite
            
        Returns:
            Liste formatierter Suchergebnisse
        """
        try:
            logger.info(f"Performing BGE-M3 multivector search with strategy: {strategy}")
            
            # Generiere Multi-Vector Embeddings
            embeddings = await self.get_bge_m3_embeddings(query)
            multivector_query = embeddings["multivector"]
            
            if not multivector_query:
                logger.warning("No multivector embeddings generated")
                return []
            
            # Erstellen des Basis-Filters
            base_filter = create_payload_filter(session_id=session_id)
            
            # Führe Multi-Vector Suche durch
            search_response = await self.qdrant_client.query_points(
                collection_name=self.text_collection_name,
                query=models.Vector(
                    name="multivector",
                    vector=multivector_query[0] if multivector_query else [],
                ),
                limit=top_k,
                query_filter=base_filter,
                with_payload=True,
                score_threshold=score_threshold,
            )
            
            # Formatieren der Ergebnisse
            formatted_results = format_bge_m3_search_results(
                search_response,
                include_scores=True,
                include_payload=True,
                format_type="detailed"
            )
            
            # Konvertiere in SearchResult-Format
            search_results = []
            for result in formatted_results:
                # Erstelle SearchResult-Objekt
                search_result = SearchResult(
                    id=result["id"],
                    score=result.get("score", 0.0),
                    document=result.get("payload", {}).get("document", ""),
                    page=result.get("payload", {}).get("page", 0),
                    image=None,  # Wird später hinzugefügt
                    metadata={
                        "session_id": result.get("payload", {}).get("session_id"),
                        "created_at": result.get("payload", {}).get("created_at"),
                        "search_type": "bge_m3_multivector",
                        "confidence": result.get("confidence", "low"),
                        "vector_types": result.get("payload", {}).get("vector_types", []),
                        "element_type": result.get("payload", {}).get("element_type"),
                        "document": result.get("payload", {}).get("document"),
                        "page": result.get("payload", {}).get("page"),
                        "multivector_strategy": strategy,
                    },
                )
                search_results.append(search_result)
            
            # Wende Paginierung an
            paginated_results = self._apply_pagination(search_results, page, page_size)
            
            logger.info(f"BGE-M3 multivector search found {len(paginated_results)} results")
            return paginated_results
            
        except Exception as e:
            logger.error(f"Error in BGE-M3 multivector search: {e}")
            return []

    async def search_text(
        self,
        query: str,
        use_bge_m3: bool = None,
        search_strategy: str = "hybrid",
        alpha: float = 0.5,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> List[SearchResult]:
        """
        Erweiterte Text-Suche mit BGE-M3 Unterstützung
        
        Args:
            query: Suchanfrage
            use_bge_m3: BGE-M3 für Embedding-Generierung verwenden
            search_strategy: Suchstrategie ("hybrid", "dense_only", "sparse_only")
            alpha: Gewicht für Dense vs Sparse Suche
            top_k: Anzahl der Ergebnisse
            score_threshold: Mindest-Score-Schwelle
            metadata_filters: Metadaten-Filter
            session_id: Session-ID für Filterung
            page: Seitennummer für Paginierung
            page_size: Ergebnisse pro Seite
            
        Returns:
            Liste formatierter Suchergebnisse
        """
        try:
            logger.info(f"Performing text search with BGE-M3 support for query: {query}")
            
            # Generiere Embeddings mit BGE-M3, wenn aktiviert
            if use_bge_m3 and self.bge_m3_service:
                embeddings = await self.get_bge_m3_embeddings(query)
                dense_vector = embeddings["dense"]
                sparse_vector = embeddings["sparse"]
            else:
                # Fallback zur ursprünglichen Methode
                dense_vector = await self.get_dense_embedding(query, use_bge_m3=False)
                sparse_vector = {}  # Sparse Vectors werden hier nicht generiert
            
            # Führe Text-Suche durch
            text_results = await self._search_text(
                query_embedding=dense_vector,
                sparse_vector=sparse_vector,
                alpha=alpha,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=create_payload_filter(session_id=session_id),
                metadata_filters=metadata_filters,
            )
            
            # Wende Paginierung an
            paginated_results = self._apply_pagination(text_results, page, page_size)
            
            # Formatiere Ergebnisse
            search_results = await self._format_search_results(paginated_results)
            
            logger.info(f"Text search found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in text search with BGE-M3: {e}")
            return []

    async def search_image(
        self,
        query: str,
        use_bge_m3: bool = None,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> List[SearchResult]:
        """
        Erweiterte Bild-Suche mit BGE-M3 Unterstützung
        
        Args:
            query: Bildsuchanfrage
            use_bge_m3: BGE-M3 für Embedding-Generierung verwenden
            top_k: Anzahl der Ergebnisse
            score_threshold: Mindest-Score-Schwelle
            metadata_filters: Metadaten-Filter
            session_id: Session-ID für Filterung
            page: Seitennummer für Paginierung
            page_size: Ergebnisse pro Seite
            
        Returns:
            Liste formatierter Suchergebnisse
        """
        try:
            logger.info(f"Performing image search with BGE-M3 support for query: {query}")
            
            # Generiere Dense Embedding mit BGE-M3, wenn aktiviert
            if use_bge_m3 and self.bge_m3_service:
                dense_vector = await self.bge_m3_service.generate_dense_embedding(query)
            else:
                # Fallback zur ursprünglichen Methode
                dense_vector = await self.get_dense_embedding(query, use_bge_m3=False)
            
            # Führe Bild-Suche durch
            image_results = await self._search_images(
                query_embedding=dense_vector,
                text_query=query,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=create_payload_filter(session_id=session_id),
                metadata_filters=metadata_filters,
            )
            
            # Wende Paginierung an
            paginated_results = self._apply_pagination(image_results, page, page_size)
            
            # Formatiere Ergebnisse
            search_results = await self._format_search_results(paginated_results)
            
            logger.info(f"Image search found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in image search with BGE-M3: {e}")
            return []

    async def search_hybrid(
        self,
        query: str,
        use_bge_m3: bool = None,
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
        Erweiterte hybride Suche mit BGE-M3 Unterstützung
        
        Args:
            query: Suchanfrage
            use_bge_m3: BGE-M3 für Embedding-Generierung verwenden
            search_strategy: Suchstrategie ("hybrid", "text_only", "image_only")
            alpha: Gewicht für Dense vs Sparse Suche
            top_k: Anzahl der Ergebnisse
            score_threshold: Mindest-Score-Schwelle
            metadata_filters: Metadaten-Filter
            include_images: Bilder in Ergebnissen einschließen
            session_id: Session-ID für Filterung
            page: Seitennummer für Paginierung
            page_size: Ergebnisse pro Seite
            
        Returns:
            Liste formatierter Suchergebnisse
        """
        try:
            logger.info(f"Performing hybrid search with BGE-M3 support for query: {query}")
            
            # Generiere Embeddings mit BGE-M3, wenn aktiviert
            if use_bge_m3 and self.bge_m3_service:
                embeddings = await self.get_bge_m3_embeddings(query)
                dense_vector = embeddings["dense"]
                sparse_vector = embeddings["sparse"]
            else:
                # Fallback zur ursprünglichen Methode
                dense_vector = await self.get_dense_embedding(query, use_bge_m3=False)
                sparse_vector = {}  # Sparse Vectors werden hier nicht generiert
            
            # Führe hybride Suche durch
            search_results = await self.perform_hybrid_search(
                query=query,
                query_embedding=dense_vector,
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
            
            logger.info(f"Hybrid search found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search with BGE-M3: {e}")
            return []

    def format_bge_m3_results(
        self,
        results: List[Dict[str, Any]],
        format_type: str = "standard",
        include_scores: bool = True,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Formatiert BGE-M3 Suchergebnisse für die Ausgabe
        
        Args:
            results: Roh-Suchergebnisse
            format_type: Format-Typ ("standard", "detailed", "compact")
            include_scores: Scores einschließen
            include_metadata: Metadaten einschließen
            
        Returns:
            Formatierte Suchergebnisse
        """
        try:
            formatted_results = []
            
            for result in results:
                formatted_result = {}
                
                # Grundlegende Informationen
                formatted_result["id"] = result.get("id")
                
                # Score-Informationen
                if include_scores:
                    formatted_result["score"] = result.get("score", 0.0)
                    formatted_result["confidence"] = self.calculate_bge_m3_confidence(result.get("score", 0.0))
                
                # Metadaten
                if include_metadata and "payload" in result:
                    payload = result["payload"]
                    formatted_result["metadata"] = {
                        "document": payload.get("document", ""),
                        "page": payload.get("page", 0),
                        "element_type": payload.get("element_type"),
                        "session_id": payload.get("session_id"),
                        "created_at": payload.get("created_at"),
                        "search_type": payload.get("search_type", "bge_m3"),
                        "vector_types": payload.get("vector_types", []),
                    }
                
                # Format-Typ spezifische Anpassungen
                if format_type == "detailed":
                    formatted_result["detailed_info"] = {
                        "vector_completeness": self._calculate_vector_completeness(result),
                        "metadata_quality": self._calculate_metadata_quality(result),
                    }
                elif format_type == "compact":
                    # Nur wesentliche Informationen behalten
                    compact_result = {"id": formatted_result["id"]}
                    if include_scores and "score" in formatted_result:
                        compact_result["score"] = formatted_result["score"]
                    if "metadata" in formatted_result and "document" in formatted_result["metadata"]:
                        compact_result["document"] = formatted_result["metadata"]["document"]
                    formatted_result = compact_result
                
                formatted_results.append(formatted_result)
            
            logger.info(f"Formatted {len(formatted_results)} BGE-M3 results in {format_type} format")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error formatting BGE-M3 results: {e}")
            return []

    def calculate_bge_m3_confidence(self, score: float) -> str:
        """
        Berechnet Vertrauenswerte für BGE-M3 Suchergebnisse
        
        Args:
            score: Suchergebnis-Score
            
        Returns:
            Vertrauensstufe ("high", "medium", "low")
        """
        try:
            if score >= 0.8:
                return "high"
            elif score >= 0.5:
                return "medium"
            else:
                return "low"
        except Exception as e:
            logger.error(f"Error calculating BGE-M3 confidence: {e}")
            return "low"

    async def prepare_bge_m3_documents(
        self,
        documents: List[Dict[str, Any]],
        include_dense: bool = True,
        include_sparse: bool = True,
        include_multivector: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Bereitet Dokumente für BGE-M3 Embeddings vor
        
        Args:
            documents: Liste von Dokumenten
            include_dense: Dense Embeddings generieren
            include_sparse: Sparse Embeddings generieren
            include_multivector: Multi-Vector Embeddings generieren
            
        Returns:
            Liste vorbereiteter Dokumente mit Embeddings
        """
        try:
            prepared_documents = []
            
            for doc in documents:
                prepared_doc = {
                    "id": doc.get("id"),
                    "document": doc.get("document", ""),
                    "page": doc.get("page", 0),
                    "element_type": doc.get("element_type", "text"),
                    "embeddings": {},
                    "errors": [],
                }
                
                # Generiere Embeddings, wenn BGE-M3 Service verfügbar ist
                if self.bge_m3_service:
                    try:
                        if include_dense:
                            prepared_doc["embeddings"]["dense"] = await self.bge_m3_service.generate_dense_embedding(doc["document"])
                        
                        if include_sparse:
                            prepared_doc["embeddings"]["sparse"] = await self.bge_m3_service.generate_sparse_embedding(doc["document"])
                        
                        if include_multivector:
                            prepared_doc["embeddings"]["multivector"] = await self.bge_m3_service.generate_multivector_embedding(doc["document"])
                            
                    except Exception as e:
                        prepared_doc["errors"].append(f"Embedding generation failed: {str(e)}")
                        logger.warning(f"Failed to generate embeddings for document {doc.get('id')}: {e}")
                else:
                    prepared_doc["errors"].append("BGE-M3 service not available")
                
                prepared_documents.append(prepared_doc)
            
            logger.info(f"Prepared {len(prepared_documents)} documents for BGE-M3 embeddings")
            return prepared_documents
            
        except Exception as e:
            logger.error(f"Error preparing BGE-M3 documents: {e}")
            return []

    def _calculate_vector_completeness(self, result: Dict[str, Any]) -> float:
        """
        Berechnet die Vollständigkeit der Vektoren in einem Suchergebnis
        
        Args:
            result: Suchergebnis
            
        Returns:
            Vollständigkeits-Score (0.0-1.0)
        """
        try:
            embeddings = result.get("payload", {}).get("embeddings", {})
            vector_types = ["dense", "sparse", "multivector"]
            
            present_types = sum(1 for vec_type in vector_types if vec_type in embeddings)
            return present_types / len(vector_types)
            
        except Exception as e:
            logger.error(f"Error calculating vector completeness: {e}")
            return 0.0

    def _calculate_metadata_quality(self, result: Dict[str, Any]) -> float:
        """
        Berechnet die Qualität der Metadaten in einem Suchergebnis
        
        Args:
            result: Suchergebnis
            
        Returns:
            Metadaten-Qualitätsscore (0.0-1.0)
        """
        try:
            payload = result.get("payload", {})
            required_fields = ["document", "page", "element_type"]
            
            present_fields = sum(1 for field in required_fields if field in payload)
            return present_fields / len(required_fields)
            
        except Exception as e:
            logger.error(f"Error calculating metadata quality: {e}")
            return 0.0

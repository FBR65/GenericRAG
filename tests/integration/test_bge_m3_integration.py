"""
Integration Tests for BGE-M3 Components
"""

import pytest
import asyncio
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List, Optional
import io

from src.app.services.bge_m3_service import BGE_M3_Service, CacheManager, ErrorHandler
from src.app.services.search_service import SearchService
from src.app.utils.qdrant_utils import BGE_M3_QdrantUtils
from src.app.models.schemas import (
    SearchRequest,
    SearchResponse,
    BGE_M3_SearchMode,
    BGE_M3_MultivectorStrategy,
    SearchResult,
    IngestRequest,
    IngestResponse,
    IngestResult,
    SessionInfo
)
from src.app.settings import Settings, BGE_M3_Settings


class TestBGE_M3Integration:
    """Integration tests for BGE-M3 components"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings"""
        return Settings()

    @pytest.fixture
    def mock_bge_m3_settings(self):
        """Mock BGE-M3 settings"""
        return BGE_M3_Settings(
            model_name="BGE-M3",
            model_device="cpu",
            max_length=8192,
            dense_dimension=1024,
            sparse_dimension=10000,
            multi_vector_count=10,
            multi_vector_dimension=768,
            dense_normalize=True,
            sparse_normalize=True,
            cache_enabled=True,
            cache_redis_url="redis://localhost:6379",
            cache_ttl=3600,
            max_retries=3,
            retry_delay=1.0,
            circuit_breaker_threshold=5
        )

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client"""
        with patch('qdrant_client.QdrantClient') as mock_client:
            mock_instance = Mock()
            mock_instance.scroll.return_value = ([], None)
            mock_instance.delete.return_value = None
            mock_instance.create_collection.return_value = None
            mock_instance.upsert.return_value = None
            mock_instance.search.return_value = []
            mock_instance.health_check.return_value = {"status": "ok"}
            mock_client.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client"""
        with patch('redis.Redis') as mock_client:
            mock_instance = Mock()
            mock_instance.ping.return_value = True
            mock_instance.get.return_value = None
            mock_instance.setex.return_value = True
            mock_instance.mget.return_value = [None]
            mock_instance.pipeline.return_value = Mock()
            mock_instance.keys.return_value = []
            mock_instance.delete.return_value = 0
            mock_client.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def bge_m3_service(self, mock_settings, mock_bge_m3_settings, mock_redis_client):
        """Create BGE-M3 service instance"""
        with patch('src.app.services.bge_m3_service.redis.Redis.from_url', return_value=mock_redis_client):
            service = BGE_M3_Service(mock_settings)
            service.model_client = Mock()
            service.model_client.generate_embeddings = AsyncMock(return_value={
                "dense": [0.1] * mock_bge_m3_settings.dense_dimension,
                "sparse": {str(i): 0.5 for i in range(0, 100, 10)},
                "multi_vector": [[0.1] * mock_bge_m3_settings.multi_vector_dimension 
                               for _ in range(mock_bge_m3_settings.multi_vector_count)]
            })
            return service

    @pytest.fixture
    def qdrant_utils(self, mock_qdrant_client):
        """Create Qdrant utils instance"""
        with patch('src.app.utils.qdrant_utils.QdrantClient', return_value=mock_qdrant_client):
            return BGE_M3_QdrantUtils(collection_name="test_collection")

    @pytest.fixture
    def search_service(self, mock_settings, bge_m3_service, qdrant_utils):
        """Create SearchService instance"""
        return SearchService(mock_settings, bge_m3_service, qdrant_utils)

    @pytest.fixture
    def test_documents(self):
        """Test documents for integration tests"""
        return [
            {
                "id": "doc1",
                "content": "Dies ist ein Testdokument über Zollbestimmungen und Importregeln.",
                "metadata": {
                    "source": "zollbestimmungen.pdf",
                    "page": 1,
                    "type": "text",
                    "author": "Zollamt",
                    "created_at": "2023-01-01"
                }
            },
            {
                "id": "doc2",
                "content": "Import von Waren in die EU erfordert bestimmte Dokumente und Zollformalitäten.",
                "metadata": {
                    "source": "import_regeln.pdf",
                    "page": 2,
                    "type": "text",
                    "author": "EU-Kommission",
                    "created_at": "2023-01-02"
                }
            },
            {
                "id": "doc3",
                "content": "Zolltarife und Warenklassifikation sind wichtig für den internationalen Handel.",
                "metadata": {
                    "source": "zolltarife.pdf",
                    "page": 3,
                    "type": "text",
                    "author": "WTO",
                    "created_at": "2023-01-03"
                }
            }
        ]

    @pytest.fixture
    def test_queries(self):
        """Test queries for integration tests"""
        return [
            "Zollbestimmungen Import",
            "Dokumente für EU Import",
            "Zolltarif Warenklassifikation",
            "Internationale Handel Regeln"
        ]


class TestBGE_M3ServiceIntegration:
    """Integration tests for BGE-M3 Service"""

    @pytest.mark.asyncio
    async def test_bge_m3_service_initialization(self, bge_m3_service):
        """Test BGE-M3 service initialization"""
        assert bge_m3_service.settings is not None
        assert bge_m3_service.bge_m3_settings is not None
        assert bge_m3_service.cache_manager is not None
        assert bge_m3_service.error_handler is not None
        assert bge_m3_service.model_client is not None

    @pytest.mark.asyncio
    async def test_generate_all_embeddings(self, bge_m3_service, test_documents):
        """Test generating all three types of embeddings"""
        results = []
        for doc in test_documents:
            result = await bge_m3_service.generate_embeddings(doc["content"])
            results.append(result)
        
        assert len(results) == len(test_documents)
        for result in results:
            assert "embeddings" in result
            assert "errors" in result
            assert "text" in result
            assert "dense" in result["embeddings"]
            assert "sparse" in result["embeddings"]
            assert "multi_vector" in result["embeddings"]
            assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_caching(self, bge_m3_service, test_documents):
        """Test embedding generation with caching"""
        # First call should generate embeddings
        result1 = await bge_m3_service.generate_embeddings(test_documents[0]["content"])
        
        # Second call should use cache
        result2 = await bge_m3_service.generate_embeddings(test_documents[0]["content"])
        
        # Results should be the same
        assert result1["embeddings"]["dense"] == result2["embeddings"]["dense"]
        assert result1["embeddings"]["sparse"] == result2["embeddings"]["sparse"]
        assert result1["embeddings"]["multi_vector"] == result2["embeddings"]["multi_vector"]

    @pytest.mark.asyncio
    async def test_generate_embeddings_error_handling(self, bge_m3_service):
        """Test embedding generation error handling"""
        # Test with empty text
        result = await bge_m3_service.generate_embeddings("")
        assert len(result["errors"]) > 0
        assert "embeddings" in result

    @pytest.mark.asyncio
    async def test_health_check(self, bge_m3_service):
        """Test BGE-M3 service health check"""
        health = await bge_m3_service.health_check()
        
        assert "status" in health
        assert "cache_status" in health
        assert "model_status" in health
        assert "error" in health

    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, bge_m3_service, test_documents):
        """Test batch embedding generation"""
        texts = [doc["content"] for doc in test_documents]
        
        results = await bge_m3_service.batch_generate_embeddings(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "embeddings" in result
            assert "errors" in result
            assert "text" in result


class TestQdrantUtilsIntegration:
    """Integration tests for Qdrant Utils"""

    @pytest.mark.asyncio
    async def test_create_collection(self, qdrant_utils):
        """Test creating collection"""
        result = await qdrant_utils.create_collection()
        
        assert result is True
        qdrant_utils.client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search(self, qdrant_utils, test_documents):
        """Test hybrid search functionality"""
        # Mock embeddings
        query_embeddings = {
            "dense": [0.1] * 1024,
            "sparse": {str(i): 0.5 for i in range(0, 100, 10)},
            "multi_vector": [[0.1] * 768 for _ in range(10)]
        }
        
        # Mock search results
        qdrant_utils.client.search.return_value = [
            Mock(id=doc["id"], score=0.9, payload=doc["metadata"])
            for doc in test_documents
        ]
        
        results = await qdrant_utils.hybrid_search(
            query_embeddings=query_embeddings,
            top_k=10,
            alpha=0.5,
            beta=0.3,
            gamma=0.2
        )
        
        assert "results" in results
        assert len(results["results"]) == len(test_documents)
        for result in results["results"]:
            assert "id" in result
            assert "score" in result
            assert "payload" in result

    @pytest.mark.asyncio
    async def test_prepare_query_embeddings(self, qdrant_utils, bge_m3_service):
        """Test preparing query embeddings"""
        query_text = "Test query"
        
        embeddings = await qdrant_utils.prepare_query_embeddings(
            query_text=query_text,
            bge_m3_service=bge_m3_service,
            search_mode=BGE_M3_SearchMode.HYBRID
        )
        
        assert "dense" in embeddings
        assert "sparse" in embeddings
        assert "multi_vector" in embeddings
        assert len(embeddings["dense"]) == 1024
        assert len(embeddings["sparse"]) > 0
        assert len(embeddings["multi_vector"]) == 10

    @pytest.mark.asyncio
    async def test_scroll_collection(self, qdrant_utils, test_documents):
        """Test scrolling collection"""
        # Mock scroll results
        qdrant_utils.client.scroll.return_value = (
            [Mock(id=doc["id"], payload=doc) for doc in test_documents],
            None
        )
        
        results = await qdrant_utils.scroll_collection(limit=100)
        
        assert results is not None
        assert len(results[0]) == len(test_documents)
        for i, result in enumerate(results[0]):
            assert result.id == test_documents[i]["id"]

    @pytest.mark.asyncio
    async def test_health_check(self, qdrant_utils):
        """Test Qdrant utils health check"""
        health = await qdrant_utils.health_check()
        
        assert "status" in health
        assert "error" in health


class TestSearchServiceIntegration:
    """Integration tests for Search Service"""

    @pytest.mark.asyncio
    async def test_search_service_initialization(self, search_service):
        """Test SearchService initialization"""
        assert search_service.settings is not None
        assert search_service.bge_m3_service is not None
        assert search_service.qdrant_utils is not None

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search(self, search_service, test_queries, test_documents):
        """Test BGE-M3 hybrid search"""
        for query in test_queries:
            result = await search_service.bge_m3_hybrid_search(
                query=query,
                session_id="test-session",
                top_k=5,
                alpha=0.5,
                beta=0.3,
                gamma=0.2
            )
            
            assert isinstance(result, SearchResponse)
            assert result.query == query
            assert result.session_id == "test-session"
            assert result.search_mode == BGE_M3_SearchMode.HYBRID
            assert "results" in result
            assert "total_results" in result

    @pytest.mark.asyncio
    async def test_bge_m3_multivector_search(self, search_service, test_queries):
        """Test BGE-M3 multivector search"""
        for query in test_queries:
            result = await search_service.bge_m3_multivector_search(
                query=query,
                session_id="test-session",
                top_k=5,
                multivector_strategy=BGE_M3_MultivectorStrategy.MAX_SIM
            )
            
            assert isinstance(result, SearchResponse)
            assert result.query == query
            assert result.session_id == "test-session"
            assert result.search_mode == BGE_M3_SearchMode.MULTIVECTOR

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings(self, search_service, test_queries):
        """Test getting BGE-M3 embeddings"""
        for query in test_queries:
            embeddings = await search_service.get_bge_m3_embeddings(
                query=query,
                mode=BGE_M3_SearchMode.HYBRID
            )
            
            assert "dense" in embeddings
            assert "sparse" in embeddings
            assert "multi_vector" in embeddings
            assert len(embeddings["dense"]) == 1024
            assert len(embeddings["multi_vector"]) == 10

    @pytest.mark.asyncio
    async def test_health_check(self, search_service):
        """Test SearchService health check"""
        health = await search_service.health_check()
        
        assert "status" in health
        assert "bge_m3_service" in health
        assert "qdrant_service" in health

    @pytest.mark.asyncio
    async def test_batch_search(self, search_service, test_queries):
        """Test batch search functionality"""
        search_requests = [
            SearchRequest(
                query=query,
                session_id="test-session",
                search_mode=BGE_M3_SearchMode.HYBRID
            )
            for query in test_queries
        ]
        
        results = await search_service.batch_search(search_requests)
        
        assert len(results) == len(search_requests)
        for result in results:
            assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_search_with_metadata_filters(self, search_service, test_queries):
        """Test search with metadata filters"""
        filters = {
            "must": [
                {"key": "author", "match": {"value": "Zollamt"}}
            ]
        }
        
        result = await search_service.bge_m3_hybrid_search(
            query=test_queries[0],
            session_id="test-session",
            metadata_filters=filters
        )
        
        assert isinstance(result, SearchResponse)
        assert "results" in result


class TestEndToEndIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, search_service, bge_m3_service, qdrant_utils, test_documents, test_queries):
        """Test complete workflow from ingestion to search"""
        # Step 1: Generate embeddings for all documents
        embeddings = []
        for doc in test_documents:
            result = await bge_m3_service.generate_embeddings(doc["content"])
            embeddings.append(result)
        
        # Step 2: Store embeddings in Qdrant
        for i, (doc, embedding) in enumerate(zip(test_documents, embeddings)):
            await qdrant_utils.upsert_points(
                points=[{
                    "id": doc["id"],
                    "vector": embedding["embeddings"]["dense"],
                    "payload": {
                        **doc["metadata"],
                        "content": doc["content"],
                        "session_id": "test-session"
                    }
                }]
            )
        
        # Step 3: Perform searches
        search_results = []
        for query in test_queries:
            result = await search_service.bge_m3_hybrid_search(
                query=query,
                session_id="test-session",
                top_k=3
            )
            search_results.append(result)
        
        # Verify results
        assert len(embeddings) == len(test_documents)
        assert len(search_results) == len(test_queries)
        
        for result in search_results:
            assert isinstance(result, SearchResponse)
            assert result.total_results >= 0

    @pytest.mark.asyncio
    async def test_workflow_with_caching(self, search_service, bge_m3_service, qdrant_utils, test_documents, test_queries):
        """Test complete workflow with caching"""
        # Step 1: Generate and store embeddings (first time)
        for doc in test_documents:
            await bge_m3_service.generate_embeddings(doc["content"])
        
        # Step 2: Perform searches (should use cache)
        start_time = time.time()
        for query in test_queries:
            await search_service.bge_m3_hybrid_search(
                query=query,
                session_id="test-session",
                top_k=3
            )
        first_run_time = time.time() - start_time
        
        # Step 3: Perform searches again (should be faster due to cache)
        start_time = time.time()
        for query in test_queries:
            await search_service.bge_m3_hybrid_search(
                query=query,
                session_id="test-session",
                top_k=3
            )
        second_run_time = time.time() - start_time
        
        # Second run should be faster (cache hit)
        assert second_run_time <= first_run_time

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, search_service, bge_m3_service, qdrant_utils):
        """Test error handling in complete workflow"""
        # Test with invalid query
        result = await search_service.bge_m3_hybrid_search(
            query="",
            session_id="test-session"
        )
        
        # Should handle gracefully
        assert isinstance(result, SearchResponse)
        assert result.total_results == 0

    @pytest.mark.asyncio
    async def test_performance_workflow(self, search_service, bge_m3_service, qdrant_utils, test_documents):
        """Test performance of complete workflow"""
        # Generate embeddings for multiple documents
        start_time = time.time()
        embeddings = []
        for doc in test_documents:
            result = await bge_m3_service.generate_embeddings(doc["content"])
            embeddings.append(result)
        embedding_time = time.time() - start_time
        
        # Store embeddings
        start_time = time.time()
        for doc, embedding in zip(test_documents, embeddings):
            await qdrant_utils.upsert_points([{
                "id": doc["id"],
                "vector": embedding["embeddings"]["dense"],
                "payload": {
                    **doc["metadata"],
                    "content": doc["content"],
                    "session_id": "test-session"
                }
            }])
        storage_time = time.time() - start_time
        
        # Perform search
        start_time = time.time()
        result = await search_service.bge_m3_hybrid_search(
            query="Zollbestimmungen",
            session_id="test-session",
            top_k=5
        )
        search_time = time.time() - start_time
        
        # Log performance metrics
        print(f"Embedding generation time: {embedding_time:.2f}s")
        print(f"Storage time: {storage_time:.2f}s")
        print(f"Search time: {search_time:.2f}s")
        
        # Verify performance is reasonable
        assert embedding_time < 10.0  # Should complete within 10 seconds
        assert storage_time < 5.0     # Should complete within 5 seconds
        assert search_time < 2.0      # Should complete within 2 seconds


class TestBGE_M3PerformanceIntegration:
    """Performance integration tests for BGE-M3"""

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, bge_m3_service, test_documents):
        """Test batch processing performance"""
        # Test with different batch sizes
        batch_sizes = [1, 5, 10, 20]
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Process documents in batches
            for i in range(0, len(test_documents), batch_size):
                batch = test_documents[i:i + batch_size]
                tasks = [bge_m3_service.generate_embeddings(doc["content"]) for doc in batch]
                await asyncio.gather(*tasks)
            
            processing_time = time.time() - start_time
            documents_per_second = len(test_documents) / processing_time
            
            print(f"Batch size {batch_size}: {processing_time:.2f}s, {documents_per_second:.2f} docs/s")
            
            # Performance should be reasonable
            assert documents_per_second > 1.0

    @pytest.mark.asyncio
    async def test_cache_performance(self, bge_m3_service, test_documents):
        """Test cache performance"""
        # First pass - generate embeddings
        start_time = time.time()
        for doc in test_documents:
            await bge_m3_service.generate_embeddings(doc["content"])
        first_pass_time = time.time() - start_time
        
        # Second pass - should use cache
        start_time = time.time()
        for doc in test_documents:
            await bge_m3_service.generate_embeddings(doc["content"])
        second_pass_time = time.time() - start_time
        
        cache_speedup = first_pass_time / second_pass_time
        
        print(f"First pass: {first_pass_time:.2f}s")
        print(f"Second pass: {second_pass_time:.2f}s")
        print(f"Cache speedup: {cache_speedup:.2f}x")
        
        # Cache should provide significant speedup
        assert cache_speedup > 2.0

    @pytest.mark.asyncio
    async def test_search_performance(self, search_service, bge_m3_service, qdrant_utils, test_documents):
        """Test search performance"""
        # Pre-populate with documents
        for doc in test_documents:
            embedding = await bge_m3_service.generate_embeddings(doc["content"])
            await qdrant_utils.upsert_points([{
                "id": doc["id"],
                "vector": embedding["embeddings"]["dense"],
                "payload": {
                    **doc["metadata"],
                    "content": doc["content"],
                    "session_id": "test-session"
                }
            }])
        
        # Test search performance
        queries = ["Zoll", "Import", "Dokument", "Handel"]
        
        start_time = time.time()
        for query in queries:
            await search_service.bge_m3_hybrid_search(
                query=query,
                session_id="test-session",
                top_k=5
            )
        total_time = time.time() - start_time
        
        searches_per_second = len(queries) / total_time
        
        print(f"Search performance: {total_time:.2f}s for {len(queries)} queries")
        print(f"Searches per second: {searches_per_second:.2f}")
        
        # Search should be fast
        assert searches_per_second > 5.0


class TestBGE_M3ErrorHandlingIntegration:
    """Error handling integration tests"""

    @pytest.mark.asyncio
    async def test_service_unavailable_handling(self, search_service, bge_m3_service, qdrant_utils):
        """Test handling when services are unavailable"""
        # Mock service failures
        bge_m3_service.generate_embeddings = AsyncMock(side_effect=Exception("Service unavailable"))
        
        # Should handle gracefully
        result = await search_service.bge_m3_hybrid_search(
            query="Test query",
            session_id="test-session"
        )
        
        assert isinstance(result, SearchResponse)
        assert result.total_results == 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, search_service, bge_m3_service):
        """Test timeout handling"""
        # Mock slow service
        async def slow_embedding_generation(text):
            await asyncio.sleep(2.0)  # Simulate timeout
            return {"embeddings": {}, "errors": [], "text": text}
        
        bge_m3_service.generate_embeddings = slow_embedding_generation
        
        # Should handle timeout
        result = await search_service.bge_m3_hybrid_search(
            query="Test query",
            session_id="test-session"
        )
        
        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, bge_m3_service):
        """Test circuit breaker functionality"""
        # Simulate multiple failures
        bge_m3_service.model_client.generate_embeddings = AsyncMock(side_effect=Exception("Service error"))
        
        # Should trigger circuit breaker after multiple failures
        for i in range(6):  # More than threshold
            try:
                await bge_m3_service.generate_embeddings("Test")
            except Exception:
                pass
        
        # Circuit breaker should be open
        assert bge_m3_service.error_handler.is_circuit_open() is True

    @pytest.mark.asyncio
    async def test_retry_mechanism_integration(self, bge_m3_service):
        """Test retry mechanism integration"""
        # Mock service that fails first few times
        call_count = 0
        async def failing_embedding_generation(text):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"embeddings": {"dense": [0.1] * 1024}, "errors": [], "text": text}
        
        bge_m3_service.model_client.generate_embeddings = failing_embedding_generation
        
        # Should eventually succeed after retries
        result = await bge_m3_service.generate_embeddings("Test")
        
        assert "embeddings" in result
        assert len(result["errors"]) == 0


if __name__ == "__main__":
    pytest.main([__file__])
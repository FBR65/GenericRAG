"""
Tests for BGE-M3 Search Service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional
import numpy as np

from src.app.services.search_service import SearchService
from src.app.services.bge_m3_service import BGE_M3_Service
from src.app.utils.qdrant_utils import BGE_M3_QdrantUtils
from src.app.models.schemas import (
    SearchRequest,
    SearchResponse,
    BGE_M3_SearchMode,
    BGE_M3_MultivectorStrategy,
    SearchResult,
    SessionInfo
)
from src.app.settings import Settings, BGE_M3_Settings


class TestBGE_M3_SearchService:
    """Test BGE-M3 Search Service functionality"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings"""
        return Settings()

    @pytest.fixture
    def mock_bge_m3_service(self):
        """Mock BGE-M3 service"""
        with patch('src.app.services.search_service.BGE_M3_Service') as mock_service:
            mock_instance = Mock(spec=BGE_M3_Service)
            mock_instance.generate_dense_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_instance.generate_sparse_embedding = AsyncMock(return_value={"0": 0.5, "1": 0.3})
            mock_instance.generate_multivector_embedding = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
            mock_instance.generate_embeddings = AsyncMock(return_value={
                "embeddings": {
                    "dense": [0.1, 0.2, 0.3],
                    "sparse": {"0": 0.5, "1": 0.3},
                    "multi_vector": [[0.1, 0.2], [0.3, 0.4]]
                },
                "errors": [],
                "text": "Test query"
            })
            mock_instance.health_check = AsyncMock(return_value={
                "status": "healthy",
                "cache_status": "healthy",
                "model_status": "healthy"
            })
            mock_service.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_qdrant_utils(self):
        """Mock Qdrant utils"""
        with patch('src.app.services.search_service.BGE_M3_QdrantUtils') as mock_utils:
            mock_instance = Mock(spec=BGE_M3_QdrantUtils)
            mock_instance.create_collection = AsyncMock(return_value=True)
            mock_instance.hybrid_search = AsyncMock(return_value={
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "payload": {
                            "content": "Test document 1",
                            "metadata": {"source": "test1.pdf", "page": 1}
                        }
                    }
                ]
            })
            mock_instance.prepare_query_embeddings = AsyncMock(return_value={
                "dense": [0.1, 0.2, 0.3],
                "sparse": {"0": 0.5, "1": 0.3},
                "multi_vector": [[0.1, 0.2], [0.3, 0.4]]
            })
            mock_instance.scroll_collection = AsyncMock(return_value=([
                {
                    "id": 1,
                    "payload": {"content": "Test document 1", "metadata": {"source": "test1.pdf"}}
                }
            ], None))
            mock_instance.health_check = AsyncMock(return_value={"status": "healthy"})
            mock_utils.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def search_service(self, mock_settings, mock_bge_m3_service, mock_qdrant_utils):
        """Create SearchService instance"""
        return SearchService(mock_settings, mock_bge_m3_service, mock_qdrant_utils)

    @pytest.fixture
    def mock_search_request(self):
        """Mock search request"""
        return SearchRequest(
            query="Test query",
            session_id="test-session",
            search_mode=BGE_M3_SearchMode.HYBRID,
            top_k=10,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            score_threshold=0.8,
            use_cache=True,
            normalize=True,
            multivector_strategy=BGE_M3_MultivectorStrategy.MAX_SIM,
            metadata_filters={"must": [{"key": "source", "match": {"value": "test.pdf"}}]},
            page=1,
            page_size=20
        )

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results"""
        return [
            SearchResult(
                id="doc1",
                document="test1.pdf",
                page=1,
                score=0.95,
                content="Test document 1 content",
                metadata={"source": "test1.pdf", "page": 1, "type": "text"},
                search_type="hybrid",
                embedding_type="dense"
            ),
            SearchResult(
                id="doc2",
                document="test2.pdf",
                page=2,
                score=0.85,
                content="Test document 2 content",
                metadata={"source": "test2.pdf", "page": 2, "type": "text"},
                search_type="hybrid",
                embedding_type="sparse"
            )
        ]


class TestBGE_M3_SearchServiceInitialization:
    """Test SearchService initialization"""

    def test_search_service_initialization(self, mock_settings, mock_bge_m3_service, mock_qdrant_utils):
        """Test SearchService initialization"""
        service = SearchService(mock_settings, mock_bge_m3_service, mock_qdrant_utils)
        
        assert service.settings == mock_settings
        assert service.bge_m3_service == mock_bge_m3_service
        assert service.qdrant_utils == mock_qdrant_utils

    def test_search_service_initialization_without_services(self, mock_settings):
        """Test SearchService initialization without services"""
        service = SearchService(mock_settings)
        
        assert service.settings == mock_settings
        assert service.bge_m3_service is None
        assert service.qdrant_utils is None


class TestGetBGE_M3Embeddings:
    """Test get_bge_m3_embeddings method"""

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_dense(self, search_service, mock_bge_m3_service):
        """Test getting dense embeddings"""
        query_text = "Test query"
        mode = BGE_M3_SearchMode.DENSE
        
        result = await search_service.get_bge_m3_embeddings(query_text, mode)
        
        assert result == {"dense": [0.1, 0.2, 0.3]}
        mock_bge_m3_service.generate_dense_embedding.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_sparse(self, search_service, mock_bge_m3_service):
        """Test getting sparse embeddings"""
        query_text = "Test query"
        mode = BGE_M3_SearchMode.SPARSE
        
        result = await search_service.get_bge_m3_embeddings(query_text, mode)
        
        assert result == {"sparse": {"0": 0.5, "1": 0.3}}
        mock_bge_m3_service.generate_sparse_embedding.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_multivector(self, search_service, mock_bge_m3_service):
        """Test getting multivector embeddings"""
        query_text = "Test query"
        mode = BGE_M3_SearchMode.MULTIVECTOR
        
        result = await search_service.get_bge_m3_embeddings(query_text, mode)
        
        assert result == {"multi_vector": [[0.1, 0.2], [0.3, 0.4]]}
        mock_bge_m3_service.generate_multivector_embedding.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_hybrid(self, search_service, mock_bge_m3_service):
        """Test getting hybrid embeddings"""
        query_text = "Test query"
        mode = BGE_M3_SearchMode.HYBRID
        
        result = await search_service.get_bge_m3_embeddings(query_text, mode)
        
        assert "dense" in result
        assert "sparse" in result
        assert "multi_vector" in result
        mock_bge_m3_service.generate_embeddings.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_with_cache(self, search_service, mock_bge_m3_service):
        """Test getting embeddings with cache"""
        query_text = "Test query"
        mode = BGE_M3_SearchMode.DENSE
        
        # Mock cache hit
        search_service.bge_m3_service.cache_manager.get_embedding = AsyncMock(
            return_value={"dense": [0.1, 0.2, 0.3]}
        )
        
        result = await search_service.get_bge_m3_embeddings(query_text, mode, use_cache=True)
        
        assert result == {"dense": [0.1, 0.2, 0.3]}
        # Should not call generate_embedding due to cache hit
        mock_bge_m3_service.generate_dense_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_with_errors(self, search_service, mock_bge_m3_service):
        """Test getting embeddings with errors"""
        query_text = "Test query"
        mode = BGE_M3_SearchMode.DENSE
        
        mock_bge_m3_service.generate_dense_embedding.side_effect = Exception("Generation error")
        
        result = await search_service.get_bge_m3_embeddings(query_text, mode)
        
        assert result == {"dense": []}  # Should return fallback

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_invalid_mode(self, search_service):
        """Test getting embeddings with invalid mode"""
        query_text = "Test query"
        mode = "invalid"
        
        with pytest.raises(ValueError, match="Unknown search mode: invalid"):
            await search_service.get_bge_m3_embeddings(query_text, mode)

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_empty_query(self, search_service):
        """Test getting embeddings with empty query"""
        query_text = ""
        mode = BGE_M3_SearchMode.DENSE
        
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            await search_service.get_bge_m3_embeddings(query_text, mode)

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_with_normalization(self, search_service, mock_bge_m3_service):
        """Test getting embeddings with normalization"""
        query_text = "Test query"
        mode = BGE_M3_SearchMode.DENSE
        
        result = await search_service.get_bge_m3_embeddings(query_text, mode, normalize=True)
        
        assert result == {"dense": [0.1, 0.2, 0.3]}
        mock_bge_m3_service.generate_dense_embedding.assert_called_once_with(query_text)


class TestBGE_M3HybridSearch:
    """Test bge_m3_hybrid_search method"""

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_success(self, search_service, mock_search_request, mock_search_results):
        """Test successful hybrid search"""
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "payload": {
                        "content": "Test document 1",
                        "metadata": {"source": "test1.pdf", "page": 1}
                    }
                }
            ]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert result.query == mock_search_request.query
        assert result.session_id == mock_search_request.session_id
        assert len(result.results) > 0
        assert result.total_results > 0
        assert result.search_mode == mock_search_request.search_mode

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_dense_only(self, search_service, mock_search_request):
        """Test hybrid search with dense only mode"""
        mock_search_request.search_mode = BGE_M3_SearchMode.DENSE
        mock_search_request.alpha = 1.0
        mock_search_request.beta = 0.0
        mock_search_request.gamma = 0.0
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert result.search_mode == BGE_M3_SearchMode.DENSE

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_sparse_only(self, search_service, mock_search_request):
        """Test hybrid search with sparse only mode"""
        mock_search_request.search_mode = BGE_M3_SearchMode.SPARSE
        mock_search_request.alpha = 0.0
        mock_search_request.beta = 1.0
        mock_search_request.gamma = 0.0
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert result.search_mode == BGE_M3_SearchMode.SPARSE

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_multivector_only(self, search_service, mock_search_request):
        """Test hybrid search with multivector only mode"""
        mock_search_request.search_mode = BGE_M3_SearchMode.MULTIVECTOR
        mock_search_request.alpha = 0.0
        mock_search_request.beta = 0.0
        mock_search_request.gamma = 1.0
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert result.search_mode == BGE_M3_SearchMode.MULTIVECTOR

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_filters(self, search_service, mock_search_request):
        """Test hybrid search with metadata filters"""
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_score_threshold(self, search_service, mock_search_request):
        """Test hybrid search with score threshold"""
        mock_search_request.score_threshold = 0.9
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [
                {"id": "doc1", "score": 0.95, "payload": {"content": "Test"}},
                {"id": "doc2", "score": 0.85, "payload": {"content": "Test"}}
            ]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        # Should only return results above threshold
        assert all(r.score >= 0.9 for r in result.results)

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_pagination(self, search_service, mock_search_request):
        """Test hybrid search with pagination"""
        mock_search_request.page = 2
        mock_search_request.page_size = 5
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert result.page == 2
        assert result.page_size == 5

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_multivector_strategy(self, search_service, mock_search_request):
        """Test hybrid search with multivector strategy"""
        mock_search_request.multivector_strategy = BGE_M3_MultivectorStrategy.AVG_POOL
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert result.multivector_strategy == BGE_M3_MultivectorStrategy.AVG_POOL

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_retry(self, search_service, mock_search_request):
        """Test hybrid search with retry mechanism"""
        call_count = 0
        
        def mock_hybrid_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return {"results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]}
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(side_effect=mock_hybrid_search)
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert call_count == 3  # Should retry 2 times then succeed

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_retry_failure(self, search_service, mock_search_request):
        """Test hybrid search when all retries fail"""
        search_service.qdrant_utils.hybrid_search = AsyncMock(side_effect=Exception("Persistent error"))
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_empty_results(self, search_service, mock_search_request):
        """Test hybrid search with empty results"""
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={"results": []})
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert len(result.results) == 0
        assert result.total_results == 0

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_cache(self, search_service, mock_search_request):
        """Test hybrid search with cache"""
        mock_search_request.use_cache = True
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_normalization(self, search_service, mock_search_request):
        """Test hybrid search with normalization"""
        mock_search_request.normalize = True
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_hybrid_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert len(result.results) > 0


class TestBGE_M3MultivectorSearch:
    """Test bge_m3_multivector_search method"""

    @pytest.mark.asyncio
    async def test_bge_m3_multivector_search_success(self, search_service, mock_search_request):
        """Test successful multivector search"""
        mock_search_request.search_mode = BGE_M3_SearchMode.MULTIVECTOR
        mock_search_request.multivector_strategy = BGE_M3_MultivectorStrategy.MAX_SIM
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "payload": {
                        "content": "Test document 1",
                        "metadata": {"source": "test1.pdf", "page": 1}
                    }
                }
            ]
        })
        
        result = await search_service.bge_m3_multivector_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert result.query == mock_search_request.query
        assert result.session_id == mock_search_request.session_id
        assert len(result.results) > 0
        assert result.search_mode == BGE_M3_SearchMode.MULTIVECTOR

    @pytest.mark.asyncio
    async def test_bge_m3_multivector_search_with_avg_pool(self, search_service, mock_search_request):
        """Test multivector search with average pooling"""
        mock_search_request.search_mode = BGE_M3_SearchMode.MULTIVECTOR
        mock_search_request.multivector_strategy = BGE_M3_MultivectorStrategy.AVG_POOL
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_multivector_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert result.multivector_strategy == BGE_M3_MultivectorStrategy.AVG_POOL

    @pytest.mark.asyncio
    async def test_bge_m3_multivector_search_with_max_sim(self, search_service, mock_search_request):
        """Test multivector search with max similarity"""
        mock_search_request.search_mode = BGE_M3_SearchMode.MULTIVECTOR
        mock_search_request.multivector_strategy = BGE_M3_MultivectorStrategy.MAX_SIM
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(return_value={
            "results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]
        })
        
        result = await search_service.bge_m3_multivector_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert result.multivector_strategy == BGE_M3_MultivectorStrategy.MAX_SIM

    @pytest.mark.asyncio
    async def test_bge_m3_multivector_search_with_retry(self, search_service, mock_search_request):
        """Test multivector search with retry mechanism"""
        mock_search_request.search_mode = BGE_M3_SearchMode.MULTIVECTOR
        
        call_count = 0
        
        def mock_hybrid_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return {"results": [{"id": "doc1", "score": 0.95, "payload": {"content": "Test"}}]}
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(side_effect=mock_hybrid_search)
        
        result = await search_service.bge_m3_multivector_search(mock_search_request)
        
        assert isinstance(result, SearchResponse)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_bge_m3_multivector_search_retry_failure(self, search_service, mock_search_request):
        """Test multivector search when all retries fail"""
        mock_search_request.search_mode = BGE_M3_SearchMode.MULTIVECTOR
        
        search_service.qdrant_utils.hybrid_search = AsyncMock(side_effect=Exception("Persistent error"))
        
        result = await search_service.bge_m3_multivector_search(mock_search_request)
        
        assert result is None


class TestBGE_M3SearchServiceIntegration:
    """Test SearchService integration methods"""

    @pytest.mark.asyncio
    async def test_search_service_health_check(self, search_service, mock_bge_m3_service, mock_qdrant_utils):
        """Test search service health check"""
        mock_bge_m3_service.health_check.return_value = {
            "status": "healthy",
            "cache_status": "healthy",
            "model_status": "healthy"
        }
        mock_qdrant_utils.health_check.return_value = {"status": "healthy"}
        
        result = await search_service.health_check()
        
        assert result["status"] == "healthy"
        assert result["bge_m3_service"]["status"] == "healthy"
        assert result["qdrant_service"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_search_service_health_check_with_errors(self, search_service, mock_bge_m3_service, mock_qdrant_utils):
        """Test search service health check with errors"""
        mock_bge_m3_service.health_check.side_effect = Exception("Service error")
        mock_qdrant_utils.health_check.return_value = {"status": "healthy"}
        
        result = await search_service.health_check()
        
        assert result["status"] == "unhealthy"
        assert "bge_m3_service" in result
        assert "qdrant_service" in result

    @pytest.mark.asyncio
    async def test_search_service_get_session_info(self, search_service, mock_qdrant_utils):
        """Test getting session information"""
        mock_qdrant_utils.scroll_collection = AsyncMock(return_value=([
            {
                "id": 1,
                "payload": {
                    "session_id": "test-session",
                    "document": "test1.pdf",
                    "created_at": "2023-01-01T00:00:00"
                }
            },
            {
                "id": 2,
                "payload": {
                    "session_id": "test-session",
                    "document": "test2.pdf",
                    "created_at": "2023-01-01T00:00:00"
                }
            }
        ], None))
        
        result = await search_service.get_session_info("test-session")
        
        assert isinstance(result, SessionInfo)
        assert result.session_id == "test-session"
        assert result.document_count == 2
        assert "test1.pdf" in result.documents
        assert "test2.pdf" in result.documents

    @pytest.mark.asyncio
    async def test_search_service_get_session_info_empty(self, search_service, mock_qdrant_utils):
        """Test getting session info when session doesn't exist"""
        mock_qdrant_utils.scroll_collection = AsyncMock(return_value=([], None))
        
        result = await search_service.get_session_info("nonexistent-session")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_search_service_get_session_documents(self, search_service, mock_qdrant_utils):
        """Test getting session documents"""
        mock_qdrant_utils.scroll_collection = AsyncMock(return_value=([
            {
                "id": 1,
                "payload": {"document": "test1.pdf", "session_id": "test-session"}
            },
            {
                "id": 2,
                "payload": {"document": "test2.pdf", "session_id": "test-session"}
            },
            {
                "id": 3,
                "payload": {"document": "test1.pdf", "session_id": "test-session"}
            }
        ], None))
        
        result = await search_service.get_session_documents("test-session")
        
        assert isinstance(result, list)
        assert len(result) == 2  # Should be unique
        assert "test1.pdf" in result
        assert "test2.pdf" in result

    @pytest.mark.asyncio
    async def test_search_service_get_session_documents_empty(self, search_service, mock_qdrant_utils):
        """Test getting session documents when session doesn't exist"""
        mock_qdrant_utils.scroll_collection = AsyncMock(return_value=([], None))
        
        result = await search_service.get_session_documents("nonexistent-session")
        
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_search_service_delete_session(self, search_service, mock_qdrant_utils):
        """Test deleting a session"""
        mock_qdrant_utils.scroll_collection = AsyncMock(return_value=([
            {
                "id": 1,
                "payload": {"document": "test1.pdf", "session_id": "test-session"}
            }
        ], None))
        mock_qdrant_utils.batch_delete = AsyncMock(return_value=True)
        
        result = await search_service.delete_session("test-session")
        
        assert result is True
        mock_qdrant_utils.batch_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_delete_session_failure(self, search_service, mock_qdrant_utils):
        """Test deleting a session when it fails"""
        mock_qdrant_utils.scroll_collection = AsyncMock(return_value=([
            {
                "id": 1,
                "payload": {"document": "test1.pdf", "session_id": "test-session"}
            }
        ], None))
        mock_qdrant_utils.batch_delete = AsyncMock(return_value=False)
        
        result = await search_service.delete_session("test-session")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_search_service_batch_search(self, search_service, mock_search_request):
        """Test batch search functionality"""
        mock_search_requests = [mock_search_request, mock_search_request]
        
        search_service.bge_m3_hybrid_search = AsyncMock(return_value=SearchResponse(
            query="Test query",
            session_id="test-session",
            results=[],
            total_results=0,
            search_mode=BGE_M3_SearchMode.HYBRID
        ))
        
        results = await search_service.batch_search(mock_search_requests)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, SearchResponse) for r in results)

    @pytest.mark.asyncio
    async def test_search_service_batch_search_with_errors(self, search_service, mock_search_request):
        """Test batch search with errors"""
        mock_search_requests = [mock_search_request, mock_search_request]
        
        search_service.bge_m3_hybrid_search = AsyncMock(side_effect=Exception("Search error"))
        
        results = await search_service.batch_search(mock_search_requests)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(r is None for r in results)

    @pytest.mark.asyncio
    async def test_search_service_get_collection_stats(self, search_service, mock_qdrant_utils):
        """Test getting collection statistics"""
        mock_qdrant_utils.get_collection_stats = AsyncMock(return_value={
            "status": "ok",
            "result": {
                "vector_count": 1000,
                "segment_count": 2
            }
        })
        
        result = await search_service.get_collection_stats()
        
        assert result is not None
        assert result["status"] == "ok"
        assert result["result"]["vector_count"] == 1000

    @pytest.mark.asyncio
    async def test_search_service_get_collection_stats_failure(self, search_service, mock_qdrant_utils):
        """Test getting collection stats when it fails"""
        mock_qdrant_utils.get_collection_stats = AsyncMock(side_effect=Exception("Stats failed"))
        
        result = await search_service.get_collection_stats()
        
        assert result is None

    @pytest.mark.asyncio
    async def test_search_service_optimize_collection(self, search_service, mock_qdrant_utils):
        """Test optimizing collection"""
        mock_qdrant_utils.optimize_collection = AsyncMock(return_value=True)
        
        result = await search_service.optimize_collection()
        
        assert result is True
        mock_qdrant_utils.optimize_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_optimize_collection_failure(self, search_service, mock_qdrant_utils):
        """Test optimizing collection when it fails"""
        mock_qdrant_utils.optimize_collection = AsyncMock(return_value=False)
        
        result = await search_service.optimize_collection()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_search_service_create_collection(self, search_service, mock_qdrant_utils):
        """Test creating collection"""
        mock_qdrant_utils.create_collection = AsyncMock(return_value=True)
        
        result = await search_service.create_collection("test_collection")
        
        assert result is True
        mock_qdrant_utils.create_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_search_service_create_collection_failure(self, search_service, mock_qdrant_utils):
        """Test creating collection when it fails"""
        mock_qdrant_utils.create_collection = AsyncMock(return_value=False)
        
        result = await search_service.create_collection("test_collection")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_search_service_delete_collection(self, search_service, mock_qdrant_utils):
        """Test deleting collection"""
        mock_qdrant_utils.delete_collection = AsyncMock(return_value=True)
        
        result = await search_service.delete_collection("test_collection")
        
        assert result is True
        mock_qdrant_utils.delete_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_search_service_delete_collection_failure(self, search_service, mock_qdrant_utils):
        """Test deleting collection when it fails"""
        mock_qdrant_utils.delete_collection = AsyncMock(return_value=False)
        
        result = await search_service.delete_collection("test_collection")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_search_service_scroll_collection(self, search_service, mock_qdrant_utils):
        """Test scrolling collection"""
        mock_qdrant_utils.scroll_collection = AsyncMock(return_value=([
            {"id": 1, "payload": {"content": "Test"}},
            {"id": 2, "payload": {"content": "Test"}}
        ], None))
        
        result = await search_service.scroll_collection(limit=10)
        
        assert result is not None
        assert len(result[0]) == 2
        mock_qdrant_utils.scroll_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_scroll_collection_with_filters(self, search_service, mock_qdrant_utils):
        """Test scrolling collection with filters"""
        filters = {"must": [{"key": "type", "match": {"value": "document"}}]}
        
        mock_qdrant_utils.scroll_collection = AsyncMock(return_value=([{"id": 1}], None))
        
        result = await search_service.scroll_collection(limit=10, filters=filters)
        
        assert result is not None
        mock_qdrant_utils.scroll_collection.assert_called_once_with(
            collection_name=None,
            limit=10,
            filters=filters,
            with_payload=True,
            with_vectors=True
        )

    @pytest.mark.asyncio
    async def test_search_service_scroll_collection_failure(self, search_service, mock_qdrant_utils):
        """Test scrolling collection when it fails"""
        mock_qdrant_utils.scroll_collection = AsyncMock(return_value=None)
        
        result = await search_service.scroll_collection()
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
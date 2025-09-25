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
    def mock_qdrant_client(self):
        """Mock Qdrant client"""
        return Mock()

    @pytest.fixture
    def mock_image_storage(self):
        """Mock image storage"""
        return Mock()

    @pytest.fixture
    def search_service(self, mock_qdrant_client, mock_image_storage, mock_settings):
        """Create SearchService instance"""
        return SearchService(mock_qdrant_client, mock_image_storage, mock_settings)

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

    def test_search_service_initialization(self, mock_qdrant_client, mock_image_storage, mock_settings):
        """Test SearchService initialization"""
        service = SearchService(mock_qdrant_client, mock_image_storage, mock_settings)
        
        assert service.settings == mock_settings
        assert service.qdrant_client == mock_qdrant_client
        assert service.image_storage == mock_image_storage

    def test_search_service_initialization_without_services(self, mock_qdrant_client, mock_image_storage, mock_settings):
        """Test SearchService initialization without services"""
        service = SearchService(mock_qdrant_client, mock_image_storage, mock_settings)
        
        assert service.settings == mock_settings
        assert service.qdrant_client == mock_qdrant_client
        assert service.image_storage == mock_image_storage


class TestGetBGE_M3Embeddings:
    """Test get_bge_m3_embeddings method"""

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings(self, search_service, mock_settings):
        """Test getting embeddings"""
        query_text = "Test query"
        
        # Mock BGE-M3 service
        mock_bge_m3_service = Mock()
        mock_bge_m3_service.generate_embeddings = AsyncMock(return_value={
            "embeddings": {
                "dense": [0.1, 0.2, 0.3],
                "sparse": {"0": 0.5, "1": 0.3},
                "multi_vector": [[0.1, 0.2], [0.3, 0.4]]
            },
            "errors": [],
            "text": query_text
        })
        search_service.bge_m3_service = mock_bge_m3_service
        
        result = await search_service.get_bge_m3_embeddings(query_text)
        
        assert "embeddings" in result
        assert "errors" in result
        assert "text" in result
        assert "dense" in result["embeddings"]
        assert "sparse" in result["embeddings"]
        assert "multi_vector" in result["embeddings"]

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_with_cache(self, search_service, mock_settings):
        """Test getting embeddings with cache"""
        query_text = "Test query"
        
        # Mock BGE-M3 service with cache hit
        mock_bge_m3_service = Mock()
        mock_bge_m3_service.cache_manager.get_embedding = AsyncMock(
            return_value={"dense": [0.1, 0.2, 0.3], "sparse": {"0": 0.5}, "multivector": [[0.1]]}
        )
        search_service.bge_m3_service = mock_bge_m3_service
        
        result = await search_service.get_bge_m3_embeddings(query_text)
        
        assert "embeddings" in result
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_with_errors(self, search_service, mock_settings):
        """Test getting embeddings with errors"""
        query_text = "Test query"
        
        # Mock BGE-M3 service with error
        mock_bge_m3_service = Mock()
        mock_bge_m3_service.generate_embeddings = AsyncMock(side_effect=Exception("Generation error"))
        search_service.bge_m3_service = mock_bge_m3_service
        
        result = await search_service.get_bge_m3_embeddings(query_text)
        
        assert "embeddings" in result
        assert "errors" in result
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_get_bge_m3_embeddings_empty_query(self, search_service, mock_settings):
        """Test getting embeddings with empty query"""
        query_text = ""
        
        # Mock BGE-M3 service
        mock_bge_m3_service = Mock()
        mock_bge_m3_service.generate_embeddings = AsyncMock(return_value={
            "embeddings": {},
            "errors": ["Empty query provided"],
            "text": query_text
        })
        search_service.bge_m3_service = mock_bge_m3_service
        
        result = await search_service.get_bge_m3_embeddings(query_text)
        
        assert "embeddings" in result
        assert "errors" in result
        assert len(result["errors"]) > 0


class TestBGE_M3_SearchMethods:
    """Test BGE-M3 search methods"""

    @pytest.mark.asyncio
    async def test_search_text(self, search_service, mock_settings):
        """Test text search"""
        query = "Test query"
        
        # Mock BGE-M3 service
        mock_bge_m3_service = Mock()
        mock_bge_m3_service.generate_embeddings = AsyncMock(return_value={
            "embeddings": {"dense": [0.1, 0.2, 0.3], "sparse": {"0": 0.5}},
            "errors": [],
            "text": query
        })
        search_service.bge_m3_service = mock_bge_m3_service
        
        # Mock Qdrant client
        search_service.qdrant_client = Mock()
        search_service.qdrant_client.query_points = AsyncMock(return_value=Mock(
            points=[Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Test document", "document": "test.pdf", "page": 1}
            )]
        ))
        
        result = await search_service.search_text(query)
        
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_search_image(self, search_service, mock_settings):
        """Test image search"""
        query = "Test query"
        
        # Mock BGE-M3 service
        mock_bge_m3_service = Mock()
        mock_bge_m3_service.generate_dense_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        search_service.bge_m3_service = mock_bge_m3_service
        
        # Mock Qdrant client
        search_service.qdrant_client = Mock()
        search_service.qdrant_client.query_points = AsyncMock(return_value=Mock(
            points=[Mock(
                id="img1",
                score=0.95,
                payload={"content": "Test image", "document": "test.jpg", "page": 1}
            )]
        ))
        
        result = await search_service.search_image(query)
        
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_search_hybrid(self, search_service, mock_settings):
        """Test hybrid search"""
        query = "Test query"
        
        # Mock BGE-M3 service
        mock_bge_m3_service = Mock()
        mock_bge_m3_service.generate_embeddings = AsyncMock(return_value={
            "embeddings": {"dense": [0.1, 0.2, 0.3], "sparse": {"0": 0.5}},
            "errors": [],
            "text": query
        })
        search_service.bge_m3_service = mock_bge_m3_service
        
        # Mock Qdrant client
        search_service.qdrant_client = Mock()
        search_service.qdrant_client.query_points = AsyncMock(return_value=Mock(
            points=[Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Test document", "document": "test.pdf", "page": 1}
            )]
        ))
        
        result = await search_service.search_hybrid(query)
        
        assert isinstance(result, list)
        assert len(result) > 0


class TestBGE_M3_ServiceIntegration:
    """Test BGE-M3 service integration methods"""

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_documents(self, search_service, mock_settings):
        """Test preparing BGE-M3 documents"""
        documents = [
            {"id": "doc1", "document": "Test document 1", "page": 1},
            {"id": "doc2", "document": "Test document 2", "page": 2}
        ]
        
        # Mock BGE-M3 service
        mock_bge_m3_service = Mock()
        mock_bge_m3_service.generate_dense_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_bge_m3_service.generate_sparse_embedding = AsyncMock(return_value={"0": 0.5})
        mock_bge_m3_service.generate_multivector_embedding = AsyncMock(return_value=[[0.1, 0.2]])
        search_service.bge_m3_service = mock_bge_m3_service
        
        result = await search_service.prepare_bge_m3_documents(documents)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("embeddings" in doc for doc in result)

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_documents_with_errors(self, search_service, mock_settings):
        """Test preparing BGE-M3 documents with errors"""
        documents = [
            {"id": "doc1", "document": "Test document 1", "page": 1}
        ]
        
        # Mock BGE-M3 service with error
        mock_bge_m3_service = Mock()
        mock_bge_m3_service.generate_dense_embedding = AsyncMock(side_effect=Exception("Generation error"))
        search_service.bge_m3_service = mock_bge_m3_service
        
        result = await search_service.prepare_bge_m3_documents(documents)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert "errors" in result[0]
        assert len(result[0]["errors"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
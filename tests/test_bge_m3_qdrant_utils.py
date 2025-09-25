"""
Tests for BGE-M3 Qdrant Utils
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional
import numpy as np

from src.app.utils.qdrant_utils import (
    create_bge_m3_collection_if_not_exists,
    bge_m3_hybrid_search_with_retry,
    convert_bge_m3_sparse_to_qdrant_format,
    prepare_bge_m3_query_embeddings,
    BGE_M3_QdrantUtils
)
from src.app.models.schemas import BGE_M3_SearchMode, BGE_M3_MultivectorStrategy
from src.app.settings import Settings, BGE_M3_Settings


class TestBGE_M3_QdrantUtils:
    """Test BGE-M3 Qdrant Utils functionality"""

@pytest.fixture
def mock_settings():
    """Mock settings fixture"""
    return Settings()

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client fixture"""
    return AsyncMock()

@pytest.fixture
def mock_collection_info():
    """Mock collection info fixture"""
    return {
        "status": "ok",
        "result": {
            "status": {"green": True},
            "vectors_count": 1000,
        }
    }

@pytest.fixture
def mock_search_results():
    """Mock search results fixture"""
    return [
        {
            "id": 1,
            "score": 0.95,
            "payload": {"content": "Test document 1"}
        },
        {
            "id": 2,
            "score": 0.85,
            "payload": {"content": "Test document 2"}
        }
    ]

@pytest.fixture
def mock_embeddings():
    """Mock embeddings fixture"""
    return {
        "dense": [0.1, 0.2, 0.3],
        "sparse": {"0": 0.5, "1": 0.3},
        "multi_vector": [[0.1, 0.2], [0.3, 0.4]]
    }

@pytest.fixture
def qdrant_utils(mock_qdrant_client):
    """BGE-M3 Qdrant Utils fixture"""
    with patch('src.app.services.bge_m3_service.BGE_M3_Service') as mock_service:
        mock_service_instance = Mock()
        mock_service.return_value = mock_service_instance
        # Pass the mock client directly instead of trying to set it on settings
        return BGE_M3_QdrantUtils(mock_qdrant_client)

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings data"""
        return {
            "dense": [0.1, 0.2, 0.3, 0.4, 0.5],
            "sparse": {"0": 0.5, "1": 0.3, "2": 0.2},
            "multi_vector": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        }

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results"""
        return [
            {
                "id": "doc1",
                "score": 0.95,
                "payload": {
                    "content": "Test document 1",
                    "metadata": {"source": "test1.pdf", "page": 1}
                }
            },
            {
                "id": "doc2",
                "score": 0.85,
                "payload": {
                    "content": "Test document 2",
                    "metadata": {"source": "test2.pdf", "page": 2}
                }
            }
        ]


class TestCreateBGE_M3Collection:
    """Test BGE-M3 collection creation functionality"""

    @pytest.mark.asyncio
    async def test_create_bge_m3_collection_success(self, mock_qdrant_client, mock_collection_info):
        """Test successful BGE-M3 collection creation"""
        # Mock the get_collection method to return the collection info
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        result = await create_bge_m3_collection_if_not_exists(
            mock_qdrant_client,
            "test_collection"
        )
        
        assert result is True
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_create_bge_m3_collection_not_exists(self, mock_qdrant_client, mock_collection_info):
        """Test BGE-M3 collection creation when collection doesn't exist"""
        # Mock collection not found
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")
        
        # Mock successful collection creation
        mock_qdrant_client.create_collection.return_value = None
        
        result = await create_bge_m3_collection_if_not_exists(
            mock_qdrant_client,
            "test_collection"
        )
        
        assert result is True
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_bge_m3_collection_creation_failure(self, mock_qdrant_client, mock_collection_info):
        """Test BGE-M3 collection creation when creation fails"""
        # Mock collection not found
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")
        
        # Mock collection creation failure
        mock_qdrant_client.create_collection.side_effect = Exception("Creation failed")
        
        result = await create_bge_m3_collection_if_not_exists(
            mock_qdrant_client,
            "test_collection",
            "http://localhost:6333"
        )
        
        assert result is False
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_bge_m3_collection_with_config(self, mock_qdrant_client, mock_collection_info):
        """Test BGE-M3 collection creation with custom configuration"""
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")
        mock_qdrant_client.create_collection.return_value = None
        
        config = {
            "vectors": {
                "dense": {"size": 1024, "distance": "Cosine"},
                "sparse": {"index": "Flat", "on_disk": True},
                "multi_vector": {"size": 1024, "distance": "Cosine"}
            },
            "shards": 2
        }
        
        result = await create_bge_m3_collection_if_not_exists(
            mock_qdrant_client,
            "test_collection",
            config=config
        )
        
        assert result is True
        # Verify create_collection was called with the custom config
        args = mock_qdrant_client.create_collection.call_args
        assert args[0][0] == "test_collection"
        assert args[1]["vectors"]["dense"]["size"] == 1024
        assert args[1]["shards"] == 2

    @pytest.mark.asyncio
    async def test_create_bge_m3_collection_already_exists(self, mock_qdrant_client, mock_collection_info):
        """Test BGE-M3 collection creation when collection already exists"""
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        result = await create_bge_m3_collection_if_not_exists(
            mock_qdrant_client,
            "test_collection"
        )
        
        assert result is True
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")
        # create_collection should not be called
        mock_qdrant_client.create_collection.assert_not_called()


class TestBGE_M3HybridSearchWithRetry:
    """Test BGE-M3 hybrid search with retry functionality"""

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_success(self, mock_qdrant_client, mock_search_results):
        """Test successful BGE-M3 hybrid search"""
        # Mock query_points to return proper QueryResponse
        from qdrant_client import models
        mock_qdrant_client.query_points.return_value = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=1,
                    score=0.95,
                    payload=mock_search_results[0]["payload"],
                    embedding=[],
                    metadata={},
                    document=""
                )
            ]
        )
        
        query_embeddings = {
            "dense": [0.1, 0.2, 0.3],
            "sparse": {"0": 0.5, "1": 0.3},
            "multi_vector": [[0.1, 0.2], [0.3, 0.4]]
        }
        
        result = await bge_m3_hybrid_search_with_retry(
            mock_qdrant_client,
            "test_collection",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"0": 0.5, "1": 0.3},
            multivector_query=[[0.1, 0.2], [0.3, 0.4]],
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            limit=10
        )
        
        assert result is not None
        assert len(result.points) > 0
        mock_qdrant_client.query_points.assert_called()

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_dense_only(self, mock_qdrant_client, mock_search_results):
        """Test BGE-M3 hybrid search with dense only mode"""
        from qdrant_client import models
        mock_qdrant_client.query_points.return_value = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=1,
                    score=0.95,
                    payload=mock_search_results[0]["payload"],
                    embedding=[],
                    metadata={},
                    document=""
                )
            ]
        )
        
        query_embeddings = {"dense": [0.1, 0.2, 0.3]}
        
        result = await bge_m3_hybrid_search_with_retry(
            mock_qdrant_client,
            "test_collection",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={},
            multivector_query=None,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            limit=5
        )
        
        assert result is not None
        mock_qdrant_client.query_points.assert_called()

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_sparse_only(self, mock_qdrant_client, mock_search_results):
        """Test BGE-M3 hybrid search with sparse only mode"""
        from qdrant_client import models
        mock_qdrant_client.query_points.return_value = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=1,
                    score=0.95,
                    payload=mock_search_results[0]["payload"],
                    embedding=[],
                    metadata={},
                    document=""
                )
            ]
        )
        
        query_embeddings = {"sparse": {"0": 0.5, "1": 0.3}}
        
        result = await bge_m3_hybrid_search_with_retry(
            mock_qdrant_client,
            "test_collection",
            dense_vector=[],
            sparse_vector={"0": 0.5, "1": 0.3},
            multivector_query=None,
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            limit=5
        )
        
        assert result is not None
        mock_qdrant_client.query_points.assert_called()

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_multivector_only(self, mock_qdrant_client, mock_search_results):
        """Test BGE-M3 hybrid search with multivector only mode"""
        from qdrant_client import models
        mock_qdrant_client.query_points.return_value = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=1,
                    score=0.95,
                    payload=mock_search_results[0]["payload"],
                    embedding=[],
                    metadata={},
                    document=""
                )
            ]
        )
        
        query_embeddings = {"multi_vector": [[0.1, 0.2], [0.3, 0.4]]}
        
        result = await bge_m3_hybrid_search_with_retry(
            mock_qdrant_client,
            "test_collection",
            dense_vector=[],
            sparse_vector={},
            multivector_query=[[0.1, 0.2], [0.3, 0.4]],
            alpha=0.0,
            beta=0.0,
            gamma=1.0,
            limit=5
        )
        
        assert result is not None
        mock_qdrant_client.query_points.assert_called()

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_retry_success(self, mock_qdrant_client, mock_search_results):
        """Test BGE-M3 hybrid search with retry mechanism"""
        from qdrant_client import models
        call_count = 0
        
        def mock_query_points(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return models.QueryResponse(
                points=[
                    models.ScoredPoint(
                        id=1,
                        score=0.95,
                        payload=mock_search_results[0]["payload"],
                        embedding=[],
                        metadata={},
                        document=""
                    )
                ]
            )
        
        mock_qdrant_client.query_points.side_effect = mock_query_points
        
        query_embeddings = {"dense": [0.1, 0.2, 0.3]}
        
        result = await bge_m3_hybrid_search_with_retry(
            mock_qdrant_client,
            "test_collection",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={},
            multivector_query=None,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            limit=10
        )
        
        assert result is not None
        assert call_count == 3  # Should retry 2 times then succeed

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_retry_failure(self, mock_qdrant_client):
        """Test BGE-M3 hybrid search when all retries fail"""
        mock_qdrant_client.query_points.side_effect = Exception("Persistent error")
        
        query_embeddings = {"dense": [0.1, 0.2, 0.3]}
        
        result = await bge_m3_hybrid_search_with_retry(
            mock_qdrant_client,
            "test_collection",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={},
            multivector_query=None,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            limit=10
        )
        
        assert result is None
        assert mock_qdrant_client.query_points.call_count == 2

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_filters(self, mock_qdrant_client, mock_search_results):
        """Test BGE-M3 hybrid search with metadata filters"""
        from qdrant_client import models
        mock_qdrant_client.query_points.return_value = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=1,
                    score=0.95,
                    payload=mock_search_results[0]["payload"],
                    embedding=[],
                    metadata={},
                    document=""
                )
            ]
        )
        
        query_embeddings = {"dense": [0.1, 0.2, 0.3]}
        filters = {
            "must": [
                {"key": "source", "match": {"value": "test.pdf"}}
            ]
        }
        
        result = await bge_m3_hybrid_search_with_retry(
            mock_qdrant_client,
            "test_collection",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={},
            multivector_query=None,
            query_filter=filters,
            limit=5
        )
        
        assert result is not None
        # Verify filters were passed to the query
        args = mock_qdrant_client.query_points.call_args
        assert "query_filter" in args[1]

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_score_threshold(self, mock_qdrant_client, mock_search_results):
        """Test BGE-M3 hybrid search with score threshold"""
        from qdrant_client import models
        mock_qdrant_client.query_points.return_value = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=1,
                    score=0.95,
                    payload=mock_search_results[0]["payload"],
                    embedding=[],
                    metadata={},
                    document=""
                )
            ]
        )
        
        query_embeddings = {"dense": [0.1, 0.2, 0.3]}
        
        result = await bge_m3_hybrid_search_with_retry(
            mock_qdrant_client,
            "test_collection",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={},
            multivector_query=None,
            score_threshold=0.8,
            limit=5
        )
        
        assert result is not None
        # Verify score threshold was applied
        args = mock_qdrant_client.query_points.call_args
        assert "score_threshold" in args[1]

    @pytest.mark.asyncio
    async def test_bge_m3_hybrid_search_with_pagination(self, mock_qdrant_client, mock_search_results):
        """Test BGE-M3 hybrid search with pagination"""
        from qdrant_client import models
        mock_qdrant_client.query_points.return_value = models.QueryResponse(
            points=[
                models.ScoredPoint(
                    id=1,
                    score=0.95,
                    payload=mock_search_results[0]["payload"],
                    embedding=[],
                    metadata={},
                    document=""
                )
            ]
        )
        
        query_embeddings = {"dense": [0.1, 0.2, 0.3]}
        
        result = await bge_m3_hybrid_search_with_retry(
            mock_qdrant_client,
            "test_collection",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={},
            multivector_query=None,
            limit=20
        )
        
        assert result is not None
        # Verify pagination parameters
        args = mock_qdrant_client.query_points.call_args
        assert "limit" in args[1]


class TestConvertBGE_M3SparseToQdrantFormat:
    """Test BGE-M3 sparse to Qdrant format conversion"""

    def test_convert_bge_m3_sparse_to_qdrant_format_basic(self):
        """Test basic sparse format conversion"""
        sparse_embedding = {"0": 0.5, "1": 0.3, "2": 0.2}
        
        result = convert_bge_m3_sparse_to_qdrant_format(sparse_embedding)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {"index": 0, "value": 0.5}
        assert result[1] == {"index": 1, "value": 0.3}
        assert result[2] == {"index": 2, "value": 0.2}

    def test_convert_bge_m3_sparse_to_qdrant_format_empty(self):
        """Test empty sparse format conversion"""
        sparse_embedding = {}
        
        result = convert_bge_m3_sparse_to_qdrant_format(sparse_embedding)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_convert_bge_m3_sparse_to_qdrant_format_unordered(self):
        """Test sparse format conversion with unordered indices"""
        sparse_embedding = {"5": 0.1, "2": 0.8, "0": 0.3}
        
        result = convert_bge_m3_sparse_to_qdrant_format(sparse_embedding)
        
        assert isinstance(result, list)
        assert len(result) == 3
        # Should be sorted by index
        assert result[0] == {"index": 0, "value": 0.3}
        assert result[1] == {"index": 2, "value": 0.8}
        assert result[2] == {"index": 5, "value": 0.1}

    def test_convert_bge_m3_sparse_to_qdrant_format_with_negative_values(self):
        """Test sparse format conversion with negative values"""
        sparse_embedding = {"0": 0.5, "1": -0.2, "2": 0.0}
        
        result = convert_bge_m3_sparse_to_qdrant_format(sparse_embedding)
        
        assert isinstance(result, list)
        # Should filter out 0.0 values but keep negative values
        assert len(result) == 2
        assert result[0] == {"index": 0, "value": 0.5}
        assert result[1] == {"index": 1, "value": -0.2}

    def test_convert_bge_m3_sparse_to_qdrant_format_large_values(self):
        """Test sparse format conversion with large values"""
        sparse_embedding = {"0": 1000.0, "1": 0.001, "2": 999.999}
        
        result = convert_bge_m3_sparse_to_qdrant_format(sparse_embedding)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {"index": 0, "value": 1000.0}
        assert result[1] == {"index": 1, "value": 0.001}
        assert result[2] == {"index": 2, "value": 999.999}

    def test_convert_bge_m3_sparse_to_qdrant_format_invalid_input(self):
        """Test sparse format conversion with invalid input"""
        # Test with None
        result = convert_bge_m3_sparse_to_qdrant_format(None)
        assert result == []
        
        # Test with string
        result = convert_bge_m3_sparse_to_qdrant_format("invalid")
        assert result == []
        
        # Test with list
        result = convert_bge_m3_sparse_to_qdrant_format([1, 2, 3])
        assert result == []


class TestPrepareBGE_M3QueryEmbeddings:
    """Test BGE-M3 query embeddings preparation"""

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_dense(self, qdrant_utils, mock_embeddings):
        """Test query embeddings preparation for dense mode"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_dense_embedding') as mock_generate:
            # Mock the async function to return the value directly
            mock_generate.return_value = mock_embeddings["dense"]
            
            result = await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service,
                dense_only=True
            )
            
            assert result["embeddings"] == {"dense": mock_embeddings["dense"]}
            mock_generate.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_sparse(self, qdrant_utils, mock_embeddings):
        """Test query embeddings preparation for sparse mode"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_sparse_embedding') as mock_generate:
            mock_generate.return_value = mock_embeddings["sparse"]
            
            result = await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service,
                sparse_only=True
            )
            
            assert result["embeddings"] == {"sparse": mock_embeddings["sparse"]}
            mock_generate.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_multivector(self, qdrant_utils, mock_embeddings):
        """Test query embeddings preparation for multivector mode"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_multivector_embedding') as mock_generate:
            mock_generate.return_value = mock_embeddings["multi_vector"]
            
            result = await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service,
                multivector_only=True
            )
            
            assert result["embeddings"] == {"multi_vector": mock_embeddings["multi_vector"]}
            mock_generate.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_hybrid(self, qdrant_utils, mock_embeddings):
        """Test query embeddings preparation for hybrid mode"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = {
                "embeddings": mock_embeddings,
                "errors": [],
                "text": query_text
            }
            
            result = await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service
            )
            
            assert result["embeddings"] == mock_embeddings
            mock_generate.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_with_cache(self, qdrant_utils, mock_embeddings):
        """Test query embeddings preparation with cache hit"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_dense_embedding') as mock_generate:
            # Mock cache hit
            qdrant_utils.bge_m3_service.cache_manager.get_embedding = AsyncMock(
                return_value={"dense": mock_embeddings["dense"]}
            )
            
            result = await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service,
                dense_only=True
            )
            
            assert result["embeddings"] == {"dense": mock_embeddings["dense"]}
            # Should not call generate_embedding due to cache hit
            mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_with_errors(self, qdrant_utils):
        """Test query embeddings preparation with errors"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_dense_embedding') as mock_generate:
            mock_generate.side_effect = Exception("Generation error")
            
            result = await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service,
                dense_only=True
            )
            
            assert result["embeddings"] == {}  # Should return fallback

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_invalid_mode(self, qdrant_utils):
        """Test query embeddings preparation with invalid mode"""
        query_text = "Test query"
        
        with pytest.raises(ValueError, match="Unknown search mode: invalid"):
            await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service,
                dense_only=True,
                sparse_only=True
            )

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_empty_query(self, qdrant_utils):
        """Test query embeddings preparation with empty query"""
        query_text = ""
        
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service,
                dense_only=True
            )

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_with_alpha_beta_gamma(self, qdrant_utils, mock_embeddings):
        """Test query embeddings preparation with hybrid weights"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = {
                "embeddings": mock_embeddings,
                "errors": [],
                "text": query_text
            }
            
            result = await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service
            )
            
            assert result["embeddings"] == mock_embeddings
            mock_generate.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_with_multivector_strategy(self, qdrant_utils, mock_embeddings):
        """Test query embeddings preparation with multivector strategy"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_multivector_embedding') as mock_generate:
            mock_generate.return_value = mock_embeddings["multi_vector"]
            
            result = await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service,
                multivector_only=True
            )
            
            assert result["embeddings"] == {"multi_vector": mock_embeddings["multi_vector"]}
            mock_generate.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_prepare_bge_m3_query_embeddings_with_normalization(self, qdrant_utils, mock_embeddings):
        """Test query embeddings preparation with normalization"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_dense_embedding') as mock_generate:
            mock_generate.return_value = mock_embeddings["dense"]
            
            result = await prepare_bge_m3_query_embeddings(
                query_text,
                bge_m3_service=qdrant_utils.bge_m3_service,
                dense_only=True
            )
            
            assert result["embeddings"] == {"dense": mock_embeddings["dense"]}
            mock_generate.assert_called_once_with(query_text)


class TestBGE_M3_QdrantUtilsIntegration:
    """Test BGE-M3 Qdrant Utils integration methods"""

    @pytest.mark.asyncio
    async def test_bge_m3_qdrant_utils_initialization(self, mock_settings):
        """Test BGE-M3 Qdrant Utils initialization"""
        with patch('src.app.services.bge_m3_service.BGE_M3_Service') as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            
            utils = BGE_M3_QdrantUtils(settings=mock_settings)
            
            assert utils.settings == mock_settings
            assert utils.bge_m3_service == mock_service_instance
            mock_service.assert_called_once_with(mock_settings)

    @pytest.mark.asyncio
    async def test_create_collection_with_utils(self, qdrant_utils, mock_qdrant_client, mock_collection_info):
        """Test collection creation using BGE-M3 Qdrant Utils"""
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        result = await qdrant_utils.create_collection("test_collection")
        
        assert result is True
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_hybrid_search_with_utils(self, qdrant_utils, mock_qdrant_client, mock_search_results):
        """Test hybrid search using BGE-M3 Qdrant Utils"""
        from qdrant_client import models
        # Create a simple mock response that matches the expected structure
        class MockPoint:
            def __init__(self, point_data):
                self.id = point_data["id"]
                self.score = point_data["score"]
                self.payload = point_data["payload"]
                self.vector = point_data.get("vector", {})
                self.order_value = point_data.get("order_value", None)
        
        class MockResponse:
            def __init__(self):
                self.points = [
                    MockPoint({
                        "id": 1,
                        "score": 0.95,
                        "payload": mock_search_results[0]["payload"],
                        "vector": {},
                        "order_value": None
                    })
                ]
        
        mock_qdrant_client.query_points.return_value = MockResponse()
        
        query_embeddings = {"dense": [0.1, 0.2, 0.3]}
        
        result = await qdrant_utils.hybrid_search(
            "test_collection",
            query_embeddings,
            search_mode="dense",
            top_k=5
        )
        
        assert result is not None
        assert "results" in result
        mock_qdrant_client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_query_embeddings_with_utils(self, qdrant_utils, mock_embeddings):
        """Test query embeddings preparation using BGE-M3 Qdrant Utils"""
        query_text = "Test query"
        
        with patch.object(qdrant_utils.bge_m3_service, 'generate_dense_embedding') as mock_generate:
            mock_generate.return_value = mock_embeddings["dense"]
            
            result = await qdrant_utils.prepare_query_embeddings(
                query_text,
                dense_only=True
            )
            
            assert result["embeddings"] == {"dense": mock_embeddings["dense"]}
            mock_generate.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_get_collection_info(self, qdrant_utils, mock_qdrant_client, mock_collection_info):
        """Test getting collection information"""
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        result = await qdrant_utils.get_collection_info("test_collection")
        
        assert result == mock_collection_info
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_get_collection_info_not_found(self, qdrant_utils, mock_qdrant_client):
        """Test getting collection info when collection doesn't exist"""
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")
        
        result = await qdrant_utils.get_collection_info("nonexistent_collection")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_collection(self, qdrant_utils, mock_qdrant_client):
        """Test deleting a collection"""
        mock_qdrant_client.delete_collection.return_value = None
        
        result = await qdrant_utils.delete_collection("test_collection")
        
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_delete_collection_failure(self, qdrant_utils, mock_qdrant_client):
        """Test deleting a collection when deletion fails"""
        mock_qdrant_client.delete_collection.side_effect = Exception("Deletion failed")
        
        result = await qdrant_utils.delete_collection("test_collection")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, qdrant_utils, mock_qdrant_client):
        """Test getting collection statistics"""
        mock_stats = {
            "status": "ok",
            "result": {
                "vector_count": 1000,
                "segment_count": 2,
                "config": {
                    "params": {
                        "vectors": {
                            "size": 1024,
                            "distance": "Cosine"
                        }
                    }
                }
            }
        }
        
        mock_qdrant_client.get_collection_stats.return_value = mock_stats
        
        result = await qdrant_utils.get_collection_stats("test_collection")
        
        assert result == mock_stats
        mock_qdrant_client.get_collection_stats.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_get_collection_stats_failure(self, qdrant_utils, mock_qdrant_client):
        """Test getting collection stats when it fails"""
        mock_qdrant_client.get_collection_stats.side_effect = Exception("Stats failed")
        
        result = await qdrant_utils.get_collection_stats("test_collection")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_optimize_collection(self, qdrant_utils, mock_qdrant_client):
        """Test optimizing a collection"""
        mock_qdrant_client.optimize.return_value = None
        
        result = await qdrant_utils.optimize_collection("test_collection")
        
        assert result is True
        mock_qdrant_client.optimize.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_optimize_collection_failure(self, qdrant_utils, mock_qdrant_client):
        """Test optimizing a collection when it fails"""
        mock_qdrant_client.optimize.side_effect = Exception("Optimization failed")
        
        result = await qdrant_utils.optimize_collection("test_collection")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check(self, qdrant_utils, mock_qdrant_client, mock_collection_info):
        """Test collection health check"""
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        result = await qdrant_utils.health_check("test_collection")
        
        assert result["status"] == "healthy"
        assert result["collection_exists"] is True
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_health_check_collection_not_found(self, qdrant_utils, mock_qdrant_client):
        """Test health check when collection doesn't exist"""
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")
        
        result = await qdrant_utils.health_check("test_collection")
        
        assert result["status"] == "unhealthy"
        assert result["collection_exists"] is False

    @pytest.mark.asyncio
    async def test_health_check_server_error(self, qdrant_utils, mock_qdrant_client):
        """Test health check when server returns error"""
        mock_qdrant_client.get_collection.side_effect = Exception("Server error")
        
        result = await qdrant_utils.health_check("test_collection")
        
        assert result["status"] == "unhealthy"
        assert result["error"] == "Server error"

    @pytest.mark.asyncio
    async def test_batch_upsert(self, qdrant_utils, mock_qdrant_client):
        """Test batch upsert operation"""
        points = [
            {
                "id": 1,
                "vector": [0.1, 0.2, 0.3],
                "payload": {"content": "Test document 1"}
            },
            {
                "id": 2,
                "vector": [0.4, 0.5, 0.6],
                "payload": {"content": "Test document 2"}
            }
        ]
        
        mock_qdrant_client.upsert.return_value = None
        
        result = await qdrant_utils.batch_upsert("test_collection", points)
        
        assert result is True
        mock_qdrant_client.upsert.assert_called_once()
        args = mock_qdrant_client.upsert.call_args
        assert args[1]["collection_name"] == "test_collection"
        assert len(args[1]["points"]) == 2

    @pytest.mark.asyncio
    async def test_batch_upsert_with_retry(self, qdrant_utils, mock_qdrant_client):
        """Test batch upsert with retry mechanism"""
        points = [{"id": 1, "vector": [0.1, 0.2, 0.3], "payload": {"content": "Test"}}]
        
        call_count = 0
        
        def mock_upsert(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return None
        
        mock_qdrant_client.upsert.side_effect = mock_upsert
        
        result = await qdrant_utils.batch_upsert("test_collection", points, max_retries=3)
        
        assert result is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_batch_upsert_failure(self, qdrant_utils, mock_qdrant_client):
        """Test batch upsert when all retries fail"""
        points = [{"id": 1, "vector": [0.1, 0.2, 0.3], "payload": {"content": "Test"}}]
        
        mock_qdrant_client.upsert.side_effect = Exception("Persistent error")
        
        result = await qdrant_utils.batch_upsert("test_collection", points, max_retries=2)
        
        assert result is False
        assert mock_qdrant_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_delete(self, qdrant_utils, mock_qdrant_client):
        """Test batch delete operation"""
        point_ids = [1, 2, 3]
        
        mock_qdrant_client.delete.return_value = None
        
        result = await qdrant_utils.batch_delete("test_collection", point_ids)
        
        assert result is True
        mock_qdrant_client.delete.assert_called_once()
        args = mock_qdrant_client.delete.call_args
        assert args[1]["collection_name"] == "test_collection"
        assert "points_selector" in args[1]

    @pytest.mark.asyncio
    async def test_batch_delete_failure(self, qdrant_utils, mock_qdrant_client):
        """Test batch delete when it fails"""
        point_ids = [1, 2, 3]
        
        mock_qdrant_client.delete.side_effect = Exception("Delete failed")
        
        result = await qdrant_utils.batch_delete("test_collection", point_ids)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_scroll_collection(self, qdrant_utils, mock_qdrant_client):
        """Test scrolling through collection"""
        mock_response = ([
            {
                "id": 1,
                "vector": [0.1, 0.2, 0.3],
                "payload": {"content": "Test document 1"}
            },
            {
                "id": 2,
                "vector": [0.4, 0.5, 0.6],
                "payload": {"content": "Test document 2"}
            }
        ], None)
        
        mock_qdrant_client.scroll.return_value = mock_response
        
        result = await qdrant_utils.scroll_collection("test_collection", limit=10)
        
        assert result == mock_response
        mock_qdrant_client.scroll.assert_called_once_with(
            collection_name="test_collection",
            limit=10,
            with_payload=True,
            with_vectors=True
        )

    @pytest.mark.asyncio
    async def test_scroll_collection_with_filter(self, qdrant_utils, mock_qdrant_client):
        """Test scrolling with filter"""
        mock_response = ([{"id": 1, "payload": {"content": "Test"}}], None)
        filters = {"must": [{"key": "type", "match": {"value": "document"}}]}
        
        mock_qdrant_client.scroll.return_value = mock_response
        
        result = await qdrant_utils.scroll_collection(
            "test_collection",
            limit=10,
            filters=filters
        )
        
        assert result == mock_response
        args = mock_qdrant_client.scroll.call_args
        assert "scroll_filter" in args[1]

    @pytest.mark.asyncio
    async def test_scroll_collection_failure(self, qdrant_utils, mock_qdrant_client):
        """Test scrolling when it fails"""
        mock_qdrant_client.scroll.side_effect = Exception("Scroll failed")
        
        result = await qdrant_utils.scroll_collection("test_collection")
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
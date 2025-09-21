"""
Tests for search service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

from src.app.services.search_service import SearchService
from src.app.models.schemas import SearchResult, SearchResultItem


class TestSearchService:
    """Test search service"""

    @pytest.fixture
    def search_service(self):
        """Create search service instance"""
        return SearchService(
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
            dense_model="test-dense-model",
            sparse_model="test-sparse-model",
            hybrid_weight=0.5,
        )

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client"""
        mock_client = Mock()
        mock_client.query_points = AsyncMock()
        mock_client.scroll = AsyncMock()
        mock_client.count = AsyncMock()
        return mock_client

    @pytest.fixture
    def mock_dense_embedding(self):
        """Mock dense embedding"""
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.fixture
    def mock_sparse_embedding(self):
        """Mock sparse embedding"""
        return {"keyword1": 0.8, "keyword2": 0.6, "keyword3": 0.4}

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results"""
        return [
            SearchResultItem(
                id="result1",
                score=0.95,
                content="First search result content",
                metadata={
                    "page_number": 1,
                    "document": "test.pdf",
                    "chunk_type": "text",
                },
            ),
            SearchResultItem(
                id="result2",
                score=0.85,
                content="Second search result content",
                metadata={
                    "page_number": 2,
                    "document": "test.pdf",
                    "chunk_type": "table",
                },
            ),
            SearchResultItem(
                id="result3",
                score=0.75,
                content="Third search result content",
                metadata={
                    "page_number": 1,
                    "document": "test.pdf",
                    "chunk_type": "image",
                },
            ),
        ]

    def test_search_service_initialization(self, search_service):
        """Test search service initialization"""
        assert search_service.qdrant_url == "http://localhost:6333"
        assert search_service.collection_name == "test_collection"
        assert search_service.dense_model == "test-dense-model"
        assert search_service.sparse_model == "test-sparse-model"
        assert search_service.hybrid_weight == 0.5
        assert search_service.logger is not None

    @pytest.mark.asyncio
    async def test_create_qdrant_client_success(self, search_service):
        """Test successful Qdrant client creation"""
        with patch("qdrant_client.QdrantClient") as mock_qdrant:
            mock_client = Mock()
            mock_qdrant.return_value = mock_client

            client = await search_service._create_qdrant_client()

            assert client is not None
            mock_qdrant.assert_called_once_with(url="http://localhost:6333", timeout=30)

    @pytest.mark.asyncio
    async def test_create_qdrant_client_error(self, search_service):
        """Test Qdrant client creation with error"""
        with patch("qdrant_client.QdrantClient") as mock_qdrant:
            mock_qdrant.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                await search_service._create_qdrant_client()

    @pytest.mark.asyncio
    async def test_dense_search_success(
        self, search_service, mock_qdrant_client, mock_dense_embedding
    ):
        """Test successful dense search"""
        # Mock Qdrant response
        mock_response = Mock()
        mock_response.points = [
            Mock(
                id="result1",
                score=0.95,
                payload={
                    "content": "First search result",
                    "page_number": 1,
                    "document": "test.pdf",
                    "chunk_type": "text",
                },
            ),
            Mock(
                id="result2",
                score=0.85,
                payload={
                    "content": "Second search result",
                    "page_number": 2,
                    "document": "test.pdf",
                    "chunk_type": "table",
                },
            ),
        ]
        mock_qdrant_client.query_points.return_value = mock_response

        # Test search
        results = await search_service.dense_search(
            query_embedding=mock_dense_embedding, limit=10, score_threshold=0.5
        )

        assert isinstance(results, list)
        assert len(results) == 2

        # Check result structure
        for result in results:
            assert isinstance(result, SearchResultItem)
            assert result.id is not None
            assert result.score >= 0.5
            assert result.content is not None
            assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_dense_search_no_results(
        self, search_service, mock_qdrant_client, mock_dense_embedding
    ):
        """Test dense search with no results"""
        # Mock empty response
        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response

        results = await search_service.dense_search(
            query_embedding=mock_dense_embedding, limit=10, score_threshold=0.5
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_dense_search_error(
        self, search_service, mock_qdrant_client, mock_dense_embedding
    ):
        """Test dense search with error"""
        mock_qdrant_client.query_points.side_effect = Exception("Search failed")

        with pytest.raises(Exception):
            await search_service.dense_search(
                query_embedding=mock_dense_embedding, limit=10, score_threshold=0.5
            )

    @pytest.mark.asyncio
    async def test_sparse_search_success(
        self, search_service, mock_qdrant_client, mock_sparse_embedding
    ):
        """Test successful sparse search"""
        # Mock Qdrant response
        mock_response = Mock()
        mock_response.points = [
            Mock(
                id="result1",
                score=0.95,
                payload={
                    "content": "First search result",
                    "page_number": 1,
                    "document": "test.pdf",
                    "chunk_type": "text",
                },
            ),
            Mock(
                id="result2",
                score=0.85,
                payload={
                    "content": "Second search result",
                    "page_number": 2,
                    "document": "test.pdf",
                    "chunk_type": "table",
                },
            ),
        ]
        mock_qdrant_client.query_points.return_value = mock_response

        # Test search
        results = await search_service.sparse_search(
            query_embedding=mock_sparse_embedding, limit=10, score_threshold=0.5
        )

        assert isinstance(results, list)
        assert len(results) == 2

        # Check result structure
        for result in results:
            assert isinstance(result, SearchResultItem)
            assert result.id is not None
            assert result.score >= 0.5
            assert result.content is not None
            assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_sparse_search_no_results(
        self, search_service, mock_qdrant_client, mock_sparse_embedding
    ):
        """Test sparse search with no results"""
        # Mock empty response
        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response

        results = await search_service.sparse_search(
            query_embedding=mock_sparse_embedding, limit=10, score_threshold=0.5
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_sparse_search_error(
        self, search_service, mock_qdrant_client, mock_sparse_embedding
    ):
        """Test sparse search with error"""
        mock_qdrant_client.query_points.side_effect = Exception("Search failed")

        with pytest.raises(Exception):
            await search_service.sparse_search(
                query_embedding=mock_sparse_embedding, limit=10, score_threshold=0.5
            )

    @pytest.mark.asyncio
    async def test_hybrid_search_success(
        self,
        search_service,
        mock_qdrant_client,
        mock_dense_embedding,
        mock_sparse_embedding,
    ):
        """Test successful hybrid search"""
        # Mock Qdrant responses
        mock_dense_response = Mock()
        mock_dense_response.points = [
            Mock(
                id="dense1",
                score=0.95,
                payload={
                    "content": "Dense search result",
                    "page_number": 1,
                    "document": "test.pdf",
                    "chunk_type": "text",
                },
            )
        ]

        mock_sparse_response = Mock()
        mock_sparse_response.points = [
            Mock(
                id="sparse1",
                score=0.85,
                payload={
                    "content": "Sparse search result",
                    "page_number": 2,
                    "document": "test.pdf",
                    "chunk_type": "table",
                },
            )
        ]

        mock_qdrant_client.query_points.side_effect = [
            mock_dense_response,
            mock_sparse_response,
        ]

        # Test search
        results = await search_service.hybrid_search(
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
            limit=10,
            score_threshold=0.5,
            hybrid_weight=0.7,
        )

        assert isinstance(results, list)
        assert len(results) == 2

        # Check result structure
        for result in results:
            assert isinstance(result, SearchResultItem)
            assert result.id is not None
            assert result.score >= 0.5
            assert result.content is not None
            assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_hybrid_search_no_results(
        self,
        search_service,
        mock_qdrant_client,
        mock_dense_embedding,
        mock_sparse_embedding,
    ):
        """Test hybrid search with no results"""
        # Mock empty responses
        mock_empty_response = Mock()
        mock_empty_response.points = []

        mock_qdrant_client.query_points.side_effect = [
            mock_empty_response,
            mock_empty_response,
        ]

        results = await search_service.hybrid_search(
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
            limit=10,
            score_threshold=0.5,
            hybrid_weight=0.7,
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_hybrid_search_error(
        self,
        search_service,
        mock_qdrant_client,
        mock_dense_embedding,
        mock_sparse_embedding,
    ):
        """Test hybrid search with error"""
        mock_qdrant_client.query_points.side_effect = Exception("Search failed")

        with pytest.raises(Exception):
            await search_service.hybrid_search(
                dense_embedding=mock_dense_embedding,
                sparse_embedding=mock_sparse_embedding,
                limit=10,
                score_threshold=0.5,
                hybrid_weight=0.7,
            )

    @pytest.mark.asyncio
    async def test_search_with_filters_success(
        self, search_service, mock_qdrant_client, mock_dense_embedding
    ):
        """Test search with filters"""
        # Mock Qdrant response
        mock_response = Mock()
        mock_response.points = [
            Mock(
                id="result1",
                score=0.95,
                payload={
                    "content": "Filtered search result",
                    "page_number": 1,
                    "document": "test.pdf",
                    "chunk_type": "text",
                },
            )
        ]
        mock_qdrant_client.query_points.return_value = mock_response

        # Test search with filters
        results = await search_service.search_with_filters(
            query_embedding=mock_dense_embedding,
            limit=10,
            score_threshold=0.5,
            filters={"document": "test.pdf", "page_number": 1, "chunk_type": "text"},
        )

        assert isinstance(results, list)
        assert len(results) == 1

        # Check result structure
        result = results[0]
        assert isinstance(result, SearchResultItem)
        assert result.id is not None
        assert result.score >= 0.5
        assert result.content is not None
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_search_with_filters_no_results(
        self, search_service, mock_qdrant_client, mock_dense_embedding
    ):
        """Test search with filters - no results"""
        # Mock empty response
        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response

        results = await search_service.search_with_filters(
            query_embedding=mock_dense_embedding,
            limit=10,
            score_threshold=0.5,
            filters={"document": "non_existent.pdf", "page_number": 999},
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_count_documents_success(self, search_service, mock_qdrant_client):
        """Test document count"""
        # Mock Qdrant response
        mock_response = Mock()
        mock_response.count = 42
        mock_qdrant_client.count.return_value = mock_response

        count = await search_service.count_documents()

        assert isinstance(count, int)
        assert count == 42

    @pytest.mark.asyncio
    async def test_count_documents_error(self, search_service, mock_qdrant_client):
        """Test document count with error"""
        mock_qdrant_client.count.side_effect = Exception("Count failed")

        with pytest.raises(Exception):
            await search_service.count_documents()

    @pytest.mark.asyncio
    async def test_get_document_metadata_success(
        self, search_service, mock_qdrant_client
    ):
        """Test document metadata retrieval"""
        # Mock Qdrant response
        mock_response = Mock()
        mock_response.points = [
            Mock(
                id="doc1",
                payload={
                    "document": "test.pdf",
                    "page_number": 1,
                    "chunk_type": "text",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ),
            Mock(
                id="doc2",
                payload={
                    "document": "test.pdf",
                    "page_number": 2,
                    "chunk_type": "table",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ),
        ]
        mock_qdrant_client.scroll.return_value = mock_response

        metadata = await search_service.get_document_metadata()

        assert isinstance(metadata, list)
        assert len(metadata) == 2

        # Check metadata structure
        for item in metadata:
            assert "document" in item
            assert "page_number" in item
            assert "chunk_type" in item
            assert "created_at" in item

    @pytest.mark.asyncio
    async def test_get_document_metadata_error(
        self, search_service, mock_qdrant_client
    ):
        """Test document metadata retrieval with error"""
        mock_qdrant_client.scroll.side_effect = Exception("Metadata retrieval failed")

        with pytest.raises(Exception):
            await search_service.get_document_metadata()

    def test_combine_search_results(self, search_service, mock_search_results):
        """Test search result combination"""
        # Test with different weights
        combined_results = search_service._combine_search_results(
            mock_search_results, dense_weight=0.7, sparse_weight=0.3
        )

        assert isinstance(combined_results, list)
        assert len(combined_results) == len(mock_search_results)

        # Check that results are sorted by score
        for i in range(len(combined_results) - 1):
            assert combined_results[i].score >= combined_results[i + 1].score

    def test_combine_search_results_empty(self, search_service):
        """Test search result combination with empty results"""
        combined_results = search_service._combine_search_results([], 0.7, 0.3)

        assert isinstance(combined_results, list)
        assert len(combined_results) == 0

    def test_combine_search_results_invalid_weights(
        self, search_service, mock_search_results
    ):
        """Test search result combination with invalid weights"""
        with pytest.raises(ValueError):
            search_service._combine_search_results(
                mock_search_results,
                dense_weight=1.5,  # Invalid weight
                sparse_weight=0.3,
            )

    def test_create_search_payload(self, search_service):
        """Test search payload creation"""
        payload = search_service._create_search_payload(
            dense_embedding=[0.1, 0.2, 0.3],
            sparse_embedding={"keyword1": 0.8},
            filters={"document": "test.pdf"},
        )

        assert isinstance(payload, dict)
        assert "query" in payload
        assert "filter" in payload
        assert "limit" in payload
        assert "score_threshold" in payload

        # Check dense embedding
        assert "dense" in payload["query"]
        assert len(payload["query"]["dense"]) == 3

        # Check sparse embedding
        assert "sparse" in payload["query"]
        assert "keyword1" in payload["query"]["sparse"]
        assert payload["query"]["sparse"]["keyword1"] == 0.8

        # Check filters
        assert payload["filter"]["must"][0].key == "document"
        assert payload["filter"]["must"][0].match.value == "test.pdf"

    def test_create_search_payload_no_filters(self, search_service):
        """Test search payload creation without filters"""
        payload = search_service._create_search_payload(
            dense_embedding=[0.1, 0.2, 0.3], sparse_embedding={"keyword1": 0.8}
        )

        assert isinstance(payload, dict)
        assert "query" in payload
        assert "filter" in payload
        assert len(payload["filter"]["must"]) == 0

    def test_create_search_payload_invalid_embeddings(self, search_service):
        """Test search payload creation with invalid embeddings"""
        with pytest.raises(ValueError):
            search_service._create_search_payload(
                dense_embedding=[],  # Empty dense embedding
                sparse_embedding={"keyword1": 0.8},
            )

        with pytest.raises(ValueError):
            search_service._create_search_payload(
                dense_embedding=[0.1, 0.2, 0.3],
                sparse_embedding={},  # Empty sparse embedding
            )


class TestSearchServiceIntegration:
    """Integration tests for search service"""

    @pytest.fixture
    def complex_search_results(self):
        """Create complex search results for integration testing"""
        return [
            SearchResultItem(
                id="result1",
                score=0.95,
                content="This is a comprehensive text chunk about machine learning algorithms and their applications in modern AI systems.",
                metadata={
                    "page_number": 1,
                    "document": "ml_algorithms.pdf",
                    "chunk_type": "text",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ),
            SearchResultItem(
                id="result2",
                score=0.92,
                content="Table showing comparison of different neural network architectures and their performance metrics.",
                metadata={
                    "page_number": 2,
                    "document": "ml_algorithms.pdf",
                    "chunk_type": "table",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ),
            SearchResultItem(
                id="result3",
                score=0.88,
                content="Visualization of neural network architecture with detailed layer information.",
                metadata={
                    "page_number": 3,
                    "document": "ml_algorithms.pdf",
                    "chunk_type": "image",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ),
            SearchResultItem(
                id="result4",
                score=0.85,
                content="Another text chunk discussing deep learning frameworks and their implementation details.",
                metadata={
                    "page_number": 4,
                    "document": "ml_algorithms.pdf",
                    "chunk_type": "text",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ),
            SearchResultItem(
                id="result5",
                score=0.82,
                content="Table summarizing evaluation metrics for different machine learning models.",
                metadata={
                    "page_number": 5,
                    "document": "ml_algorithms.pdf",
                    "chunk_type": "table",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ),
        ]

    def test_full_search_workflow(self, search_service, complex_search_results):
        """Test complete search workflow"""
        # Test result combination
        combined_results = search_service._combine_search_results(
            complex_search_results, dense_weight=0.6, sparse_weight=0.4
        )

        # Verify results
        assert isinstance(combined_results, list)
        assert len(combined_results) == len(complex_search_results)

        # Check sorting
        for i in range(len(combined_results) - 1):
            assert combined_results[i].score >= combined_results[i + 1].score

        # Check result structure
        for result in combined_results:
            assert isinstance(result, SearchResultItem)
            assert result.id is not None
            assert result.score >= 0.0
            assert result.content is not None
            assert result.metadata is not None
            assert "page_number" in result.metadata
            assert "document" in result.metadata
            assert "chunk_type" in result.metadata

    def test_search_result_filtering(self, search_service, complex_search_results):
        """Test search result filtering by chunk type"""
        # Filter for text chunks only
        text_results = [
            r for r in complex_search_results if r.metadata["chunk_type"] == "text"
        ]

        assert len(text_results) == 2
        assert all(r.metadata["chunk_type"] == "text" for r in text_results)

        # Filter for table chunks only
        table_results = [
            r for r in complex_search_results if r.metadata["chunk_type"] == "table"
        ]

        assert len(table_results) == 2
        assert all(r.metadata["chunk_type"] == "table" for r in table_results)

        # Filter for image chunks only
        image_results = [
            r for r in complex_search_results if r.metadata["chunk_type"] == "image"
        ]

        assert len(image_results) == 1
        assert all(r.metadata["chunk_type"] == "image" for r in image_results)

    def test_search_result_pagination(self, search_service, complex_search_results):
        """Test search result pagination"""
        # Test first page
        page1 = complex_search_results[:3]
        assert len(page1) == 3

        # Test second page
        page2 = complex_search_results[3:]
        assert len(page2) == 2

        # Test combined results
        all_results = page1 + page2
        assert len(all_results) == len(complex_search_results)

    def test_search_result_metadata_analysis(
        self, search_service, complex_search_results
    ):
        """Test search result metadata analysis"""
        # Analyze chunk types
        chunk_types = {}
        for result in complex_search_results:
            chunk_type = result.metadata["chunk_type"]
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = []
            chunk_types[chunk_type].append(result)

        # Verify chunk type distribution
        assert "text" in chunk_types
        assert "table" in chunk_types
        assert "image" in chunk_types

        # Verify document distribution
        documents = {}
        for result in complex_search_results:
            document = result.metadata["document"]
            if document not in documents:
                documents[document] = []
            documents[document].append(result)

        assert len(documents) == 1  # All results from same document
        assert "ml_algorithms.pdf" in documents

    @pytest.mark.asyncio
    async def test_concurrent_search_operations(
        self,
        search_service,
        mock_qdrant_client,
        mock_dense_embedding,
        mock_sparse_embedding,
    ):
        """Test concurrent search operations"""
        # Mock Qdrant responses
        mock_response = Mock()
        mock_response.points = [
            Mock(
                id="result1",
                score=0.95,
                payload={
                    "content": "Concurrent search result",
                    "page_number": 1,
                    "document": "test.pdf",
                    "chunk_type": "text",
                },
            )
        ]
        mock_qdrant_client.query_points.return_value = mock_response

        # Test concurrent searches
        tasks = [
            search_service.dense_search(mock_dense_embedding, 5, 0.5),
            search_service.sparse_search(mock_sparse_embedding, 5, 0.5),
            search_service.hybrid_search(
                mock_dense_embedding, mock_sparse_embedding, 5, 0.5, 0.7
            ),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify results
        successful_results = [r for r in results if isinstance(r, list)]
        assert len(successful_results) == 3

        for result in successful_results:
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], SearchResultItem)


if __name__ == "__main__":
    pytest.main([__file__])

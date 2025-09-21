"""
Tests for utility functions
"""

import pytest
import torch
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from PIL import Image
import numpy as np
import asyncio

from src.app.utils import qdrant_utils


class TestQdrantUtils:
    """Test Qdrant utility functions"""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client"""
        mock_client = Mock()
        # Mock the query_points method as async
        mock_query_points = AsyncMock()
        mock_client.query_points = mock_query_points
        return mock_client

    def test_create_payload_filter_session_id(self):
        """Test payload filter creation with session ID"""
        filter_dict = qdrant_utils.create_payload_filter(session_id="test-session")

        assert filter_dict is not None
        assert hasattr(filter_dict, "must")
        assert len(filter_dict.must) == 1
        assert filter_dict.must[0].key == "session_id"
        assert filter_dict.must[0].match.value == "test-session"

    def test_create_payload_filter_document(self):
        """Test payload filter creation with document name"""
        filter_dict = qdrant_utils.create_payload_filter(document="test.pdf")

        assert filter_dict is not None
        assert hasattr(filter_dict, "must")
        assert len(filter_dict.must) == 1
        assert filter_dict.must[0].key == "document"
        assert filter_dict.must[0].match.value == "test.pdf"

    def test_create_payload_filter_combined(self):
        """Test payload filter creation with multiple filters"""
        filter_dict = qdrant_utils.create_payload_filter(
            session_id="test-session", document="test.pdf"
        )

        assert filter_dict is not None
        assert hasattr(filter_dict, "must")
        assert len(filter_dict.must) == 2

        # Check session_id filter
        session_filter = next(f for f in filter_dict.must if f.key == "session_id")
        assert session_filter.match.value == "test-session"

        # Check document filter
        doc_filter = next(f for f in filter_dict.must if f.key == "document")
        assert doc_filter.match.value == "test.pdf"

    def test_create_payload_filter_no_filters(self):
        """Test payload filter creation with no filters"""
        filter_dict = qdrant_utils.create_payload_filter()

        assert filter_dict is not None
        assert hasattr(filter_dict, "must")
        assert len(filter_dict.must) == 0

    @pytest.mark.asyncio
    async def test_search_with_retry_success(self, mock_qdrant_client):
        """Test successful search with retry"""
        # Mock successful response
        mock_response = Mock()
        mock_response.points = [
            Mock(id="test-id", score=0.9, payload={"document": "test.pdf", "page": 1})
        ]
        mock_qdrant_client.query_points.return_value = mock_response

        # Test search
        result = await qdrant_utils.search_with_retry(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            query_vector=[0.1] * 768,
            limit=10,
        )

        assert result == mock_response
        mock_qdrant_client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_retry_failure(self, mock_qdrant_client):
        """Test search with retry on failure"""
        # Mock initial failure, then success
        mock_qdrant_client.query_points.side_effect = [
            Exception("Search failed"),
            Mock(points=[]),
        ]

        # Test search
        result = await qdrant_utils.search_with_retry(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            query_vector=[0.1] * 768,
            limit=10,
        )

        assert result.points == []
        assert mock_qdrant_client.query_points.call_count == 2

    @pytest.mark.asyncio
    async def test_search_with_retry_max_attempts(self, mock_qdrant_client):
        """Test search with retry reaching max attempts"""

        # Mock persistent failure
        async def mock_query_points(*args, **kwargs):
            raise Exception("Search failed")

        # Use AsyncMock consistently
        mock_qdrant_client.query_points = AsyncMock(side_effect=mock_query_points)

        # Test search
        with pytest.raises(Exception):
            await qdrant_utils.search_with_retry(
                qdrant_client=mock_qdrant_client,
                collection_name="test_collection",
                query_vector=[0.1] * 768,
                limit=10,
            )

        assert mock_qdrant_client.query_points.call_count == 3

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, mock_qdrant_client):
        """Test search with score threshold"""
        # Mock response with scores
        mock_response = Mock()
        mock_response.points = [
            Mock(id="id1", score=0.9, payload={}),
            Mock(id="id2", score=0.7, payload={}),
            Mock(id="id3", score=0.5, payload={}),
        ]
        mock_qdrant_client.query_points.return_value = mock_response

        # Test search with threshold
        result = await qdrant_utils.search_with_retry(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            query_vector=[0.1] * 768,
            limit=10,
            score_threshold=0.6,
        )

        # Should only return points with score >= threshold
        assert len(result.points) == 2
        assert result.points[0].score >= 0.6
        assert result.points[1].score >= 0.6


class TestNewUtilityFunctions:
    """Test utility functions for new services"""

    def test_text_chunking_utils(self):
        """Test text chunking utility functions"""
        from src.app.services.text_preprocessor import TextPreprocessor

        # Create mock text elements
        mock_elements = [
            Mock(type="text", bbox=[0, 0, 100, 20], content="First line of text"),
            Mock(type="text", bbox=[0, 15, 100, 35], content="Second line of text"),
            Mock(type="text", bbox=[0, 40, 100, 60], content="Third line of text"),
        ]

        # Test chunk creation
        preprocessor = TextPreprocessor()
        chunks = preprocessor._process_text_elements(mock_elements, 1)

        assert len(chunks) == 2  # First two lines should be grouped together
        assert all("content" in chunk for chunk in chunks)
        assert all("type" in chunk for chunk in chunks)
        assert all("page_number" in chunk for chunk in chunks)
        assert all(chunk["type"] == "text" for chunk in chunks)

    def test_bbox_calculation(self):
        """Test bounding box calculation utility"""
        from src.app.services.text_preprocessor import TextPreprocessor

        # Create mock elements with different bounding boxes
        mock_elements = [
            Mock(bbox=[0, 0, 100, 20]),
            Mock(bbox=[10, 10, 110, 30]),
            Mock(bbox=[20, 20, 120, 40]),
        ]

        preprocessor = TextPreprocessor()
        avg_bbox = preprocessor._calculate_average_bbox(mock_elements)

        assert len(avg_bbox) == 4
        assert avg_bbox[0] == 10.0  # Average x0
        assert avg_bbox[1] == 10.0  # Average y0
        assert avg_bbox[2] == 110.0  # Average x1
        assert avg_bbox[3] == 30.0  # Average y1

    def test_sparse_embedding_generation(self):
        """Test sparse embedding generation"""
        from src.app.services.text_preprocessor import TextPreprocessor

        preprocessor = TextPreprocessor()

        # Test with simple text
        test_text = "This is a test document with some important keywords"
        sparse_embedding = preprocessor.generate_sparse_embedding(test_text)

        assert isinstance(sparse_embedding, dict)
        assert len(sparse_embedding) > 0

        # Check that values are between 0 and 1
        for value in sparse_embedding.values():
            assert 0 <= value <= 1

    def test_text_merging(self):
        """Test text merging utility"""
        from src.app.services.text_preprocessor import TextPreprocessor
        from src.app.models.schemas import ExtractedElement, ElementType

        preprocessor = TextPreprocessor()

        # Create mock elements
        mock_elements = [
            ExtractedElement(
                type=ElementType.TEXT, bbox=[0, 0, 100, 20], content="First part"
            ),
            ExtractedElement(
                type=ElementType.TEXT, bbox=[0, 25, 100, 45], content="Second part"
            ),
            ExtractedElement(
                type=ElementType.TEXT, bbox=[0, 50, 100, 70], content="Third part"
            ),
        ]

        merged_text = preprocessor._merge_text_elements(mock_elements)

        assert isinstance(merged_text, str)
        assert "First part" in merged_text
        assert "Second part" in merged_text
        assert "Third part" in merged_text

    def test_chunk_creation(self):
        """Test chunk creation utility"""
        from src.app.services.text_preprocessor import TextPreprocessor

        preprocessor = TextPreprocessor()

        # Create mock elements
        mock_elements = [Mock(bbox=[0, 0, 100, 20]), Mock(bbox=[10, 10, 110, 30])]

        chunk = preprocessor._create_chunk(
            content="Test chunk content",
            chunk_type="text",
            page_num=1,
            elements=mock_elements,
        )

        assert chunk["content"] == "Test chunk content"
        assert chunk["type"] == "text"
        assert chunk["page_number"] == 1
        assert chunk["element_count"] == 2
        assert "bbox" in chunk

    def test_table_processing(self):
        """Test table processing utility"""
        from src.app.services.text_preprocessor import TextPreprocessor

        preprocessor = TextPreprocessor()

        # Create mock table elements
        mock_elements = [
            Mock(
                type="table",
                bbox=[0, 0, 200, 100],
                content=[["Header1", "Header2"], ["Data1", "Data2"]],
            )
        ]

        chunks = preprocessor._process_table_elements(mock_elements, 1)

        assert len(chunks) == 1
        assert chunks[0]["type"] == "table"
        assert chunks[0]["page_number"] == 1
        assert "table_content" in chunks[0]

    def test_image_processing(self):
        """Test image processing utility"""
        from src.app.services.text_preprocessor import TextPreprocessor

        preprocessor = TextPreprocessor()

        # Create mock image elements
        mock_elements = [
            Mock(
                type="image",
                bbox=[0, 0, 100, 100],
                content="image_path.png",
                file_path="image_path.png",
            )
        ]

        chunks = preprocessor._process_image_elements(mock_elements, 1)

        assert len(chunks) == 1
        assert chunks[0]["type"] == "image"
        assert chunks[0]["page_number"] == 1
        assert chunks[0]["image_path"] == "image_path.png"


if __name__ == "__main__":
    pytest.main([__file__])

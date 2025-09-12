"""
Tests for utility functions
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from src.app.utils import colpali_utils, qdrant_utils


class TestColPaliUtils:
    """Test ColPali utility functions"""
    
    @pytest.fixture
    def mock_model(self):
        """Mock ColPali model"""
        return Mock()
    
    @pytest.fixture
    def mock_processor(self):
        """Mock ColPali processor"""
        return Mock()
    
    def test_process_query_success(self, mock_model, mock_processor):
        """Test successful query processing"""
        # Mock processor output
        mock_processor.return_value = Mock(
            pixel_values=Mock(tolist=lambda: [[0.1] * 768])
        )
        
        # Mock model output
        mock_model.return_value = Mock(
            last_hidden_state=Mock(
                tolist=lambda: [[0.1] * 768]
            )
        )
        
        # Test processing
        result = colpali_utils.process_query(
            model=mock_model,
            processor=mock_processor,
            query="test query"
        )
        
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
    
    def test_process_query_empty_query(self, mock_model, mock_processor):
        """Test processing empty query"""
        with pytest.raises(ValueError):
            colpali_utils.process_query(
                model=mock_model,
                processor=mock_processor,
                query=""
            )
    
    def test_process_query_invalid_model(self):
        """Test processing with invalid model"""
        with pytest.raises(Exception):
            colpali_utils.process_query(
                model=None,
                processor=Mock(),
                query="test query"
            )
    
    def test_process_query_invalid_processor(self):
        """Test processing with invalid processor"""
        with pytest.raises(Exception):
            colpali_utils.process_query(
                model=Mock(),
                processor=None,
                query="test query"
            )


class TestQdrantUtils:
    """Test Qdrant utility functions"""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client"""
        return Mock()
    
    def test_create_payload_filter_session_id(self):
        """Test payload filter creation with session ID"""
        filter_dict = qdrant_utils.create_payload_filter(session_id="test-session")
        
        assert "must" in filter_dict
        assert len(filter_dict["must"]) == 1
        assert filter_dict["must"][0]["key"] == "session_id"
        assert filter_dict["must"][0]["match"] == {"value": "test-session"}
    
    def test_create_payload_filter_document(self):
        """Test payload filter creation with document name"""
        filter_dict = qdrant_utils.create_payload_filter(document="test.pdf")
        
        assert "must" in filter_dict
        assert len(filter_dict["must"]) == 1
        assert filter_dict["must"][0]["key"] == "document"
        assert filter_dict["must"][0]["match"] == {"value": "test.pdf"}
    
    def test_create_payload_filter_combined(self):
        """Test payload filter creation with multiple filters"""
        filter_dict = qdrant_utils.create_payload_filter(
            session_id="test-session",
            document="test.pdf"
        )
        
        assert "must" in filter_dict
        assert len(filter_dict["must"]) == 2
        
        # Check session_id filter
        session_filter = next(f for f in filter_dict["must"] if f["key"] == "session_id")
        assert session_filter["match"]["value"] == "test-session"
        
        # Check document filter
        doc_filter = next(f for f in filter_dict["must"] if f["key"] == "document")
        assert doc_filter["match"]["value"] == "test.pdf"
    
    def test_create_payload_filter_no_filters(self):
        """Test payload filter creation with no filters"""
        filter_dict = qdrant_utils.create_payload_filter()
        
        assert filter_dict == {"must": []}
    
    @pytest.mark.asyncio
    async def test_search_with_retry_success(self, mock_qdrant_client):
        """Test successful search with retry"""
        # Mock successful response
        mock_response = Mock()
        mock_response.points = [
            Mock(
                id="test-id",
                score=0.9,
                payload={"document": "test.pdf", "page": 1}
            )
        ]
        mock_qdrant_client.search.return_value = mock_response
        
        # Test search
        result = await qdrant_utils.search_with_retry(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            query_vector=[0.1] * 768,
            limit=10
        )
        
        assert result == mock_response
        mock_qdrant_client.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_with_retry_failure(self, mock_qdrant_client):
        """Test search with retry on failure"""
        # Mock initial failure, then success
        mock_qdrant_client.search.side_effect = [
            Exception("Search failed"),
            Mock(points=[])
        ]
        
        # Test search
        result = await qdrant_utils.search_with_retry(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            query_vector=[0.1] * 768,
            limit=10
        )
        
        assert result.points == []
        assert mock_qdrant_client.search.call_count == 2
    
    @pytest.mark.asyncio
    async def test_search_with_retry_max_attempts(self, mock_qdrant_client):
        """Test search with retry reaching max attempts"""
        # Mock persistent failure
        mock_qdrant_client.search.side_effect = Exception("Search failed")
        
        # Test search
        with pytest.raises(Exception):
            await qdrant_utils.search_with_retry(
                qdrant_client=mock_qdrant_client,
                collection_name="test_collection",
                query_vector=[0.1] * 768,
                limit=10,
                max_attempts=3
            )
        
        assert mock_qdrant_client.search.call_count == 3
    
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
        mock_qdrant_client.search.return_value = mock_response
        
        # Test search with threshold
        result = await qdrant_utils.search_with_retry(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            query_vector=[0.1] * 768,
            limit=10,
            score_threshold=0.6
        )
        
        # Should only return points with score >= threshold
        assert len(result.points) == 2
        assert result.points[0].score >= 0.6
        assert result.points[1].score >= 0.6


class TestImageProcessing:
    """Test image processing utilities"""
    
    def test_create_thumbnail(self):
        """Test thumbnail creation"""
        # Create test image
        image = Image.new('RGB', (1000, 1000), color='red')
        
        # Create thumbnail
        thumbnail = colpali_utils.create_thumbnail(image, size=224)
        
        assert thumbnail.size == (224, 224)
        assert thumbnail.mode == 'RGB'
    
    def test_create_thumbnail_square(self):
        """Test thumbnail creation with square aspect ratio"""
        # Create test image
        image = Image.new('RGB', (500, 300), color='blue')
        
        # Create thumbnail
        thumbnail = colpali_utils.create_thumbnail(image, size=224)
        
        assert thumbnail.size == (224, 224)
        assert thumbnail.mode == 'RGB'
    
    def test_image_to_base64(self):
        """Test image to base64 conversion"""
        # Create test image
        image = Image.new('RGB', (100, 100), color='green')
        
        # Convert to base64
        base64_str = colpali_utils.image_to_base64(image)
        
        assert isinstance(base64_str, str)
        assert base64_str.startswith("data:image/png;base64,")
        assert len(base64_str) > 0
    
    def test_base64_to_image(self):
        """Test base64 to image conversion"""
        # Create test image
        original_image = Image.new('RGB', (100, 100), color='yellow')
        
        # Convert to base64
        base64_str = colpali_utils.image_to_base64(original_image)
        
        # Convert back to image
        converted_image = colpali_utils.base64_to_image(base64_str)
        
        assert converted_image.size == original_image.size
        assert converted_image.mode == original_image.mode


if __name__ == "__main__":
    pytest.main([__file__])
"""
Tests for text preprocessor service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.app.services.text_preprocessor import TextPreprocessor
from src.app.models.schemas import (
    ExtractionResult,
    PageData,
    ExtractedElement,
    ElementType,
)


class TestTextPreprocessor:
    """Test text preprocessor service"""

    @pytest.fixture
    def text_preprocessor(self):
        """Create text preprocessor instance"""
        return TextPreprocessor(
            embedding_model="test-model",
            embedding_endpoint="http://localhost:11434/v1/",
            sparse_max_features=1000,
        )

    @pytest.fixture
    def mock_extraction_result(self):
        """Create mock extraction result"""
        return ExtractionResult(
            filename="test.pdf",
            total_pages=2,
            pages=[
                PageData(
                    page_number=1,
                    elements=[
                        ExtractedElement(
                            type=ElementType.TEXT,
                            bbox=[0, 0, 100, 20],
                            content="First paragraph of text content",
                        ),
                        ExtractedElement(
                            type=ElementType.TEXT,
                            bbox=[0, 25, 100, 45],
                            content="Second paragraph of text content",
                        ),
                        ExtractedElement(
                            type=ElementType.TABLE,
                            bbox=[0, 50, 200, 150],
                            content=[["Header1", "Header2"], ["Data1", "Data2"]],
                        ),
                        ExtractedElement(
                            type=ElementType.IMAGE,
                            bbox=[0, 160, 100, 260],
                            content="test_image.png",
                            file_path="test_image.png",
                        ),
                    ],
                ),
                PageData(
                    page_number=2,
                    elements=[
                        ExtractedElement(
                            type=ElementType.TEXT,
                            bbox=[0, 0, 100, 20],
                            content="Third paragraph of text content",
                        )
                    ],
                ),
            ],
            extraction_time=1.5,
        )

    def test_text_preprocessor_initialization(self, text_preprocessor):
        """Test text preprocessor initialization"""
        assert text_preprocessor.embedding_model == "test-model"
        assert text_preprocessor.embedding_endpoint == "http://localhost:11434/v1/"
        assert text_preprocessor.sparse_max_features == 1000
        assert text_preprocessor.logger is not None

    def test_create_chunks(self, text_preprocessor, mock_extraction_result):
        """Test chunk creation from extraction result"""
        chunks = text_preprocessor.create_chunks(mock_extraction_result)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check that all chunks have required fields
        for chunk in chunks:
            assert "content" in chunk
            assert "type" in chunk
            assert "page_number" in chunk
            assert "bbox" in chunk
            assert "element_count" in chunk
            assert "created_at" in chunk

    def test_process_text_elements(self, text_preprocessor):
        """Test text element processing"""
        mock_elements = [
            ExtractedElement(
                type=ElementType.TEXT,
                bbox=[0, 0, 100, 20],
                content="First line of text",
            ),
            ExtractedElement(
                type=ElementType.TEXT,
                bbox=[0, 25, 100, 45],
                content="Second line of text",
            ),
            ExtractedElement(
                type=ElementType.TEXT,
                bbox=[0, 50, 100, 70],
                content="Third line of text",
            ),
        ]

        chunks = text_preprocessor._process_text_elements(mock_elements, 1)

        assert isinstance(chunks, list)
        assert len(chunks) == 3

        # Check chunk properties
        for chunk in chunks:
            assert chunk["type"] == "text"
            assert chunk["page_number"] == 1
            assert "text_content" in chunk
            assert chunk["element_count"] == 1

    def test_process_text_elements_empty(self, text_preprocessor):
        """Test text element processing with empty elements"""
        chunks = text_preprocessor._process_text_elements([], 1)

        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_process_table_elements(self, text_preprocessor):
        """Test table element processing"""
        mock_elements = [
            ExtractedElement(
                type=ElementType.TABLE,
                bbox=[0, 0, 200, 100],
                content=[["Header1", "Header2"], ["Data1", "Data2"]],
            )
        ]

        chunks = text_preprocessor._process_table_elements(mock_elements, 1)

        assert isinstance(chunks, list)
        assert len(chunks) == 1

        chunk = chunks[0]
        assert chunk["type"] == "table"
        assert chunk["page_number"] == 1
        assert "table_content" in chunk
        assert chunk["element_count"] == 1

    def test_process_image_elements(self, text_preprocessor):
        """Test image element processing"""
        mock_elements = [
            ExtractedElement(
                type=ElementType.IMAGE,
                bbox=[0, 0, 100, 100],
                content="test_image.png",
                file_path="test_image.png",
            )
        ]

        chunks = text_preprocessor._process_image_elements(mock_elements, 1)

        assert isinstance(chunks, list)
        assert len(chunks) == 1

        chunk = chunks[0]
        assert chunk["type"] == "image"
        assert chunk["page_number"] == 1
        assert "image_path" in chunk
        assert chunk["element_count"] == 1

    def test_merge_text_elements(self, text_preprocessor):
        """Test text element merging"""
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

        merged_text = text_preprocessor._merge_text_elements(mock_elements)

        assert isinstance(merged_text, str)
        assert "First part" in merged_text
        assert "Second part" in merged_text
        assert "Third part" in merged_text

    def test_merge_text_elements_empty(self, text_preprocessor):
        """Test text element merging with empty elements"""
        merged_text = text_preprocessor._merge_text_elements([])

        assert isinstance(merged_text, str)
        assert merged_text == ""

    def test_calculate_average_bbox(self, text_preprocessor):
        """Test average bounding box calculation"""
        mock_elements = [
            ExtractedElement(bbox=[0, 0, 100, 20]),
            ExtractedElement(bbox=[10, 10, 110, 30]),
            ExtractedElement(bbox=[20, 20, 120, 40]),
        ]

        avg_bbox = text_preprocessor._calculate_average_bbox(mock_elements)

        assert len(avg_bbox) == 4
        assert avg_bbox[0] == 10.0  # Average x0
        assert avg_bbox[1] == 10.0  # Average y0
        assert avg_bbox[2] == 110.0  # Average x1
        assert avg_bbox[3] == 30.0  # Average y1

    def test_calculate_average_bbox_empty(self, text_preprocessor):
        """Test average bounding box calculation with empty elements"""
        avg_bbox = text_preprocessor._calculate_average_bbox([])

        assert len(avg_bbox) == 4
        assert avg_bbox == [0, 0, 0, 0]

    def test_create_chunk(self, text_preprocessor):
        """Test chunk creation"""
        mock_elements = [
            ExtractedElement(
                type=ElementType.TEXT, bbox=[0, 0, 100, 20], content="Test content"
            )
        ]

        chunk = text_preprocessor._create_chunk(
            content="Test chunk content",
            chunk_type="text",
            page_num=1,
            elements=mock_elements,
        )

        assert chunk["content"] == "Test chunk content"
        assert chunk["type"] == "text"
        assert chunk["page_number"] == 1
        assert chunk["element_count"] == 1
        assert "bbox" in chunk
        assert "text_content" in chunk

    def test_create_chunk_empty_elements(self, text_preprocessor):
        """Test chunk creation with empty elements"""
        chunk = text_preprocessor._create_chunk(
            content="Test chunk content", chunk_type="text", page_num=1, elements=[]
        )

        assert chunk == {}

    @pytest.mark.asyncio
    async def test_get_embedding_from_external_api_success(self, text_preprocessor):
        """Test successful embedding generation from external API"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            embedding = await text_preprocessor.get_embedding_from_external_api(
                "test text"
            )

            assert isinstance(embedding, list)
            assert len(embedding) == 4
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_get_embedding_from_external_api_error(self, text_preprocessor):
        """Test embedding generation with API error"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception):
                await text_preprocessor.get_embedding_from_external_api("test text")

    def test_generate_sparse_embedding(self, text_preprocessor):
        """Test sparse embedding generation"""
        text = "This is a test document with some important keywords"
        sparse_embedding = text_preprocessor.generate_sparse_embedding(text)

        assert isinstance(sparse_embedding, dict)
        assert len(sparse_embedding) > 0

        # Check that values are between 0 and 1
        for value in sparse_embedding.values():
            assert 0 <= value <= 1

    def test_generate_sparse_embedding_empty_text(self, text_preprocessor):
        """Test sparse embedding generation with empty text"""
        sparse_embedding = text_preprocessor.generate_sparse_embedding("")

        assert isinstance(sparse_embedding, dict)
        assert len(sparse_embedding) == 0

    @pytest.mark.asyncio
    async def test_get_dense_embedding_from_external_api_success(
        self, text_preprocessor
    ):
        """Test successful dense embedding generation from external API"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            embedding = await text_preprocessor.get_dense_embedding_from_external_api(
                "test text"
            )

            assert isinstance(embedding, list)
            assert len(embedding) == 5
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_get_dense_embedding_from_external_api_error(self, text_preprocessor):
        """Test dense embedding generation with API error"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception):
                await text_preprocessor.get_dense_embedding_from_external_api(
                    "test text"
                )


class TestTextPreprocessorIntegration:
    """Integration tests for text preprocessor"""

    @pytest.fixture
    def complex_extraction_result(self):
        """Create complex extraction result for integration testing"""
        return ExtractionResult(
            filename="complex_document.pdf",
            total_pages=3,
            pages=[
                PageData(
                    page_number=1,
                    elements=[
                        ExtractedElement(
                            type=ElementType.TEXT,
                            bbox=[0, 0, 200, 30],
                            content="Introduction paragraph with important concepts",
                        ),
                        ExtractedElement(
                            type=ElementType.TEXT,
                            bbox=[0, 35, 200, 65],
                            content="Second paragraph discussing methodology",
                        ),
                        ExtractedElement(
                            type=ElementType.TABLE,
                            bbox=[0, 70, 300, 170],
                            content=[["Method", "Result"], ["A", "1"], ["B", "2"]],
                        ),
                        ExtractedElement(
                            type=ElementType.IMAGE,
                            bbox=[0, 180, 150, 330],
                            content="figure1.png",
                            file_path="figure1.png",
                        ),
                    ],
                ),
                PageData(
                    page_number=2,
                    elements=[
                        ExtractedElement(
                            type=ElementType.TEXT,
                            bbox=[0, 0, 200, 25],
                            content="Conclusion and future work",
                        )
                    ],
                ),
                PageData(
                    page_number=3,
                    elements=[
                        ExtractedElement(
                            type=ElementType.TEXT,
                            bbox=[0, 0, 200, 30],
                            content="References and bibliography",
                        )
                    ],
                ),
            ],
            extraction_time=2.5,
        )

    def test_full_chunking_workflow(self, text_preprocessor, complex_extraction_result):
        """Test complete chunking workflow"""
        chunks = text_preprocessor.create_chunks(complex_extraction_result)

        # Verify chunk creation
        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check chunk types
        chunk_types = [chunk["type"] for chunk in chunks]
        assert "text" in chunk_types
        assert "table" in chunk_types
        assert "image" in chunk_types

        # Check that all chunks have required fields
        for chunk in chunks:
            assert "content" in chunk
            assert "type" in chunk
            assert "page_number" in chunk
            assert "bbox" in chunk
            assert "element_count" in chunk
            assert "created_at" in chunk

    def test_chunk_grouping_logic(self, text_preprocessor):
        """Test text chunk grouping logic"""
        # Create elements that should be grouped together
        mock_elements = [
            ExtractedElement(
                type=ElementType.TEXT, bbox=[0, 0, 100, 20], content="First line"
            ),
            ExtractedElement(
                type=ElementType.TEXT, bbox=[0, 15, 100, 35], content="Second line"
            ),
            ExtractedElement(
                type=ElementType.TEXT,
                bbox=[0, 50, 100, 70],
                content="Third line (separate paragraph)",
            ),
        ]

        chunks = text_preprocessor._process_text_elements(mock_elements, 1)

        # Should create 2 chunks: one for the first two lines, one for the third
        assert len(chunks) == 2

        # Check that the first chunk contains both lines
        first_chunk_content = chunks[0]["text_content"]
        assert "First line" in first_chunk_content
        assert "Second line" in first_chunk_content

        # Check that the second chunk contains the third line
        second_chunk_content = chunks[1]["text_content"]
        assert "Third line (separate paragraph)" in second_chunk_content

    @pytest.mark.asyncio
    async def test_embedding_generation_workflow(self, text_preprocessor):
        """Test complete embedding generation workflow"""
        with patch("aiohttp.ClientSession") as mock_session:
            # Mock successful API responses
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            # Test both embedding types
            dense_embedding = (
                await text_preprocessor.get_dense_embedding_from_external_api(
                    "test text"
                )
            )
            sparse_embedding = text_preprocessor.generate_sparse_embedding("test text")

            assert isinstance(dense_embedding, list)
            assert len(dense_embedding) == 5
            assert all(isinstance(x, float) for x in dense_embedding)

            assert isinstance(sparse_embedding, dict)
            assert len(sparse_embedding) > 0


if __name__ == "__main__":
    pytest.main([__file__])

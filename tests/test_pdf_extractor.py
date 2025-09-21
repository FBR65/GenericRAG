"""
Tests for PDF extractor service
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import io

from src.app.services.pdf_extractor import PDFExtractor
from src.app.models.schemas import (
    ExtractionResult,
    PageData,
    ExtractedElement,
    ElementType,
)


class TestPDFExtractor:
    """Test PDF extractor service"""

    @pytest.fixture
    def pdf_extractor(self):
        """Create PDF extractor instance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            return PDFExtractor(output_dir=temp_dir)

    @pytest.fixture
    def mock_pdf_content(self):
        """Create mock PDF content"""
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"

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
                            content="Test text content",
                        )
                    ],
                ),
                PageData(
                    page_number=2,
                    elements=[
                        ExtractedElement(
                            type=ElementType.TABLE,
                            bbox=[0, 0, 200, 100],
                            content=[["Header1", "Header2"], ["Data1", "Data2"]],
                        ),
                        ExtractedElement(
                            type=ElementType.IMAGE,
                            bbox=[0, 0, 100, 100],
                            content="test_image.png",
                            file_path="test_image.png",
                        ),
                    ],
                ),
            ],
            extraction_time=1.5,
        )

    def test_pdf_extractor_initialization(self, pdf_extractor):
        """Test PDF extractor initialization"""
        assert pdf_extractor.output_dir is not None
        assert pdf_extractor.image_dir is not None
        assert pdf_extractor.logger is not None
        assert pdf_extractor.settings is not None

    def test_extract_pdf_data_success(self, pdf_extractor, mock_pdf_content):
        """Test successful PDF data extraction"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(mock_pdf_content)
            temp_file_path = temp_file.name

        try:
            with (
                patch("pdfplumber.open") as mock_plumber,
                patch("fitz.open") as mock_fitz,
            ):
                # Mock pdfplumber document
                mock_plumber_page = Mock()
                mock_plumber_page.extract_tables.return_value = []
                mock_plumber_page.find_tables.return_value = []
                mock_plumber_page.get_text.return_value = []
                mock_plumber_doc = Mock()
                mock_plumber_doc.pages = [mock_plumber_page, mock_plumber_page]
                mock_plumber.return_value.__enter__.return_value = mock_plumber_doc

                # Mock fitz document
                mock_fitz_page = Mock()
                mock_fitz_page.get_text.return_value = []
                mock_fitz_page.get_images.return_value = []
                mock_fitz_doc = Mock()
                mock_fitz_doc.load_page.return_value = mock_fitz_page
                mock_fitz.return_value.__enter__.return_value = mock_fitz_doc

                # Test extraction
                result = pdf_extractor.extract_pdf_data(temp_file_path)

                assert result is not None
                assert result["filename"] == "test.pdf"
                assert result["total_pages"] == 2
                assert "pages" in result
                assert "extraction_time" in result

        finally:
            os.unlink(temp_file_path)

    def test_extract_pdf_data_file_not_found(self, pdf_extractor):
        """Test PDF extraction with non-existent file"""
        with pytest.raises(Exception):
            pdf_extractor.extract_pdf_data("non_existent_file.pdf")

    def test_extract_page_data(self, pdf_extractor):
        """Test page data extraction"""
        mock_plumber_page = Mock()
        mock_plumber_page.extract_tables.return_value = []
        mock_plumber_page.find_tables.return_value = []

        mock_fitz_page = Mock()
        mock_fitz_page.get_text.return_value = []
        mock_fitz_page.get_images.return_value = []

        page_data = pdf_extractor._extract_page_data(
            mock_plumber_page, mock_fitz_page, 1
        )

        assert page_data is not None
        assert page_data["page_number"] == 2  # 1 + 1
        assert "elements" in page_data

    def test_extract_text_elements(self, pdf_extractor):
        """Test text element extraction"""
        mock_fitz_page = Mock()
        mock_fitz_page.get_text.return_value = [
            [0, 0, 100, 20, "Test text 1"],
            [0, 25, 100, 45, "Test text 2"],
        ]

        elements = pdf_extractor._extract_text_elements(mock_fitz_page)

        assert len(elements) == 2
        assert all(element["type"] == "text" for element in elements)
        assert all("bbox" in element for element in elements)
        assert all("content" in element for element in elements)

    def test_extract_table_elements(self, pdf_extractor):
        """Test table element extraction"""
        mock_plumber_page = Mock()
        mock_plumber_page.extract_tables.return_value = [
            [["Header1", "Header2"], ["Data1", "Data2"]]
        ]

        mock_table = Mock()
        mock_table.bbox = [0, 0, 200, 100]
        mock_plumber_page.find_tables.return_value = [mock_table]

        elements = pdf_extractor._extract_table_elements(mock_plumber_page)

        assert len(elements) == 1
        assert elements[0]["type"] == "table"
        assert "bbox" in elements[0]
        assert "content" in elements[0]

    def test_extract_image_elements(self, pdf_extractor):
        """Test image element extraction"""
        mock_fitz_page = Mock()
        mock_fitz_page.get_images.return_value = [(1,)]

        mock_image_info = Mock()
        mock_image_info.bbox = [0, 0, 100, 100]
        mock_fitz_page.get_image_info.return_value = [mock_image_info]

        with patch("fitz.Pixmap") as mock_pixmap:
            mock_pix = Mock()
            mock_pix.n = 3
            mock_pix.alpha = 0
            mock_pix.width = 100
            mock_pix.height = 100
            mock_pix.save = Mock()
            mock_pixmap.return_value = mock_pix

            with patch.object(Path, "mkdir"):
                elements = pdf_extractor._extract_image_elements(mock_fitz_page, 1)

                assert len(elements) == 1
                assert elements[0]["type"] == "image"
                assert "bbox" in elements[0]
                assert "content" in elements[0]
                assert "file_path" in elements[0]
                assert "filename" in elements[0]
                assert "page_number" in elements[0]
                assert "requires_embedding" in elements[0]
                assert "image_metadata" in elements[0]

    def test_extract_to_pydantic(self, pdf_extractor, mock_extraction_result):
        """Test extraction to Pydantic model"""
        with patch.object(pdf_extractor, "extract_pdf_data") as mock_extract:
            mock_extract.return_value = {
                "filename": "test.pdf",
                "total_pages": 2,
                "pages": [
                    {
                        "page_number": 1,
                        "elements": [
                            {
                                "type": "text",
                                "bbox": [0, 0, 100, 20],
                                "content": "Test text",
                            }
                        ],
                    }
                ],
                "extraction_time": 1.0,
            }

            result = pdf_extractor.extract_to_pydantic("test.pdf")

            assert isinstance(result, ExtractionResult)
            assert result.filename == "test.pdf"
            assert result.total_pages == 2
            assert len(result.pages) == 1
            assert result.pages[0].page_number == 1
            assert len(result.pages[0].elements) == 1

    def test_get_images_for_embedding(self, pdf_extractor):
        """Test getting images for embedding"""
        with patch.object(pdf_extractor, "extract_pdf_data") as mock_extract:
            mock_extract.return_value = {
                "filename": "test.pdf",
                "total_pages": 1,
                "pages": [
                    {
                        "page_number": 1,
                        "elements": [
                            {
                                "type": "image",
                                "bbox": [0, 0, 100, 100],
                                "content": "test_image.png",
                                "file_path": "test_image.png",
                                "filename": "test_image.png",
                                "requires_embedding": True,
                                "image_metadata": {"width": 100, "height": 100},
                            }
                        ],
                    }
                ],
                "extraction_time": 1.0,
            }

            images = pdf_extractor.get_images_for_embedding("test.pdf")

            assert len(images) == 1
            assert images[0]["file_path"] == "test_image.png"
            assert images[0]["filename"] == "test_image.png"
            assert images[0]["page_number"] == 1
            assert images[0]["requires_embedding"] is True
            assert "metadata" in images[0]
            assert "document_id" in images[0]

    def test_extract_image_elements_error_handling(self, pdf_extractor):
        """Test image element extraction error handling"""
        mock_fitz_page = Mock()
        mock_fitz_page.get_images.return_value = [(1,)]

        mock_image_info = Mock()
        mock_image_info.bbox = [0, 0, 100, 100]
        mock_fitz_page.get_image_info.return_value = [mock_image_info]

        with patch("fitz.Pixmap") as mock_pixmap:
            mock_pixmap.side_effect = Exception("Image extraction failed")

            elements = pdf_extractor._extract_image_elements(mock_fitz_page, 1)

            # Should handle errors gracefully and return empty list
            assert isinstance(elements, list)

    def test_extract_table_elements_empty_tables(self, pdf_extractor):
        """Test table element extraction with empty tables"""
        mock_plumber_page = Mock()
        mock_plumber_page.extract_tables.return_value = []
        mock_plumber_page.find_tables.return_value = []

        elements = pdf_extractor._extract_table_elements(mock_plumber_page)

        assert len(elements) == 0

    def test_extract_text_elements_empty_text(self, pdf_extractor):
        """Test text element extraction with empty text"""
        mock_fitz_page = Mock()
        mock_fitz_page.get_text.return_value = []

        elements = pdf_extractor._extract_text_elements(mock_fitz_page)

        assert len(elements) == 0


class TestPDFExtractorIntegration:
    """Integration tests for PDF extractor"""

    def test_full_pdf_extraction_workflow(self):
        """Test complete PDF extraction workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = PDFExtractor(output_dir=temp_dir)

            # Create a simple PDF file for testing
            pdf_content = (
                b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            )

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name

            try:
                with (
                    patch("pdfplumber.open") as mock_plumber,
                    patch("fitz.open") as mock_fitz,
                ):
                    # Setup mocks
                    mock_plumber_page = Mock()
                    mock_plumber_page.extract_tables.return_value = []
                    mock_plumber_page.find_tables.return_value = []
                    mock_plumber_page.get_text.return_value = []
                    mock_plumber_doc = Mock()
                    mock_plumber_doc.pages = [mock_plumber_page]
                    mock_plumber.return_value.__enter__.return_value = mock_plumber_doc

                    mock_fitz_page = Mock()
                    mock_fitz_page.get_text.return_value = []
                    mock_fitz_page.get_images.return_value = []
                    mock_fitz_doc = Mock()
                    mock_fitz_doc.load_page.return_value = mock_fitz_page
                    mock_fitz.return_value.__enter__.return_value = mock_fitz_doc

                    # Test extraction
                    raw_data = extractor.extract_pdf_data(temp_file_path)
                    pydantic_result = extractor.extract_to_pydantic(temp_file_path)
                    images_for_embedding = extractor.get_images_for_embedding(
                        temp_file_path
                    )

                    # Verify results
                    assert raw_data["filename"] == "test.pdf"
                    assert raw_data["total_pages"] == 1
                    assert isinstance(pydantic_result, ExtractionResult)
                    assert pydantic_result.filename == "test.pdf"
                    assert len(images_for_embedding) == 0

            finally:
                os.unlink(temp_file_path)


if __name__ == "__main__":
    pytest.main([__file__])

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

from src.extractor import PDFExtractor
from src.models import ExtractionResult, ElementType


class TestPDFExtractor:
    @pytest.fixture
    def temp_dir(self):
        """Temporäres Verzeichnis für Tests"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        import shutil

        shutil.rmtree(temp_path, onerror=None)

    @pytest.fixture
    def extractor(self, temp_dir):
        """PDFExtractor-Instanz für Tests"""
        return PDFExtractor(temp_dir)

    @pytest.fixture
    def mock_pdf_content(self):
        """Mock PDF-Inhalt für Tests"""
        return "%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"

    @patch("src.extractor.pdfplumber.open")
    @patch("src.extractor.fitz.open")
    def test_extract_pdf_data_success(
        self, mock_fitz_open, mock_plumber_open, extractor
    ):
        """Test erfolgreiche PDF-Extraktion mit vollständigen Mocks"""
        import logging
        import time

        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        logger.debug("Starte PDF Extraktion Test")

        start_time = time.time()

        # Mock pdfplumber Dokument
        mock_plumber_page = Mock()
        mock_plumber_page.extract_tables.return_value = [
            [["Name", "Alter"], ["Max", "30"]]
        ]
        mock_plumber_page.find_tables.return_value = [Mock(bbox=(100, 200, 400, 300))]

        mock_plumber_doc = Mock()
        mock_plumber_doc.pages = [mock_plumber_page]
        mock_plumber_open.return_value.__enter__.return_value = mock_plumber_doc

        # Mock PyMuPDF Dokument
        mock_fitz_page = Mock()
        mock_fitz_page.get_text.return_value = [(100, 200, 400, 250, "Test Text", None)]
        mock_fitz_page.get_images.return_value = [(1,)]
        mock_fitz_page.get_image_info.return_value = [{"bbox": (50, 50, 150, 100)}]

        mock_fitz_doc = Mock()
        mock_fitz_doc.__len__ = Mock(return_value=1)
        mock_fitz_doc.load_page.return_value = mock_fitz_page
        mock_fitz_open.return_value.__enter__.return_value = mock_fitz_doc

        # Mock Pixmap für Bildspeicherung
        with patch("src.extractor.fitz.Pixmap") as mock_pixmap:
            logger.debug("Mocke Pixmap")
            mock_pix = Mock()
            mock_pix.n = 3  # RGB
            mock_pix.alpha = 0
            mock_pix.save = Mock()
            mock_pixmap.return_value = mock_pix

            logger.debug("Führe Extraktion durch")
            # Führe Extraktion durch
            result = extractor.extract_pdf_data("test.pdf")

            elapsed = time.time() - start_time
            logger.debug(f"Extraktion completed in {elapsed:.4f}s")

            # Überprüfe Ergebnis
            assert result["filename"] == "test.pdf"
            assert result["total_pages"] == 1
            assert len(result["pages"]) == 1
            assert result["extraction_time"] >= 0

            # Überprüfe Seite
            page = result["pages"][0]
            assert page["page_number"] == 1
            assert len(page["elements"]) == 3  # Text, Tabelle, Bild

            # Überprüfe Textelement
            text_element = next(e for e in page["elements"] if e["type"] == "text")
            assert text_element["bbox"] == [100, 200, 400, 250]
            assert text_element["content"] == "Test Text"

            # Überprüfe Tabellenelement
            table_element = next(e for e in page["elements"] if e["type"] == "table")
            assert table_element["bbox"] == [100, 200, 400, 300]
            assert table_element["content"] == [["Name", "Alter"], ["Max", "30"]]

            # Überprüfe Bildelement
            image_element = next(e for e in page["elements"] if e["type"] == "image")
            assert list(image_element["bbox"]) == [50, 50, 150, 100]
            assert "img1.png" in image_element["content"]

            # Überprüfe dass Bild gespeichert wurde
            mock_pix.save.assert_called_once()

            elapsed_total = time.time() - start_time
            logger.debug(f"Test completed in {elapsed_total:.4f}s")

    @patch("src.extractor.pdfplumber.open")
    def test_extract_pdf_data_invalid_file(self, mock_plumber_open, extractor):
        """Test mit ungültiger PDF-Datei"""
        mock_plumber_open.side_effect = Exception("Invalid PDF")

        with pytest.raises(Exception) as exc_info:
            extractor.extract_pdf_data("invalid.pdf")

        assert "Fehler bei der PDF-Extraktion" in str(
            exc_info.value
        ) or "Invalid PDF" in str(exc_info.value)

    @patch("src.extractor.pdfplumber.open")
    @patch("src.extractor.fitz.open")
    def test_extract_text_elements(self, mock_fitz_open, mock_plumber_open, extractor):
        """Test Text-Extraktion"""
        mock_fitz_page = Mock()
        mock_fitz_page.get_text.return_value = [
            (100, 200, 400, 250, "Text 1", None),
            (150, 300, 450, 350, "Text 2", None),
            (200, 400, 500, 450, "", None),  # Leerer Text
        ]

        mock_fitz_doc = Mock()
        mock_fitz_doc.load_page.return_value = mock_fitz_page
        mock_fitz_open.return_value.__enter__.return_value = mock_fitz_doc

        elements = extractor._extract_text_elements(mock_fitz_page)

        assert len(elements) == 2  # Nur nicht-leere Texte
        assert elements[0]["type"] == "text"
        assert elements[0]["content"] == "Text 1"
        assert elements[1]["content"] == "Text 2"

    @patch("src.extractor.pdfplumber.open")
    def test_extract_table_elements(self, mock_plumber_open, extractor):
        """Test Tabellen-Extraktion"""
        mock_plumber_page = Mock()
        mock_plumber_page.extract_tables.return_value = [
            [["Name", "Alter"], ["Max", "30"]],
            [["A", "B"], ["C", "D"]],  # Leere Zeile
            [],  # Leere Tabelle
        ]
        mock_plumber_page.find_tables.return_value = [
            Mock(bbox=(100, 200, 400, 300)),
            Mock(bbox=(150, 250, 450, 350)),
        ]

        elements = extractor._extract_table_elements(mock_plumber_page)

        assert len(elements) == 2  # Nur nicht-leere Tabellen
        assert elements[0]["type"] == "table"
        assert elements[0]["content"] == [["Name", "Alter"], ["Max", "30"]]
        assert elements[1]["content"] == [["A", "B"], ["C", "D"]]

    @patch("src.extractor.pdfplumber.open")
    @patch("src.extractor.fitz.open")
    def test_extract_image_elements(self, mock_fitz_open, mock_plumber_open, extractor):
        """Test Bild-Extraktion"""
        mock_fitz_page = Mock()
        mock_fitz_page.get_images.return_value = [(1,), (2,)]
        mock_fitz_page.get_image_info.return_value = [
            {"bbox": (50, 50, 150, 100)},
            {"bbox": (200, 200, 300, 250)},
        ]

        mock_fitz_doc = Mock()
        mock_fitz_doc.load_page.return_value = mock_fitz_page
        mock_fitz_open.return_value.__enter__.return_value = mock_fitz_doc

        # Mock Pixmap
        with patch("src.extractor.fitz.Pixmap") as mock_pixmap:
            mock_pix = Mock()
            mock_pix.n = 3  # RGB
            mock_pix.alpha = 0
            mock_pix.save = Mock()
            mock_pixmap.return_value = mock_pix

            elements = extractor._extract_image_elements(mock_fitz_page, 0)

            assert len(elements) == 2
            assert elements[0]["type"] == "image"
            assert list(elements[0]["bbox"]) == [50, 50, 150, 100]
            assert list(elements[1]["bbox"]) == [200, 200, 300, 250]

            # Überprüfe, dass Bilder gespeichert wurden
            mock_pix.save.assert_called()

    @patch("src.extractor.pdfplumber.open")
    @patch("src.extractor.fitz.open")
    def test_extract_to_pydantic(self, mock_fitz_open, mock_plumber_open, extractor):
        """Test Konvertierung zu Pydantic-Modell"""
        import logging
        import time

        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        logger.debug("Starte Pydantic Konvertierung Test")

        start_time = time.time()

        # Mock Daten
        mock_plumber_page = Mock()
        mock_plumber_page.extract_tables.return_value = [
            [["Name", "Alter"], ["Max", "30"]]
        ]
        mock_plumber_page.find_tables.return_value = [Mock(bbox=(100, 200, 400, 300))]

        mock_plumber_doc = Mock()
        mock_plumber_doc.pages = [mock_plumber_page]
        mock_plumber_open.return_value.__enter__.return_value = mock_plumber_doc

        mock_fitz_page = Mock()
        mock_fitz_page.get_text.return_value = [(100, 200, 400, 250, "Test Text", None)]
        mock_fitz_page.get_images.return_value = [(1,)]
        mock_fitz_page.get_image_info.return_value = [{"bbox": (50, 50, 150, 100)}]

        mock_fitz_doc = Mock()
        mock_fitz_doc.__len__ = Mock(return_value=1)
        mock_fitz_doc.load_page.return_value = mock_fitz_page
        mock_fitz_open.return_value.__enter__.return_value = mock_fitz_doc

        # Mock Pixmap für Bildspeicherung
        with patch("src.extractor.fitz.Pixmap") as mock_pixmap:
            mock_pix = Mock()
            mock_pix.n = 3
            mock_pix.alpha = 0
            mock_pix.save = Mock()
            mock_pixmap.return_value = mock_pix

            logger.debug("Führe Extraktion durch")
            # Führe Extraktion durch
            result = extractor.extract_pdf_data("test.pdf")

            elapsed = time.time() - start_time
            logger.debug(f"Extraktion completed in {elapsed:.4f}s")

            logger.debug("Konvertiere zu Pydantic-Modell")
            # Konvertiere zu Pydantic-Modell
            extraction_result = ExtractionResult(**result)

            # Überprüfe Typ
            assert isinstance(extraction_result, ExtractionResult)
            assert extraction_result.filename == "test.pdf"
            assert extraction_result.total_pages == 1
            assert len(extraction_result.pages) == 1

            # Überprüfe Seiten
            page = extraction_result.pages[0]
            assert page.page_number == 1
            assert len(page.elements) == 3  # Text, Tabelle, Bild

            # Überprüfe Elemente
            text_element = next(e for e in page.elements if e.type == ElementType.TEXT)
            assert text_element.bbox == [100, 200, 400, 250]
            assert text_element.content == "Test Text"

            table_element = next(
                e for e in page.elements if e.type == ElementType.TABLE
            )
            assert table_element.bbox == [100, 200, 400, 300]
            assert table_element.content == [["Name", "Alter"], ["Max", "30"]]

            image_element = next(
                e for e in page.elements if e.type == ElementType.IMAGE
            )
            assert image_element.bbox == [50, 50, 150, 100]

            elapsed_total = time.time() - start_time
            logger.debug(f"Test completed in {elapsed_total:.4f}s")

    def test_output_directory_creation(self, temp_dir):
        """Test Erstellung des Ausgabeverzeichnisses"""
        import logging
        import time

        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        logger.debug("Starte Output Directory Creation Test")

        start_time = time.time()

        new_extractor = PDFExtractor(temp_dir)
        assert Path(temp_dir).exists()

        elapsed_total = time.time() - start_time
        logger.debug(f"Test completed in {elapsed_total:.4f}s")

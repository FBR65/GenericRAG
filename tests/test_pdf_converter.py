import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from src.utils.pdf_converter import PDFConverter


class TestPDFConverter:
    """Test cases for PDF converter utility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = PDFConverter(dpi=300, quality=95)
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test PDFConverter initialization."""
        converter = PDFConverter(dpi=200, quality=90)
        assert converter.dpi == 200
        assert converter.quality == 90
    
    def test_validate_pdf_valid(self):
        """Test PDF validation with valid PDF."""
        # Create a temporary PDF file
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        
        # Mock fitz.open to simulate a valid PDF
        with patch('src.utils.pdf_converter.fitz.open') as mock_open:
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 5  # 5 pages
            mock_open.return_value = mock_doc
            
            result = self.converter.validate_pdf(pdf_path)
            assert result is True
            mock_open.assert_called_once_with(pdf_path)
    
    def test_validate_pdf_invalid(self):
        """Test PDF validation with invalid PDF."""
        # Test non-existent file
        result = self.converter.validate_pdf("/non/existent/file.pdf")
        assert result is False
        
        # Test non-PDF file
        txt_path = os.path.join(self.temp_dir, "test.txt")
        with open(txt_path, 'w') as f:
            f.write("This is not a PDF")
        
        result = self.converter.validate_pdf(txt_path)
        assert result is False
    
    def test_validate_pdf_corrupted(self):
        """Test PDF validation with corrupted PDF."""
        pdf_path = os.path.join(self.temp_dir, "corrupted.pdf")
        
        # Mock fitz.open to raise an exception
        with patch('src.utils.pdf_converter.fitz.open') as mock_open:
            mock_open.side_effect = Exception("Corrupted PDF")
            
            result = self.converter.validate_pdf(pdf_path)
            assert result is False
    
    def test_get_pdf_info(self):
        """Test getting PDF information."""
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        
        # Mock fitz.open and its metadata
        with patch('src.utils.pdf_converter.fitz.open') as mock_open:
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 3
            mock_doc.metadata = {
                "title": "Test Document",
                "author": "Test Author",
                "subject": "Test Subject",
                "creator": "Test Creator",
                "creationDate": "2024-01-01T10:00:00",
                "modDate": "2024-01-01T12:00:00"
            }
            mock_open.return_value = mock_doc
            
            # Mock file size
            with patch('os.path.getsize', return_value=1024):
                info = self.converter.get_pdf_info(pdf_path)
                
                assert info["filename"] == "test.pdf"
                assert info["file_size"] == 1024
                assert info["page_count"] == 3
                assert info["title"] == "Test Document"
                assert info["author"] == "Test Author"
                assert info["subject"] == "Test Subject"
                assert info["creator"] == "Test Creator"
                assert info["creation_date"] == "2024-01-01T10:00:00"
                assert info["modification_date"] == "2024-01-01T12:00:00"
    
    def test_get_pdf_info_error(self):
        """Test PDF info retrieval with error."""
        pdf_path = os.path.join(self.temp_dir, "error.pdf")
        
        # Mock fitz.open to raise an exception
        with patch('src.utils.pdf_converter.fitz.open') as mock_open:
            mock_open.side_effect = Exception("PDF read error")
            
            info = self.converter.get_pdf_info(pdf_path)
            assert info == {}
    
    def test_save_images(self):
        """Test saving images to disk."""
        # Create mock images
        images = [
            (1, Image.new('RGB', (100, 100), color='red')),
            (2, Image.new('RGB', (200, 200), color='blue')),
            (3, Image.new('RGB', (300, 300), color='green'))
        ]
        
        output_dir = os.path.join(self.temp_dir, "output")
        saved_paths = self.converter.save_images(images, output_dir)
        
        # Check that images were saved
        assert len(saved_paths) == 3
        assert os.path.exists(saved_paths[0])
        assert os.path.exists(saved_paths[1])
        assert os.path.exists(saved_paths[2])
        
        # Check filenames
        assert "page_0001.png" in saved_paths[0]
        assert "page_0002.png" in saved_paths[1]
        assert "page_0003.png" in saved_paths[2]
    
    def test_save_images_empty_list(self):
        """Test saving empty image list."""
        output_dir = os.path.join(self.temp_dir, "output")
        saved_paths = self.converter.save_images([], output_dir)
        
        assert saved_paths == []
        assert os.path.exists(output_dir)  # Directory should still be created
    
    @patch('src.utils.pdf_converter.fitz.open')
    def test_pdf_to_images_success(self, mock_open):
        """Test successful PDF to image conversion."""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 2
        
        # Mock pages
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        
        # Mock pixmap
        mock_pix1 = MagicMock()
        mock_pix1.tobytes.return_value = b"fake_image_data_1"
        
        mock_pix2 = MagicMock()
        mock_pix2.tobytes.return_value = b"fake_image_data_2"
        
        mock_page1.get_pixmap.return_value = mock_pix1
        mock_page2.get_pixmap.return_value = mock_pix2
        
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]
        mock_open.return_value = mock_doc
        
        # Mock PIL Image
        with patch('src.utils.pdf_converter.Image.open') as mock_pil_open:
            mock_img1 = MagicMock()
            mock_img2 = MagicMock()
            mock_pil_open.side_effect = [mock_img1, mock_img2]
            
            pdf_path = os.path.join(self.temp_dir, "test.pdf")
            result = self.converter.pdf_to_images(pdf_path)
            
            assert len(result) == 2
            assert result[0] == (1, mock_img1)
            assert result[1] == (2, mock_img2)
            
            # Verify calls
            mock_open.assert_called_once_with(pdf_path)
            assert mock_doc.load_page.call_count == 2
            mock_pil_open.assert_called()
    
    @patch('src.utils.pdf_converter.fitz.open')
    def test_pdf_to_images_error(self, mock_open):
        """Test PDF to image conversion with error."""
        mock_open.side_effect = Exception("PDF read error")
        
        pdf_path = os.path.join(self.temp_dir, "error.pdf")
        
        with pytest.raises(Exception) as exc_info:
            self.converter.pdf_to_images(pdf_path)
        
        assert "PDF read error" in str(exc_info.value)
    
    @patch('src.utils.pdf_converter.fitz.open')
    def test_pdf_to_images_empty_pdf(self, mock_open):
        """Test PDF to image conversion with empty PDF."""
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 0
        mock_open.return_value = mock_doc
        
        pdf_path = os.path.join(self.temp_dir, "empty.pdf")
        result = self.converter.pdf_to_images(pdf_path)
        
        assert result == []
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create test images
        rgb_image = Image.new('RGB', (200, 200), color='red')
        cmyk_image = Image.new('CMYK', (200, 200), color='blue')
        grayscale_image = Image.new('L', (200, 200), color='gray')
        
        # Test RGB image (should remain unchanged)
        result1 = self.converter._preprocess_image(rgb_image)
        assert result1.mode == 'RGB'
        assert result1.size == (200, 200)
        
        # Test CMYK image (should be converted to RGB)
        result2 = self.converter._preprocess_image(cmyk_image)
        assert result2.mode == 'RGB'
        
        # Test grayscale image (should be converted to RGB)
        result3 = self.converter._preprocess_image(grayscale_image)
        assert result3.mode == 'RGB'
    
    def test_preprocess_image_resize(self):
        """Test image resizing when dimensions exceed maximum."""
        # Create large image
        large_image = Image.new('RGB', (1500, 1500), color='green')
        
        with patch('src.utils.pdf_converter.Image') as mock_pil:
            mock_image_instance = MagicMock()
            mock_image_instance.thumbnail = MagicMock()
            mock_pil.new.return_value = mock_image_instance
            
            result = self.converter._preprocess_image(large_image)
            
            # Verify thumbnail was called
            mock_image_instance.thumbnail.assert_called_once_with((1024, 1024), mock_pil.LANCZOS)
    
    def test_preprocess_image_small(self):
        """Test that small images are not resized."""
        # Create small image
        small_image = Image.new('RGB', (500, 500), color='blue')
        
        with patch('src.utils.pdf_converter.Image') as mock_pil:
            mock_image_instance = MagicMock()
            mock_image_instance.thumbnail = MagicMock()
            mock_pil.new.return_value = mock_image_instance
            
            result = self.converter._preprocess_image(small_image)
            
            # Verify thumbnail was not called
            mock_image_instance.thumbnail.assert_not_called()
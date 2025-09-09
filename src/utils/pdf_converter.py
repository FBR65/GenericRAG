import os
import fitz  # PyMuPDF
from PIL import Image
import io
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFConverter:
    def __init__(self, dpi: int = 300, quality: int = 95):
        self.dpi = dpi
        self.quality = quality
    
    def pdf_to_images(self, pdf_path: str) -> List[Tuple[int, Image.Image]]:
        """
        Convert PDF pages to PIL Images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of tuples (page_number, image)
        """
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                images.append((page_num + 1, img))  # 1-based page numbering
            
            doc.close()
            logger.info(f"Converted {len(images)} pages from {pdf_path}")
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {str(e)}")
            raise
    
    def save_images(self, images: List[Tuple[int, Image.Image]], output_dir: str) -> List[str]:
        """
        Save images to disk and return file paths.
        
        Args:
            images: List of (page_number, image) tuples
            output_dir: Directory to save images
            
        Returns:
            List of image file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        
        for page_num, img in images:
            filename = f"page_{page_num:04d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, "PNG", quality=self.quality)
            image_paths.append(filepath)
        
        logger.info(f"Saved {len(image_paths)} images to {output_dir}")
        return image_paths
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate if the file is a valid PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            if not os.path.exists(pdf_path):
                return False
            
            if not pdf_path.lower().endswith('.pdf'):
                return False
            
            doc = fitz.open(pdf_path)
            is_valid = len(doc) > 0
            doc.close()
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating PDF {pdf_path}: {str(e)}")
            return False
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get basic information about the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            doc = fitz.open(pdf_path)
            info = {
                "filename": os.path.basename(pdf_path),
                "file_size": os.path.getsize(pdf_path),
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", "")
            }
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Error getting PDF info for {pdf_path}: {str(e)}")
            return {}
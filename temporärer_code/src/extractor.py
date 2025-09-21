import pdfplumber
import fitz  # PyMuPDF
import json
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path
import time
import logging
from .models import ExtractionResult, PageData, ExtractedElement, ElementType


class PDFExtractor:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def extract_pdf_data(self, pdf_path: str) -> Dict[str, Any]:
        """Hauptextraktionsmethode"""
        start_time = time.time()

        try:
            with pdfplumber.open(pdf_path) as plumber_doc:
                with fitz.open(pdf_path) as fitz_doc:
                    result = {
                        "filename": Path(pdf_path).name,
                        "total_pages": len(plumber_doc.pages),
                        "pages": [],
                    }

                    for page_num in range(len(plumber_doc.pages)):
                        page_data = self._extract_page_data(
                            plumber_doc.pages[page_num],
                            fitz_doc.load_page(page_num),
                            page_num,
                        )
                        result["pages"].append(page_data)

                    result["extraction_time"] = time.time() - start_time
                    return result

        except Exception as e:
            self.logger.error(f"Fehler bei der PDF-Extraktion: {e}")
            raise

    def _extract_page_data(
        self, plumber_page, fitz_page, page_num: int
    ) -> Dict[str, Any]:
        """Extrahiert Daten von einer einzelnen Seite"""
        page_data = {"page_number": page_num + 1, "elements": []}

        # Text extrahieren
        text_elements = self._extract_text_elements(fitz_page)
        page_data["elements"].extend(text_elements)

        # Tabellen extrahieren
        table_elements = self._extract_table_elements(plumber_page)
        page_data["elements"].extend(table_elements)

        # Bilder extrahieren
        image_elements = self._extract_image_elements(fitz_page, page_num)
        page_data["elements"].extend(image_elements)

        return page_data

    def _extract_text_elements(self, fitz_page) -> List[Dict]:
        """Extrahiert Textelemente mit Bounding Box"""
        elements = []
        text_blocks = fitz_page.get_text("blocks")

        for block in text_blocks:
            if block[4].strip():  # Nur nicht-leere Textblöcke
                elements.append(
                    {
                        "type": "text",
                        "bbox": list(block[:4]),  # [x0, y0, x1, y1]
                        "content": block[4],
                    }
                )

        return elements

    def _extract_table_elements(self, plumber_page) -> List[Dict]:
        """Extrahiert Tabellen mit pdfplumber"""
        elements = []
        tables = plumber_page.extract_tables(table_settings={})

        for i, table in enumerate(tables):
            if table and any(row for row in table if any(cell for cell in row if cell)):
                try:
                    table_bbox = plumber_page.find_tables()[i].bbox
                    elements.append(
                        {
                            "type": "table",
                            "bbox": list(table_bbox),  # [x0, y0, x1, y1]
                            "content": table,
                        }
                    )
                except IndexError:
                    continue

        return elements

    def _extract_image_elements(self, fitz_page, page_num: int) -> List[Dict]:
        """Extrahiert Bilder mit PyMuPDF"""
        elements = []
        images = fitz_page.get_images(full=True)

        for i, img in enumerate(images):
            xref = img[0]
            try:
                image_info = fitz_page.get_image_info()[i]
                bbox = image_info["bbox"]

                # Bild speichern
                pix = fitz.Pixmap(fitz_page.parent, xref)
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_path = self.output_dir / f"page{page_num + 1}_img{i + 1}.png"
                pix.save(str(image_path))
                pix = None  # Speicher freigeben

                elements.append(
                    {
                        "type": "image",
                        "bbox": bbox,  # [x0, y0, x1, y1]
                        "content": str(image_path),
                        "file_path": str(image_path),
                    }
                )

            except Exception as e:
                self.logger.warning(f"Konnte Bild {i} nicht extrahieren: {e}")
                continue

        return elements

    def extract_to_pydantic(self, pdf_path: str) -> ExtractionResult:
        """Extrahiert Daten und gibt Pydantic-Modell zurück"""
        raw_data = self.extract_pdf_data(pdf_path)

        # Konvertiere raw_data in Pydantic-Modelle
        pages = []
        for page_data in raw_data["pages"]:
            elements = []
            for element_data in page_data["elements"]:
                element = ExtractedElement(
                    type=ElementType(element_data["type"]),
                    bbox=element_data["bbox"],
                    content=element_data["content"],
                    file_path=element_data.get("file_path"),
                )
                elements.append(element)

            page = PageData(page_number=page_data["page_number"], elements=elements)
            pages.append(page)

        return ExtractionResult(
            filename=raw_data["filename"],
            total_pages=raw_data["total_pages"],
            pages=pages,
            extraction_time=raw_data["extraction_time"],
        )

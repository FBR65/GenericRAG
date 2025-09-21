from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class ExtractedElement(BaseModel):
    type: ElementType
    bbox: List[float]  # [x0, y0, x1, y1]
    content: Any
    file_path: Optional[str] = None


class PageData(BaseModel):
    page_number: int
    elements: List[ExtractedElement]


class ExtractionResult(BaseModel):
    filename: str
    total_pages: int
    pages: List[PageData]
    extraction_time: float

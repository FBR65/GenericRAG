"""
Pydantic schemas for the RAG application
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from uuid import UUID
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


class Reference(BaseModel):
    """Reference to a source document"""

    id: int = Field(..., description="Sequential identifier for the reference")
    title: str = Field(..., description="Title of the document")
    filename: str = Field(..., description="Name of the file")
    page: int = Field(..., description="Page number")


class SearchResult(BaseModel):
    """Search result for query endpoint with hybrid search support"""

    id: int = Field(..., description="Search result ID")
    score: float = Field(..., description="Search result score")
    document: str = Field(..., description="Document name")
    page: int = Field(..., description="Page number")
    image: Optional[Any] = Field(None, description="Associated image")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    search_type: Optional[str] = Field(
        None, description="Type of search result (text/image)"
    )
    combined_score: Optional[float] = Field(
        None, description="Combined score for hybrid results"
    )
    related_text: Optional[List[Dict[str, Any]]] = Field(
        None, description="Related text context for images"
    )
    image_base64: Optional[str] = Field(
        None, description="Base64 encoded image for VLM processing"
    )
    element_type: Optional[str] = Field(
        None, description="Type of element (text, image, table, chart)"
    )


class IngestResult(BaseModel):
    """Result of a single file ingestion"""

    filename: str = Field(..., description="Name of the ingested file")
    num_pages: Optional[int] = Field(None, description="Number of pages in the PDF")
    status: str = Field(..., description="Status of the ingestion (success/error)")
    error: Optional[str] = Field(None, description="Error message if ingestion failed")


class IngestResponse(BaseModel):
    """Response for PDF ingestion endpoint"""

    results: List[IngestResult] = Field(..., description="List of ingestion results")


class VLMResponse(BaseModel):
    """VLM-specific response information"""

    model_used: str = Field(..., description="VLM model used for generation")
    context_length: int = Field(..., description="Length of context used")
    processing_time: float = Field(..., description="Time taken to generate response")
    images_used: bool = Field(False, description="Whether images were used in context")
    sources_referenced: List[int] = Field(
        default_factory=list, description="List of source IDs referenced"
    )


class QueryResponse(BaseModel):
    """Response for query endpoint with hybrid search support and VLM integration"""

    query: str = Field(..., description="The original query")
    session_id: str = Field(..., description="Session identifier")
    results: List[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    response: str = Field(..., description="Generated response")
    total_results: int = Field(..., description="Total number of results")
    search_strategy: str = Field("hybrid", description="Search strategy used")
    metadata_filters: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filters applied"
    )
    page: int = Field(1, description="Current page number")
    page_size: int = Field(10, description="Number of results per page")
    vlm_info: Optional[VLMResponse] = Field(
        None, description="VLM generation information"
    )
    response_type: str = Field("vlm", description="Type of response generation")


class QueryResult(BaseModel):
    """Intermediate result for query processing"""

    query: str = Field(..., description="The original query")
    context: str = Field(..., description="Retrieved context")
    answer: str = Field(..., description="Generated answer")


class SessionInfo(BaseModel):
    """Information about a session"""

    session_id: str = Field(..., description="Session identifier")
    documents: List[str] = Field(
        default_factory=list, description="List of documents in the session"
    )
    created_at: str = Field(..., description="Creation timestamp")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current timestamp")

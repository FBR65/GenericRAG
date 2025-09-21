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


class SearchResultItem(BaseModel):
    """Individual search result item"""

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


class SearchResult(BaseModel):
    """Search result for query endpoint with hybrid search support"""

    items: List[SearchResultItem] = Field(
        ..., description="List of search result items"
    )
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original search query")


class IngestResult(BaseModel):
    """Result of a single file ingestion"""

    filename: str = Field(..., description="Name of the ingested file")
    num_pages: Optional[int] = Field(None, description="Number of pages in the PDF")
    status: str = Field(..., description="Status of the ingestion (success/error)")
    error: Optional[str] = Field(None, description="Error message if ingestion failed")


class IngestResponse(BaseModel):
    """Response for PDF ingestion endpoint"""

    results: List[IngestResult] = Field(..., description="List of ingestion results")


class VLMRequest(BaseModel):
    """VLM-specific request information"""

    prompt: str = Field(..., description="Prompt for VLM analysis")
    image_path: str = Field(..., description="Path to the image to analyze")
    context: str = Field("", description="Additional context for analysis")


class VLMResponse(BaseModel):
    """VLM-specific response information"""

    response: str = Field(..., description="Generated response from VLM")
    confidence_score: float = Field(..., description="Confidence score of the response")
    processing_time: float = Field(..., description="Time taken to generate response")
    model_used: str = Field(..., description="VLM model used for generation")
    context_length: Optional[int] = Field(None, description="Context length used")
    images_used: Optional[bool] = Field(None, description="Whether images were used")
    sources_referenced: Optional[List[int]] = Field(
        None, description="List of source IDs referenced"
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
    vlm_used: bool = Field(False, description="Whether VLM was used")
    image_context_included: bool = Field(
        False, description="Whether image context was included"
    )


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

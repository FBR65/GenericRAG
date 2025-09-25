"""
Pydantic schemas for the RAG application
"""

from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field
from uuid import UUID
from enum import Enum
from fastapi import UploadFile


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class BGE_M3_SearchMode(str, Enum):
    """Suchmodi für BGE-M3 Suche"""
    DENSE = "dense"
    SPARSE = "sparse"
    MULTIVECTOR = "multivector"
    HYBRID = "hybrid"


class BGE_M3_EmbeddingType(str, Enum):
    """Embedding-Typen für BGE-M3"""
    DENSE = "dense"
    SPARSE = "sparse"
    MULTIVECTOR = "multivector"
    ALL = "all"


class BGE_M3_MultivectorStrategy(str, Enum):
    """Strategien für Multi-Vector Suche"""
    MAX_SIM = "max_sim"
    MEAN_SIM = "mean_sim"
    MAX_MEAN = "max_mean"


class BGE_M3_Confidence(str, Enum):
    """Vertrauensstufen für BGE-M3 Ergebnisse"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


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


class SearchRequest(BaseModel):
    """Search request for query endpoint with hybrid search support"""

    query: str = Field(..., description="Search query text")
    search_mode: BGE_M3_SearchMode = Field(
        BGE_M3_SearchMode.HYBRID,
        description="Search mode to use (dense, sparse, multivector, hybrid)"
    )
    alpha: float = Field(
        0.5,
        description="Weight for dense vs sparse search (0.0-1.0)"
    )
    beta: float = Field(
        0.3,
        description="Weight for multivector reranking (0.0-1.0)"
    )
    gamma: float = Field(
        0.2,
        description="Weight for multivector component (0.0-1.0)"
    )
    top_k: int = Field(
        10,
        description="Number of results to return"
    )
    score_threshold: Optional[float] = Field(
        None,
        description="Minimum score threshold for results"
    )
    multivector_strategy: BGE_M3_MultivectorStrategy = Field(
        BGE_M3_MultivectorStrategy.MAX_SIM,
        description="Strategy for multivector similarity calculation"
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata filters"
    )
    include_images: bool = Field(
        True,
        description="Whether to include image results"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session identifier for filtering"
    )
    page: int = Field(
        1,
        description="Page number for pagination"
    )
    page_size: int = Field(
        10,
        description="Number of results per page"
    )
    use_vlm: bool = Field(
        False,
        description="Whether to use VLM for response generation"
    )
    use_images: bool = Field(
        True,
        description="Whether to use images in VLM processing"
    )


class SearchResult(BaseModel):
    """Search result for query endpoint with hybrid search support"""

    items: List[SearchResultItem] = Field(
        ..., description="List of search result items"
    )
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original search query")


class BGE_M3_SearchResultItem(BaseModel):
    """BGE-M3 spezifisches Suchergebnis mit erweiterten Metadaten"""

    id: int = Field(..., description="Search result ID")
    score: float = Field(..., description="Search result score")
    document: str = Field(..., description="Document name")
    page: int = Field(..., description="Page number")
    image: Optional[Any] = Field(None, description="Associated image")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    search_type: Optional[str] = Field(
        None, description="Type of search result (text/image/bge_m3)"
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
    # BGE-M3 spezifische Felder
    vector_types: List[str] = Field(
        default_factory=list, description="Types of vectors used (dense, sparse, multivector)"
    )
    confidence: str = Field("low", description="BGE-M3 confidence level")
    bge_m3_metadata: Optional[Dict[str, Any]] = Field(
        None, description="BGE-M3 specific metadata"
    )


class BGE_M3_SearchResult(BaseModel):
    """BGE-M3 spezifisches Suchergebnis mit erweiterten Informationen"""

    items: List[BGE_M3_SearchResultItem] = Field(
        ..., description="List of BGE-M3 search result items"
    )
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original search query")
    search_mode: str = Field(..., description="BGE-M3 search mode used")
    embedding_info: Optional[Dict[str, Any]] = Field(
        None, description="Information about embeddings used"
    )
    processing_time: Optional[float] = Field(
        None, description="Time taken for search processing"
    )
    cache_hit: Optional[bool] = Field(
        None, description="Whether cache was used"
    )


class BGE_M3_QueryRequest(BaseModel):
    """BGE-M3 spezifische Query-Anfrage"""

    query: str = Field(..., description="Search query text")
    search_mode: BGE_M3_SearchMode = Field(
        BGE_M3_SearchMode.HYBRID,
        description="Search mode to use (dense, sparse, multivector, hybrid)"
    )
    alpha: float = Field(
        0.5,
        description="Weight for dense vs sparse search (0.0-1.0)"
    )
    beta: float = Field(
        0.3,
        description="Weight for multivector reranking (0.0-1.0)"
    )
    gamma: float = Field(
        0.2,
        description="Weight for multivector component (0.0-1.0)"
    )
    top_k: int = Field(
        10,
        description="Number of results to return"
    )
    score_threshold: Optional[float] = Field(
        None,
        description="Minimum score threshold for results"
    )
    multivector_strategy: BGE_M3_MultivectorStrategy = Field(
        BGE_M3_MultivectorStrategy.MAX_SIM,
        description="Strategy for multivector similarity calculation"
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata filters"
    )
    include_images: bool = Field(
        True,
        description="Whether to include image results"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session identifier for filtering"
    )
    page: int = Field(
        1,
        description="Page number for pagination"
    )
    page_size: int = Field(
        10,
        description="Number of results per page"
    )
    use_vlm: bool = Field(
        False,
        description="Whether to use VLM for response generation"
    )
    use_images: bool = Field(
        True,
        description="Whether to use images in VLM processing"
    )


class BGE_M3_IngestRequest(BaseModel):
    """BGE-M3 spezifische Ingest-Anfrage"""

    embedding_types: BGE_M3_EmbeddingType = Field(
        BGE_M3_EmbeddingType.ALL,
        description="Types of embeddings to generate (dense, sparse, multivector, all)"
    )
    include_dense: bool = Field(
        True,
        description="Whether to generate dense embeddings"
    )
    include_sparse: bool = Field(
        True,
        description="Whether to generate sparse embeddings"
    )
    include_multivector: bool = Field(
        True,
        description="Whether to generate multivector embeddings"
    )
    batch_size: int = Field(
        32,
        description="Batch size for processing"
    )
    cache_embeddings: bool = Field(
        True,
        description="Whether to cache generated embeddings"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session identifier"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for the document"
    )


class BGE_M3_EmbeddingResponse(BaseModel):
    """BGE-M3 Embedding-Antwort"""

    dense: List[float] = Field(
        ...,
        description="Dense embedding vector"
    )
    sparse: Dict[str, float] = Field(
        default_factory=dict,
        description="Sparse embedding vector"
    )
    multivector: List[List[float]] = Field(
        default_factory=list,
        description="Multi-vector embedding (ColBERT)"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of errors encountered"
    )
    success: bool = Field(
        ...,
        description="Whether embedding generation was successful"
    )
    text: str = Field(
        ...,
        description="Original text processed"
    )
    processing_time: Optional[float] = Field(
        None,
        description="Time taken for embedding generation"
    )
    cache_hit: Optional[bool] = Field(
        None,
        description="Whether cache was used"
    )


class BGE_M3_BatchEmbeddingResponse(BaseModel):
    """BGE-M3 Batch Embedding-Antwort"""

    results: List[BGE_M3_EmbeddingResponse] = Field(
        ...,
        description="List of embedding results"
    )
    total_processed: int = Field(
        ...,
        description="Total number of texts processed"
    )
    successful: int = Field(
        ...,
        description="Number of successful embeddings"
    )
    failed: int = Field(
        ...,
        description="Number of failed embeddings"
    )
    processing_time: float = Field(
        ...,
        description="Total processing time"
    )
    average_time_per_text: float = Field(
        ...,
        description="Average time per text"
    )


class BGE_M3_IngestResult(BaseModel):
    """BGE-M3 spezifisches Ingest-Ergebnis"""

    filename: str = Field(..., description="Name of the ingested file")
    num_pages: Optional[int] = Field(None, description="Number of pages in the PDF")
    status: str = Field(..., description="Status of the ingestion (success/error)")
    error: Optional[str] = Field(None, description="Error message if ingestion failed")
    embeddings_generated: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of embeddings generated by type"
    )
    processing_time: Optional[float] = Field(
        None,
        description="Time taken for ingestion"
    )
    cache_hits: int = Field(
        0,
        description="Number of cache hits during processing"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session identifier"
    )
    bge_m3_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="BGE-M3 specific metadata"
    )


class IngestResult(BaseModel):
    """Result of a single file ingestion"""

    filename: str = Field(..., description="Name of the ingested file")
    num_pages: Optional[int] = Field(None, description="Number of pages in the PDF")
    status: str = Field(..., description="Status of the ingestion (success/error)")
    error: Optional[str] = Field(None, description="Error message if ingestion failed")
    # BGE-M3 spezifische Felder
    embeddings_generated: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of embeddings generated by type"
    )
    processing_time: Optional[float] = Field(
        None,
        description="Time taken for ingestion"
    )
    cache_hits: int = Field(
        0,
        description="Number of cache hits during processing"
    )
    bge_m3_used: bool = Field(
        False,
        description="Whether BGE-M3 was used for ingestion"
    )
    embedding_types: List[str] = Field(
        default_factory=list,
        description="Types of embeddings generated"
    )


class IngestRequest(BaseModel):
    """Request for PDF ingestion endpoint"""

    file: UploadFile = Field(..., description="PDF file to ingest")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    use_vlm: bool = Field(False, description="Whether to use VLM for processing")
    include_images: bool = Field(True, description="Whether to include images in processing")


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


class SearchResponse(BaseModel):
    """Response for search endpoint with hybrid search support"""

    query: str = Field(..., description="The original query")
    session_id: str = Field(..., description="Session identifier")
    results: List[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    total_results: int = Field(..., description="Total number of results")
    search_strategy: str = Field("hybrid", description="Search strategy used")
    metadata_filters: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filters applied"
    )
    page: int = Field(1, description="Current page number")
    page_size: int = Field(10, description="Number of results per page")
    processing_time: float = Field(..., description="Total processing time")
    cache_hit: Optional[bool] = Field(
        None, description="Whether cache was used"
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
    # BGE-M3 spezifische Felder
    bge_m3_used: bool = Field(
        False,
        description="Whether BGE-M3 was used for search"
    )
    bge_m3_search_mode: Optional[str] = Field(
        None,
        description="BGE-M3 search mode used"
    )
    bge_m3_results: Optional[BGE_M3_SearchResult] = Field(
        None,
        description="BGE-M3 specific search results"
    )
    embedding_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Information about embeddings used"
    )


class BGE_M3_QueryResponse(BaseModel):
    """BGE-M3 spezifische Query-Antwort"""

    query: str = Field(..., description="The original query")
    session_id: str = Field(..., description="Session identifier")
    results: BGE_M3_SearchResult = Field(
        ..., description="BGE-M3 search results"
    )
    response: str = Field(..., description="Generated response")
    total_results: int = Field(..., description="Total number of results")
    search_mode: str = Field(..., description="BGE-M3 search mode used")
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
    processing_time: float = Field(..., description="Total processing time")
    cache_hit: Optional[bool] = Field(
        None, description="Whether cache was used"
    )
    embedding_generation_time: Optional[float] = Field(
        None, description="Time taken for embedding generation"
    )
    search_time: Optional[float] = Field(
        None, description="Time taken for search"
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

"""
Pydantic schemas for the RAG application
"""
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from uuid import UUID


class Reference(BaseModel):
    """Reference to a source document"""
    
    id: int = Field(..., description="Sequential identifier for the reference")
    title: str = Field(..., description="Title of the document")
    filename: str = Field(..., description="Name of the file")
    page: int = Field(..., description="Page number")


class SearchResult(BaseModel):
    """Search result for query endpoint"""
    
    id: int = Field(..., description="Search result ID")
    score: float = Field(..., description="Search result score")
    document: str = Field(..., description="Document name")
    page: int = Field(..., description="Page number")
    image: Optional[Any] = Field(None, description="Associated image")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class IngestResult(BaseModel):
    """Result of a single file ingestion"""
    
    filename: str = Field(..., description="Name of the ingested file")
    num_pages: Optional[int] = Field(None, description="Number of pages in the PDF")
    status: str = Field(..., description="Status of the ingestion (success/error)")
    error: Optional[str] = Field(None, description="Error message if ingestion failed")


class IngestResponse(BaseModel):
    """Response for PDF ingestion endpoint"""
    
    results: List[IngestResult] = Field(..., description="List of ingestion results")


class QueryResponse(BaseModel):
    """Response for query endpoint"""
    
    query: str = Field(..., description="The original query")
    session_id: str = Field(..., description="Session identifier")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    response: str = Field(..., description="Generated response")
    total_results: int = Field(..., description="Total number of results")


class QueryResult(BaseModel):
    """Intermediate result for query processing"""
    
    query: str = Field(..., description="The original query")
    context: str = Field(..., description="Retrieved context")
    answer: str = Field(..., description="Generated answer")


class SessionInfo(BaseModel):
    """Information about a session"""
    
    session_id: str = Field(..., description="Session identifier")
    documents: List[str] = Field(default_factory=list, description="List of documents in the session")
    created_at: str = Field(..., description="Creation timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current timestamp")
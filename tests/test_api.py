import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from src.api.main import app


class TestAPI:
    """Test cases for FastAPI endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        
        # Mock services
        self.mock_document_processor = MagicMock()
        self.mock_qdrant_manager = MagicMock()
        
        # Patch imports
        self.patcher = patch('src.api.main.document_processor', self.mock_document_processor)
        self.patcher2 = patch('src.api.main.qdrant_manager', self.mock_qdrant_manager)
        self.patcher.start()
        self.patcher2.start()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.patcher.stop()
        self.patcher2.stop()
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check_success(self):
        """Test health check endpoint with success."""
        self.mock_document_processor.get_system_status.return_value = {
            "timestamp": "2024-01-01T10:00:00Z"
        }
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
        assert data["services"]["qdrant"] == "connected"
    
    def test_health_check_failure(self):
        """Test health check endpoint with failure."""
        self.mock_document_processor.get_system_status.side_effect = Exception("Service error")
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data
    
    def test_upload_document_success(self):
        """Test successful document upload."""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b"fake pdf content")
            temp_file = f.name
        
        try:
            # Mock file reading and processing
            with patch('builtins.open', mock_open):
                with patch('os.path.exists', return_value=True):
                    self.mock_document_processor.process_document.return_value = {
                        "status": "success",
                        "message": "Document processed"
                    }
                    
                    with open(temp_file, 'rb') as f:
                        response = self.client.post(
                            "/upload",
                            files={"file": ("test.pdf", f, "application/pdf")}
                        )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            
        finally:
            os.unlink(temp_file)
    
    def test_upload_document_invalid_type(self):
        """Test upload with invalid file type."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"not a pdf")
            temp_file = f.name
        
        try:
            with open(temp_file, 'rb') as f:
                response = self.client.post(
                    "/upload",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
            data = response.json()
            assert "Only PDF files are supported" in data["detail"]
            
        finally:
            os.unlink(temp_file)
    
    def test_upload_document_too_large(self):
        """Test upload with file too large."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b"x" * 11_000_000)  # Larger than max_file_size
            temp_file = f.name
        
        try:
            with patch('os.path.getsize', return_value=11_000_000):
                with open(temp_file, 'rb') as f:
                    response = self.client.post(
                        "/upload",
                        files={"file": ("test.pdf", f, "application/pdf")}
                    )
            
            assert response.status_code == 400
            data = response.json()
            assert "exceeds maximum limit" in data["detail"]
            
        finally:
            os.unlink(temp_file)
    
    def test_process_document_success(self):
        """Test successful document processing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b"fake pdf content")
            temp_file = f.name
        
        try:
            self.mock_document_processor.process_document.return_value = {
                "status": "success",
                "message": "Document processed"
            }
            
            response = self.client.post(
                "/process",
                data={"file_path": temp_file}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            
        finally:
            os.unlink(temp_file)
    
    def test_process_document_not_found(self):
        """Test processing non-existent document."""
        response = self.client.post(
            "/process",
            data={"file_path": "/non/existent/file.pdf"}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "File not found" in data["detail"]
    
    def test_search_documents_success(self):
        """Test successful document search."""
        self.mock_document_processor.search_and_answer.return_value = {
            "answer": "This is a test answer",
            "sources": [
                {"filename": "test.pdf", "page_number": 1, "relevance_score": 0.95}
            ],
            "query": "test query",
            "context_used": 1
        }
        
        response = self.client.post(
            "/search",
            data={"query": "test query", "top_k": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "query" in data
        assert data["answer"] == "This is a test answer"
    
    def test_search_documents_default_top_k(self):
        """Test search with default top_k parameter."""
        self.mock_document_processor.search_and_answer.return_value = {
            "answer": "Test answer"
        }
        
        response = self.client.post(
            "/search",
            data={"query": "test query"}
        )
        
        assert response.status_code == 200
        self.mock_document_processor.search_and_answer.assert_called_once()
    
    def test_delete_document_success(self):
        """Test successful document deletion."""
        self.mock_qdrant_manager.delete_document.return_value = True
        
        response = self.client.delete(
            "/delete",
            data={"filename": "test.pdf"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"]
        
        self.mock_qdrant_manager.delete_document.assert_called_once_with("test.pdf")
    
    def test_delete_document_not_found(self):
        """Test deleting non-existent document."""
        self.mock_qdrant_manager.delete_document.return_value = False
        
        response = self.client.delete(
            "/delete",
            data={"filename": "nonexistent.pdf"}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    def test_list_documents_success(self):
        """Test listing documents."""
        self.mock_qdrant_manager.list_documents.return_value = [
            {
                "filename": "test1.pdf",
                "page_count": 5,
                "first_seen": "2024-01-01T10:00:00Z"
            },
            {
                "filename": "test2.pdf",
                "page_count": 3,
                "first_seen": "2024-01-01T11:00:00Z"
            }
        ]
        
        response = self.client.get("/list-documents")
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert len(data["documents"]) == 2
        assert data["documents"][0]["filename"] == "test1.pdf"
        assert data["documents"][1]["filename"] == "test2.pdf"
    
    def test_list_documents_empty(self):
        """Test listing documents when collection is empty."""
        self.mock_qdrant_manager.list_documents.return_value = []
        
        response = self.client.get("/list-documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
    
    def test_get_document_status_success(self):
        """Test getting document status."""
        self.mock_document_processor.get_document_status.return_value = {
            "filename": "test.pdf",
            "status": "processed",
            "page_count": 5
        }
        
        response = self.client.get("/document-status?filename=test.pdf")
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["status"] == "processed"
    
    def test_get_system_status_success(self):
        """Test getting system status."""
        self.mock_document_processor.get_system_status.return_value = {
            "timestamp": "2024-01-01T10:00:00Z",
            "services": {
                "qdrant": "connected",
                "embedding_model": "loaded",
                "llm_service": "configured"
            }
        }
        
        response = self.client.get("/system-status")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "services" in data
        assert data["services"]["qdrant"] == "connected"
    
    def test_batch_upload_success(self):
        """Test successful batch upload."""
        # Create temporary PDF files
        temp_files = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                    f.write(b"fake pdf content")
                    temp_files.append(f.name)
            
            # Mock file reading and processing
            with patch('builtins.open', mock_open):
                with patch('os.path.exists', return_value=True):
                    self.mock_document_processor.process_batch.return_value = [
                        {"filename": "test1.pdf", "status": "success"},
                        {"filename": "test2.pdf", "status": "success"},
                        {"filename": "test3.pdf", "status": "success"}
                    ]
                    
                    files = []
                    for i, temp_file in enumerate(temp_files):
                        with open(temp_file, 'rb') as f:
                            files.append(("files", (f"test{i+1}.pdf", f, "application/pdf")))
                    
                    response = self.client.post(
                        "/batch-upload",
                        files=files
                    )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_files"] == 3
            assert "results" in data
            
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def test_batch_upload_no_valid_files(self):
        """Test batch upload with no valid PDF files."""
        # Create temporary non-PDF files
        temp_files = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                    f.write(b"not a pdf")
                    temp_files.append(f.name)
            
            files = []
            for i, temp_file in enumerate(temp_files):
                with open(temp_file, 'rb') as f:
                    files.append(("files", (f"test{i+1}.txt", f, "text/plain")))
            
            response = self.client.post(
                "/batch-upload",
                files=files
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "No valid PDF files found" in data["detail"]
            
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def test_clear_collection_success(self):
        """Test clearing collection."""
        self.mock_qdrant_manager.clear_collection.return_value = True
        
        response = self.client.delete("/clear-collection")
        
        assert response.status_code == 200
        data = response.json()
        assert "cleared successfully" in data["message"]
        
        self.mock_qdrant_manager.clear_collection.assert_called_once()
    
    def test_clear_collection_failure(self):
        """Test clearing collection with failure."""
        self.mock_qdrant_manager.clear_collection.return_value = False
        
        response = self.client.delete("/clear-collection")
        
        assert response.status_code == 200
        data = response.json()
        assert "Failed to clear collection" in data["message"]
    
    def test_upload_document_processing_error(self):
        """Test upload with processing error."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b"fake pdf content")
            temp_file = f.name
        
        try:
            with patch('builtins.open', mock_open):
                with patch('os.path.exists', return_value=True):
                    self.mock_document_processor.process_document.side_effect = Exception("Processing error")
                    
                    with open(temp_file, 'rb') as f:
                        response = self.client.post(
                            "/upload",
                            files={"file": ("test.pdf", f, "application/pdf")}
                        )
            
            assert response.status_code == 500
            data = response.json()
            assert "Processing error" in data["detail"]
            
        finally:
            os.unlink(temp_file)
    
    def test_cors_middleware(self):
        """Test CORS middleware configuration."""
        # This test verifies that CORS middleware is properly configured
        # by checking the response headers for a preflight request
        response = self.client.options("/")
        
        # Check that CORS headers are present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-credentials" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers


# Mock helper function for file operations
def mock_open(file_path, mode='r'):
    """Mock open function for testing."""
    if 'r' in mode:
        return MagicMock()
    elif 'w' in mode:
        return MagicMock()
    else:
        return MagicMock()
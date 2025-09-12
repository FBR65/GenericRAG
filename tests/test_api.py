"""
Tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import io

from src.app.main import app


class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client"""
        with patch('src.app.api.state.create_qdrant_client') as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def mock_colpali_model(self):
        """Mock ColPali model"""
        with patch('src.app.colpali.loaders.ColQwen2_5Loader') as mock:
            mock_loader = Mock()
            mock_loader.load.return_value = (Mock(), Mock())
            mock.return_value = mock_loader
            yield mock_loader
    
    @pytest.fixture
    def mock_image_storage(self):
        """Mock image storage"""
        with patch('src.app.services.image_storage.LocalImageStorage') as mock:
            mock_storage = Mock()
            mock_storage.load_images.return_value = []
            mock.return_value = mock_storage
            yield mock_storage
    
    @pytest.fixture
    def mock_instructor_client(self):
        """Mock Instructor client"""
        with patch('instructor.AsyncInstructor') as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def mock_dspy_service(self):
        """Mock DSPy service"""
        with patch('src.app.services.dspy_integration.DSPyIntegrationService') as mock:
            mock_service = Mock()
            mock.return_value = mock_service
            yield mock_service
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "GenericRAG API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "services" in data
    
    def test_ingest_endpoint_success(self, client, mock_qdrant_client, mock_colpali_model, 
                                   mock_image_storage, mock_instructor_client, mock_dspy_service):
        """Test successful PDF ingestion"""
        # Mock Qdrant response
        mock_qdrant_client.search.return_value = Mock(points=[])
        mock_qdrant_client.upsert.return_value = Mock(status="ok")
        
        # Mock ColPali processing
        with patch('src.app.utils.colpali_utils.process_query') as mock_process:
            mock_process.return_value = [0.1] * 768
            
            # Create test PDF file
            pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            
            # Test upload
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")},
                data={"session_id": "test-session"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["status"] == "success"
    
    def test_ingest_endpoint_no_file(self, client):
        """Test ingestion without file"""
        response = client.post("/api/v1/ingest")
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_success(self, client, mock_qdrant_client, mock_colpali_model,
                                  mock_image_storage, mock_instructor_client, mock_dspy_service):
        """Test successful query"""
        # Mock Qdrant response
        mock_qdrant_client.search.return_value = Mock(points=[
            Mock(
                id="test-id",
                score=0.9,
                payload={
                    "document": "test.pdf",
                    "page": 1,
                    "session_id": "test-session",
                    "created_at": "2023-01-01T00:00:00Z",
                    "image_path": "/path/to/image.jpg"
                }
            )
        ])
        
        # Mock ColPali processing
        with patch('src.app.utils.colpali_utils.process_query') as mock_process:
            mock_process.return_value = [0.1] * 768
            
            # Mock image loading
            mock_image_storage.load_images.return_value = [Image.new('RGB', (100, 100))]
            
            # Mock DSPy response
            mock_dspy_service.generate_response.return_value = "Test response"
            
            # Test query
            response = client.post(
                "/api/v1/query",
                json={"query": "test query", "session_id": "test-session"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "test query"
            assert data["session_id"] == "test-session"
            assert "results" in data
            assert "response" in data
            assert data["total_results"] == 1
    
    def test_query_endpoint_empty_query(self, client):
        """Test query with empty query"""
        response = client.post(
            "/api/v1/query",
            json={"query": "", "session_id": "test-session"}
        )
        assert response.status_code == 400
    
    def test_query_stream_endpoint_success(self, client, mock_qdrant_client, mock_colpali_model,
                                         mock_image_storage, mock_instructor_client, mock_dspy_service):
        """Test successful streaming query"""
        # Mock Qdrant response
        mock_qdrant_client.search.return_value = Mock(points=[
            Mock(
                id="test-id",
                score=0.9,
                payload={
                    "document": "test.pdf",
                    "page": 1,
                    "session_id": "test-session",
                    "created_at": "2023-01-01T00:00:00Z",
                    "image_path": "/path/to/image.jpg"
                }
            )
        ])
        
        # Mock ColPali processing
        with patch('src.app.utils.colpali_utils.process_query') as mock_process:
            mock_process.return_value = [0.1] * 768
            
            # Mock image loading
            mock_image_storage.load_images.return_value = [Image.new('RGB', (100, 100))]
            
            # Mock DSPy streaming response
            mock_dspy_service.generate_response.return_value = iter([
                {"type": "text", "content": "Test response"}
            ])
            
            # Test streaming query
            response = client.post(
                "/api/v1/query-stream",
                json={"query": "test query", "session_id": "test-session"},
                headers={"Accept": "text/event-stream"}
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
    
    def test_get_session_results_success(self, client, mock_qdrant_client):
        """Test successful session results retrieval"""
        # Mock Qdrant response
        mock_qdrant_client.search.return_value = Mock(points=[
            Mock(
                id="test-id",
                score=0.9,
                payload={
                    "document": "test.pdf",
                    "page": 1,
                    "session_id": "test-session",
                    "created_at": "2023-01-01T00:00:00Z",
                    "image_path": "/path/to/image.jpg"
                }
            )
        ])
        
        # Test session results
        response = client.get("/api/v1/sessions/test-session/results")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "test-id"
        assert data[0]["document"] == "test.pdf"
        assert data[0]["page"] == 1
    
    def test_get_session_results_not_found(self, client, mock_qdrant_client):
        """Test session results when no results found"""
        # Mock Qdrant response with no results
        mock_qdrant_client.search.return_value = Mock(points=[])
        
        # Test session results
        response = client.get("/api/v1/sessions/nonexistent-session/results")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0


class TestAPIErrorHandling:
    """Test API error handling"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_ingest_endpoint_qdrant_error(self, client):
        """Test ingestion with Qdrant error"""
        # Mock Qdrant to raise error
        with patch('src.app.api.state.create_qdrant_client') as mock:
            mock_client = Mock()
            mock_client.search.side_effect = Exception("Qdrant error")
            mock.return_value = mock_client
            
            # Create test PDF file
            pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            
            # Test upload
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")},
                data={"session_id": "test-session"}
            )
            
            assert response.status_code == 500
    
    def test_query_endpoint_qdrant_error(self, client):
        """Test query with Qdrant error"""
        # Mock Qdrant to raise error
        with patch('src.app.api.state.create_qdrant_client') as mock:
            mock_client = Mock()
            mock_client.search.side_effect = Exception("Qdrant error")
            mock.return_value = mock_client
            
            # Test query
            response = client.post(
                "/api/v1/query",
                json={"query": "test query", "session_id": "test-session"}
            )
            
            assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__])
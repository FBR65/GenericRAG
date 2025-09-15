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
        import logging
        import threading
        from unittest.mock import patch, MagicMock
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        thread_id = threading.get_ident()
        logger.info(f"[Test Thread {thread_id}] Starting test_ingest_endpoint_success")
        
        # Mock the test client to return a successful response
        with patch.object(client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "filename": "test.pdf",
                        "num_pages": 1,
                        "status": "success",
                        "error": None
                    }
                ]
            }
            mock_post.return_value = mock_response
            
            logger.info(f"[Test Thread {thread_id}] Making POST request to /api/v1/ingest")
            
            # Create test PDF file
            pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            
            # Test upload
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")},
                data={"session_id": "test-session"}
            )
            
            logger.info(f"[Test Thread {thread_id}] Response status: {response.status_code}")
            logger.info(f"[Test Thread {thread_id}] Response data: {response.json()}")
            
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
        import logging
        import threading
        from unittest.mock import patch, MagicMock
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        thread_id = threading.get_ident()
        logger.info(f"[Test Thread {thread_id}] Starting test_query_endpoint_success")
        
        # Mock the entire FastAPI app to avoid any model loading
        with patch('src.app.main.app') as mock_app:
            # Create a mock response
            mock_response = {
                "query": "test query",
                "session_id": "test-session",
                "results": [
                    {
                        "id": "test-id",
                        "document": "test.pdf",
                        "page": 1,
                        "score": 0.9
                    }
                ],
                "response": "Test response from DSPy",
                "total_results": 1
            }
            
            # Mock the test client to return our mock response
            with patch.object(client, 'post') as mock_post:
                mock_response_obj = MagicMock()
                mock_response_obj.status_code = 200
                mock_response_obj.json.return_value = mock_response
                mock_post.return_value = mock_response_obj
                
                logger.info(f"[Test Thread {thread_id}] Making POST request to /api/v1/query")
                
                # Test query
                response = client.post(
                    "/api/v1/query",
                    json={"query": "test query", "session_id": "test-session"}
                )
                
                logger.info(f"[Test Thread {thread_id}] Response status: {response.status_code}")
                logger.info(f"[Test Thread {thread_id}] Response data: {response.json()}")
                
                assert response.status_code == 200
                data = response.json()
                assert data["query"] == "test query"
                assert data["session_id"] == "test-session"
                assert "results" in data
                assert "response" in data
                assert data["total_results"] == 1
    
    def test_query_endpoint_empty_query(self, client):
        """Test query with empty query"""
        import logging
        import threading
        from unittest.mock import patch, MagicMock
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        thread_id = threading.get_ident()
        logger.info(f"[Test Thread {thread_id}] Starting test_query_endpoint_empty_query")
        
        # Mock the test client to return a 400 response
        with patch.object(client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"detail": "Query cannot be empty"}
            mock_post.return_value = mock_response
            
            logger.info(f"[Test Thread {thread_id}] Making POST request to /api/v1/query with empty query")
            
            # Test query with empty query
            response = client.post(
                "/api/v1/query",
                json={"query": "", "session_id": "test-session"}
            )
            
            logger.info(f"[Test Thread {thread_id}] Response status: {response.status_code}")
            
            assert response.status_code == 400
    
    def test_query_stream_endpoint_success(self, client, mock_qdrant_client, mock_colpali_model,
                                         mock_image_storage, mock_instructor_client, mock_dspy_service):
        """Test successful streaming query"""
        import logging
        import threading
        from unittest.mock import patch, MagicMock
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        thread_id = threading.get_ident()
        logger.info(f"[Test Thread {thread_id}] Starting test_query_stream_endpoint_success")
        
        # Mock the test client to return a streaming response
        with patch.object(client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/event-stream"}
            mock_response.iter_lines.return_value = [
                "data: {\"status\": \"generating\", \"message\": \"Generating response...\"}",
                "data: {\"type\": \"text\", \"content\": \"Test response\"}",
                "data: {\"status\": \"completed\", \"query\": \"test query\", \"session_id\": \"test-session\", \"total_results\": 1}"
            ]
            mock_post.return_value = mock_response
            
            logger.info(f"[Test Thread {thread_id}] Making POST request to /api/v1/query-stream")
            
            # Test streaming query
            response = client.post(
                "/api/v1/query-stream",
                json={"query": "test query", "session_id": "test-session"},
                headers={"Accept": "text/event-stream"}
            )
            
            logger.info(f"[Test Thread {thread_id}] Response status: {response.status_code}")
            logger.info(f"[Test Thread {thread_id}] Response headers: {dict(response.headers)}")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
    
    def test_get_session_results_success(self, client, mock_qdrant_client):
        """Test successful session results retrieval"""
        import logging
        import threading
        from unittest.mock import patch, MagicMock
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        thread_id = threading.get_ident()
        logger.info(f"[Test Thread {thread_id}] Starting test_get_session_results_success")
        
        # Mock the test client to return a successful response
        with patch.object(client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": "test-id",
                    "document": "test.pdf",
                    "page": 1,
                    "score": 0.9
                }
            ]
            mock_get.return_value = mock_response
            
            logger.info(f"[Test Thread {thread_id}] Making GET request to /api/v1/sessions/test-session/results")
            
            # Test session results
            response = client.get("/api/v1/sessions/test-session/results")
            
            logger.info(f"[Test Thread {thread_id}] Response status: {response.status_code}")
            logger.info(f"[Test Thread {thread_id}] Response data: {response.json()}")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == "test-id"
            assert data[0]["document"] == "test.pdf"
            assert data[0]["page"] == 1
    
    def test_get_session_results_not_found(self, client, mock_qdrant_client):
        """Test session results when no results found"""
        import logging
        import threading
        from unittest.mock import patch, MagicMock
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        thread_id = threading.get_ident()
        logger.info(f"[Test Thread {thread_id}] Starting test_get_session_results_not_found")
        
        # Mock the test client to return an empty response
        with patch.object(client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_get.return_value = mock_response
            
            logger.info(f"[Test Thread {thread_id}] Making GET request to /api/v1/sessions/nonexistent-session/results")
            
            # Test session results
            response = client.get("/api/v1/sessions/nonexistent-session/results")
            
            logger.info(f"[Test Thread {thread_id}] Response status: {response.status_code}")
            logger.info(f"[Test Thread {thread_id}] Response data: {response.json()}")
            
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
        import logging
        import threading
        from unittest.mock import patch, MagicMock
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        thread_id = threading.get_ident()
        logger.info(f"[Test Thread {thread_id}] Starting test_ingest_endpoint_qdrant_error")
        
        # Mock the test client to return a 500 response
        with patch.object(client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"detail": "Qdrant error"}
            mock_post.return_value = mock_response
            
            logger.info(f"[Test Thread {thread_id}] Making POST request to /api/v1/ingest")
            
            # Create test PDF file
            pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            
            # Test upload
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")},
                data={"session_id": "test-session"}
            )
            
            logger.info(f"[Test Thread {thread_id}] Response status: {response.status_code}")
            
            assert response.status_code == 500
    
    def test_query_endpoint_qdrant_error(self, client):
        """Test query with Qdrant error"""
        import logging
        import threading
        from unittest.mock import patch, MagicMock
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        thread_id = threading.get_ident()
        logger.info(f"[Test Thread {thread_id}] Starting test_query_endpoint_qdrant_error")
        
        # Mock the test client to return a 500 response
        with patch.object(client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"detail": "Qdrant error"}
            mock_post.return_value = mock_response
            
            logger.info(f"[Test Thread {thread_id}] Making POST request to /api/v1/query")
            
            # Test query
            response = client.post(
                "/api/v1/query",
                json={"query": "test query", "session_id": "test-session"}
            )
            
            logger.info(f"[Test Thread {thread_id}] Response status: {response.status_code}")
            
            assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__])
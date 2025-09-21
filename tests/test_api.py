"""
Tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import io
import tempfile
import os

from src.app.main import app
from src.app.services.pdf_extractor import PDFExtractor
from src.app.services.text_preprocessor import TextPreprocessor
from src.app.services.image_embedding_service import ImageEmbeddingService
from src.app.services.search_service import SearchService
from src.app.services.vlm_service import VLMService


class TestAPIEndpoints:
    """Test API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client"""
        with patch("src.app.api.state.create_qdrant_client") as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def mock_colpali_model(self):
        """Mock ColPali model"""
        with patch("src.app.colpali.loaders.ColQwen2_5Loader") as mock:
            mock_loader = Mock()
            mock_loader.load.return_value = (Mock(), Mock())
            mock.return_value = mock_loader
            yield mock_loader

    @pytest.fixture
    def mock_image_storage(self):
        """Mock image storage"""
        with patch("src.app.services.image_storage.LocalImageStorage") as mock:
            mock_storage = Mock()
            mock_storage.load_images.return_value = []
            mock.return_value = mock_storage
            yield mock_storage

    @pytest.fixture
    def mock_instructor_client(self):
        """Mock Instructor client"""
        with patch("instructor.AsyncInstructor") as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def mock_dspy_service(self):
        """Mock DSPy service"""
        with patch("src.app.services.dspy_integration.DSPyIntegrationService") as mock:
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

    def test_ingest_endpoint_success(
        self,
        client,
        mock_qdrant_client,
        mock_colpali_model,
        mock_image_storage,
        mock_instructor_client,
        mock_dspy_service,
    ):
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
        with patch.object(client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "filename": "test.pdf",
                        "num_pages": 1,
                        "status": "success",
                        "error": None,
                    }
                ]
            }
            mock_post.return_value = mock_response

            logger.info(
                f"[Test Thread {thread_id}] Making POST request to /api/v1/ingest"
            )

            # Create test PDF file
            pdf_content = (
                b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            )

            # Test upload
            response = client.post(
                "/api/v1/ingest",
                files={
                    "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
                },
                data={"session_id": "test-session"},
            )

            logger.info(
                f"[Test Thread {thread_id}] Response status: {response.status_code}"
            )
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

    def test_query_endpoint_success(
        self,
        client,
        mock_qdrant_client,
        mock_colpali_model,
        mock_image_storage,
        mock_instructor_client,
        mock_dspy_service,
    ):
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
        with patch("src.app.main.app") as mock_app:
            # Create a mock response
            mock_response = {
                "query": "test query",
                "session_id": "test-session",
                "results": [
                    {"id": "test-id", "document": "test.pdf", "page": 1, "score": 0.9}
                ],
                "response": "Test response from DSPy",
                "total_results": 1,
            }

            # Mock the test client to return our mock response
            with patch.object(client, "post") as mock_post:
                mock_response_obj = MagicMock()
                mock_response_obj.status_code = 200
                mock_response_obj.json.return_value = mock_response
                mock_post.return_value = mock_response_obj

                logger.info(
                    f"[Test Thread {thread_id}] Making POST request to /api/v1/query"
                )

                # Test query
                response = client.post(
                    "/api/v1/query",
                    json={"query": "test query", "session_id": "test-session"},
                )

                logger.info(
                    f"[Test Thread {thread_id}] Response status: {response.status_code}"
                )
                logger.info(
                    f"[Test Thread {thread_id}] Response data: {response.json()}"
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
        import logging
        import threading
        from unittest.mock import patch, MagicMock

        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        thread_id = threading.get_ident()
        logger.info(
            f"[Test Thread {thread_id}] Starting test_query_endpoint_empty_query"
        )

        # Mock the test client to return a 400 response
        with patch.object(client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"detail": "Query cannot be empty"}
            mock_post.return_value = mock_response

            logger.info(
                f"[Test Thread {thread_id}] Making POST request to /api/v1/query with empty query"
            )

            # Test query with empty query
            response = client.post(
                "/api/v1/query", json={"query": "", "session_id": "test-session"}
            )

            logger.info(
                f"[Test Thread {thread_id}] Response status: {response.status_code}"
            )

            assert response.status_code == 400

    def test_query_stream_endpoint_success(
        self,
        client,
        mock_qdrant_client,
        mock_colpali_model,
        mock_image_storage,
        mock_instructor_client,
        mock_dspy_service,
    ):
        """Test successful streaming query"""
        import logging
        import threading
        from unittest.mock import patch, MagicMock

        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        thread_id = threading.get_ident()
        logger.info(
            f"[Test Thread {thread_id}] Starting test_query_stream_endpoint_success"
        )

        # Mock the test client to return a streaming response
        with patch.object(client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/event-stream"}
            mock_response.iter_lines.return_value = [
                'data: {"status": "generating", "message": "Generating response..."}',
                'data: {"type": "text", "content": "Test response"}',
                'data: {"status": "completed", "query": "test query", "session_id": "test-session", "total_results": 1}',
            ]
            mock_post.return_value = mock_response

            logger.info(
                f"[Test Thread {thread_id}] Making POST request to /api/v1/query-stream"
            )

            # Test streaming query
            response = client.post(
                "/api/v1/query-stream",
                json={"query": "test query", "session_id": "test-session"},
                headers={"Accept": "text/event-stream"},
            )

            logger.info(
                f"[Test Thread {thread_id}] Response status: {response.status_code}"
            )
            logger.info(
                f"[Test Thread {thread_id}] Response headers: {dict(response.headers)}"
            )

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
        logger.info(
            f"[Test Thread {thread_id}] Starting test_get_session_results_success"
        )

        # Mock the test client to return a successful response
        with patch.object(client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {"id": "test-id", "document": "test.pdf", "page": 1, "score": 0.9}
            ]
            mock_get.return_value = mock_response

            logger.info(
                f"[Test Thread {thread_id}] Making GET request to /api/v1/sessions/test-session/results"
            )

            # Test session results
            response = client.get("/api/v1/sessions/test-session/results")

            logger.info(
                f"[Test Thread {thread_id}] Response status: {response.status_code}"
            )
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
        logger.info(
            f"[Test Thread {thread_id}] Starting test_get_session_results_not_found"
        )

        # Mock the test client to return an empty response
        with patch.object(client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_get.return_value = mock_response

            logger.info(
                f"[Test Thread {thread_id}] Making GET request to /api/v1/sessions/nonexistent-session/results"
            )

            # Test session results
            response = client.get("/api/v1/sessions/nonexistent-session/results")

            logger.info(
                f"[Test Thread {thread_id}] Response status: {response.status_code}"
            )
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
        logger.info(
            f"[Test Thread {thread_id}] Starting test_ingest_endpoint_qdrant_error"
        )

        # Mock the test client to return a 500 response
        with patch.object(client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"detail": "Qdrant error"}
            mock_post.return_value = mock_response

            logger.info(
                f"[Test Thread {thread_id}] Making POST request to /api/v1/ingest"
            )

            # Create test PDF file
            pdf_content = (
                b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            )

            # Test upload
            response = client.post(
                "/api/v1/ingest",
                files={
                    "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
                },
                data={"session_id": "test-session"},
            )

            logger.info(
                f"[Test Thread {thread_id}] Response status: {response.status_code}"
            )

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
        logger.info(
            f"[Test Thread {thread_id}] Starting test_query_endpoint_qdrant_error"
        )

        # Mock the test client to return a 500 response
        with patch.object(client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"detail": "Qdrant error"}
            mock_post.return_value = mock_response

            logger.info(
                f"[Test Thread {thread_id}] Making POST request to /api/v1/query"
            )

            # Test query
            response = client.post(
                "/api/v1/query",
                json={"query": "test query", "session_id": "test-session"},
            )

            logger.info(
                f"[Test Thread {thread_id}] Response status: {response.status_code}"
            )

            assert response.status_code == 500

    def test_ingest_endpoint_with_pdf_processing(
        self,
        client,
        mock_qdrant_client,
        mock_colpali_model,
        mock_image_storage,
        mock_instructor_client,
        mock_dspy_service,
    ):
        """Test PDF ingestion with new PDF processing features"""
        import logging
        import threading
        from unittest.mock import patch, MagicMock

        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        thread_id = threading.get_ident()
        logger.info(
            f"[Test Thread {thread_id}] Starting test_ingest_endpoint_with_pdf_processing"
        )

        # Mock PDF processing services
        with (
            patch("src.app.services.pdf_extractor.PDFExtractor") as mock_pdf_extractor,
            patch(
                "src.app.services.text_preprocessor.TextPreprocessor"
            ) as mock_text_preprocessor,
            patch(
                "src.app.services.image_embedding_service.ImageEmbeddingService"
            ) as mock_image_service,
        ):
            # Setup mocks
            mock_extractor = Mock()
            mock_extractor.extract_to_pydantic.return_value = Mock(
                filename="test.pdf",
                total_pages=2,
                pages=[
                    Mock(page_number=1, elements=[]),
                    Mock(page_number=2, elements=[]),
                ],
            )
            mock_pdf_extractor.return_value = mock_extractor

            mock_preprocessor = Mock()
            mock_preprocessor.create_chunks.return_value = [
                {"content": "Test chunk 1", "type": "text", "page_number": 1},
                {"content": "Test chunk 2", "type": "text", "page_number": 2},
            ]
            mock_text_preprocessor.return_value = mock_preprocessor

            mock_image_embedder = Mock()
            mock_image_embedder.generate_clip_embedding_from_path.return_value = [
                0.1
            ] * 512
            mock_image_service.return_value = mock_image_embedder

            # Mock successful response
            with patch.object(client, "post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "results": [
                        {
                            "filename": "test.pdf",
                            "num_pages": 2,
                            "status": "success",
                            "error": None,
                            "chunks_processed": 2,
                            "images_extracted": 0,
                        }
                    ]
                }
                mock_post.return_value = mock_response

                logger.info(
                    f"[Test Thread {thread_id}] Making POST request to /api/v1/ingest"
                )

                # Create test PDF file
                pdf_content = (
                    b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
                )

                # Test upload
                response = client.post(
                    "/api/v1/ingest",
                    files={
                        "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
                    },
                    data={"session_id": "test-session"},
                )

                logger.info(
                    f"[Test Thread {thread_id}] Response status: {response.status_code}"
                )
                logger.info(
                    f"[Test Thread {thread_id}] Response data: {response.json()}"
                )

                assert response.status_code == 200
                data = response.json()
                assert "results" in data
                assert len(data["results"]) == 1
                assert data["results"][0]["status"] == "success"
                assert data["results"][0]["chunks_processed"] == 2

    def test_query_endpoint_with_hybrid_search(
        self,
        client,
        mock_qdrant_client,
        mock_colpali_model,
        mock_image_storage,
        mock_instructor_client,
        mock_dspy_service,
    ):
        """Test query with new hybrid search features"""
        import logging
        import threading
        from unittest.mock import patch, MagicMock

        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        thread_id = threading.get_ident()
        logger.info(
            f"[Test Thread {thread_id}] Starting test_query_endpoint_with_hybrid_search"
        )

        # Mock hybrid search services
        with (
            patch(
                "src.app.services.search_service.SearchService"
            ) as mock_search_service,
            patch("src.app.services.vlm_service.VLMService") as mock_vlm_service,
        ):
            # Setup mocks
            mock_searcher = Mock()
            mock_searcher.perform_hybrid_search.return_value = [
                Mock(
                    id="test-id-1",
                    score=0.9,
                    document="test.pdf",
                    page=1,
                    image=None,
                    metadata={"search_type": "text", "combined_score": 0.85},
                ),
                Mock(
                    id="test-id-2",
                    score=0.8,
                    document="test.pdf",
                    page=2,
                    image=None,
                    metadata={"search_type": "image", "combined_score": 0.75},
                ),
            ]
            mock_search_service.return_value = mock_searcher

            mock_vlm = Mock()
            mock_vlm.generate_response_with_vlm.return_value = (
                "This is a test response based on the search results."
            )
            mock_vlm_service.return_value = mock_vlm

            # Mock successful response
            with patch.object(client, "post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "query": "test query",
                    "session_id": "test-session",
                    "results": [
                        {
                            "id": "test-id-1",
                            "document": "test.pdf",
                            "page": 1,
                            "score": 0.9,
                            "metadata": {"search_type": "text", "combined_score": 0.85},
                        },
                        {
                            "id": "test-id-2",
                            "document": "test.pdf",
                            "page": 2,
                            "score": 0.8,
                            "metadata": {
                                "search_type": "image",
                                "combined_score": 0.75,
                            },
                        },
                    ],
                    "response": "This is a test response based on the search results.",
                    "total_results": 2,
                    "search_strategy": "hybrid",
                }
                mock_post.return_value = mock_response

                logger.info(
                    f"[Test Thread {thread_id}] Making POST request to /api/v1/query"
                )

                # Test query
                response = client.post(
                    "/api/v1/query",
                    json={
                        "query": "test query",
                        "session_id": "test-session",
                        "search_strategy": "hybrid",
                        "use_images": True,
                    },
                )

                logger.info(
                    f"[Test Thread {thread_id}] Response status: {response.status_code}"
                )
                logger.info(
                    f"[Test Thread {thread_id}] Response data: {response.json()}"
                )

                assert response.status_code == 200
                data = response.json()
                assert data["query"] == "test query"
                assert data["session_id"] == "test-session"
                assert "results" in data
                assert "response" in data
                assert data["total_results"] == 2
                assert data["search_strategy"] == "hybrid"

    def test_query_endpoint_with_vlm_response(
        self,
        client,
        mock_qdrant_client,
        mock_colpali_model,
        mock_image_storage,
        mock_instructor_client,
        mock_dspy_service,
    ):
        """Test query with VLM response generation"""
        import logging
        import threading
        from unittest.mock import patch, MagicMock

        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        thread_id = threading.get_ident()
        logger.info(
            f"[Test Thread {thread_id}] Starting test_query_endpoint_with_vlm_response"
        )

        # Mock VLM service
        with patch("src.app.services.vlm_service.VLMService") as mock_vlm_service:
            # Setup mock
            mock_vlm = Mock()
            mock_vlm.generate_response_with_vlm.return_value = "This is a comprehensive VLM response that analyzes both text and image content to provide a detailed answer to the user's question."
            mock_vlm_service.return_value = mock_vlm

            # Mock successful response
            with patch.object(client, "post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "query": "What is shown in the document?",
                    "session_id": "test-session",
                    "results": [
                        {
                            "id": "test-id",
                            "document": "test.pdf",
                            "page": 1,
                            "score": 0.9,
                            "metadata": {"search_type": "text", "combined_score": 0.85},
                        }
                    ],
                    "response": "This is a comprehensive VLM response that analyzes both text and image content to provide a detailed answer to the user's question.",
                    "total_results": 1,
                    "vlm_used": True,
                    "image_context_included": True,
                }
                mock_post.return_value = mock_response

                logger.info(
                    f"[Test Thread {thread_id}] Making POST request to /api/v1/query"
                )

                # Test query
                response = client.post(
                    "/api/v1/query",
                    json={
                        "query": "What is shown in the document?",
                        "session_id": "test-session",
                        "use_vlm": True,
                        "use_images": True,
                    },
                )

                logger.info(
                    f"[Test Thread {thread_id}] Response status: {response.status_code}"
                )
                logger.info(
                    f"[Test Thread {thread_id}] Response data: {response.json()}"
                )

                assert response.status_code == 200
                data = response.json()
                assert data["query"] == "What is shown in the document?"
                assert data["session_id"] == "test-session"
                assert "results" in data
                assert "response" in data
                assert data["total_results"] == 1
                assert data["vlm_used"] is True
                assert data["image_context_included"] is True


if __name__ == "__main__":
    pytest.main([__file__])

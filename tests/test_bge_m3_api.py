"""
Tests for BGE-M3 API Endpoints
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from typing import Dict, Any, List, Optional
import io
import tempfile
import os

from src.app.main import app
from src.app.services.bge_m3_service import BGE_M3_Service
from src.app.services.search_service import SearchService
from src.app.utils.qdrant_utils import BGE_M3_QdrantUtils
from src.app.models.schemas import (
    SearchRequest,
    SearchResponse,
    BGE_M3_SearchMode,
    BGE_M3_MultivectorStrategy,
    SearchResult,
    SearchResultItem,
    IngestRequest,
    IngestResponse,
    IngestResult,
    SessionInfo
)
from src.app.settings import Settings, BGE_M3_Settings


@pytest.fixture
def client():
    """Create test client"""
    # Override dependencies with mocks
    app.dependency_overrides = {}
    
    # Mock the dependencies
    with patch('src.app.api.dependencies.get_settings') as mock_settings, \
         patch('src.app.api.dependencies.get_qdrant_client') as mock_qdrant, \
         patch('src.app.api.dependencies.get_image_storage') as mock_image_storage, \
         patch('src.app.api.dependencies.get_search_service') as mock_search_service:
        
        # Configure mocks
        mock_settings.return_value = Settings()
        mock_qdrant_client = Mock()
        # Remove the qdrant attribute that doesn't exist on AsyncQdrantClient
        # mock_qdrant_client.qdrant = mock_qdrant_client  # This line causes the error
        mock_qdrant.return_value = mock_qdrant_client
        mock_image_storage.return_value = Mock()
        mock_search_service.return_value = Mock()
        
        # Add query_points method to mock qdrant client
        mock_qdrant_client.query_points = AsyncMock()
        mock_qdrant_client.query_points.return_value = Mock()
        mock_qdrant_client.query_points.return_value.points = []
        mock_qdrant_client.query_points.return_value.result = []
        
        # Add models attribute to mock qdrant client
        mock_qdrant_client.models = Mock()
        mock_qdrant_client.models.Vector = Mock()
        mock_qdrant_client.models.SparseVector = Mock()
        mock_qdrant_client.models.Filter = Mock()
        mock_qdrant_client.models.FieldCondition = Mock()
        mock_qdrant_client.models.MatchValue = Mock()
        mock_qdrant_client.models.MatchAny = Mock()
        
        # Add get_collections method to mock qdrant client
        mock_qdrant_client.get_collections = AsyncMock()
        mock_qdrant_client.get_collections.return_value = Mock()
        mock_qdrant_client.get_collections.return_value.collections = []
        
        # Add create_collection method to mock qdrant client
        mock_qdrant_client.create_collection = AsyncMock()
        
        # Add get_collection method to mock qdrant client
        mock_qdrant_client.get_collection = AsyncMock()
        mock_qdrant_client.get_collection.return_value = Mock()
        
        # Add get_collection_stats method to mock qdrant client
        mock_qdrant_client.get_collection_stats = AsyncMock()
        mock_qdrant_client.get_collection_stats.return_value = Mock()
        
        # Add optimize method to mock qdrant client
        mock_qdrant_client.optimize = AsyncMock()
        
        # Add delete method to mock qdrant client
        mock_qdrant_client.delete = AsyncMock()
        
        # Add scroll method to mock qdrant client
        mock_qdrant_client.scroll = AsyncMock()
        mock_qdrant_client.scroll.return_value = ([], None)
        
        # Add delete_collection method to mock qdrant client
        mock_qdrant_client.delete_collection = AsyncMock()
        
        # Remove the qdrant attribute that doesn't exist on AsyncQdrantClient
        # mock_qdrant_client.qdrant = mock_qdrant_client  # This line causes the error
        
        yield TestClient(app)
    
    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def mock_settings():
    """Mock settings"""
    return Settings()


@pytest.fixture
def mock_bge_m3_service():
    """Mock BGE-M3 service"""
    with patch('src.app.services.bge_m3_service.BGE_M3_Service') as mock_service:
        mock_instance = Mock(spec=BGE_M3_Service)
        mock_instance.generate_dense_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_instance.generate_sparse_embedding = AsyncMock(return_value={"0": 0.5, "1": 0.3})
        mock_instance.generate_multivector_embedding = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        mock_instance.generate_embeddings = AsyncMock(return_value={
            "embeddings": {
                "dense": [0.1, 0.2, 0.3],
                "sparse": {"0": 0.5, "1": 0.3},
                "multi_vector": [[0.1, 0.2], [0.3, 0.4]]
            },
            "errors": [],
            "text": "Test query"
        })
        mock_instance.health_check = AsyncMock(return_value={
            "status": "healthy",
            "cache_status": "healthy",
            "model_status": "healthy"
        })
        mock_service.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_search_service():
    """Mock search service"""
    with patch('src.app.services.search_service.SearchService') as mock_service:
        mock_instance = Mock(spec=SearchService)
        mock_instance.bge_m3_hybrid_search = AsyncMock(return_value=SearchResponse(
            query="Test query",
            session_id="test-session",
            results=[
                SearchResult(
                    items=[
                        SearchResultItem(
                            id=1,
                            score=0.95,
                            document="test1.pdf",
                            page=1,
                            metadata={"source": "test1.pdf", "page": 1, "type": "text"},
                            search_type="hybrid"
                        )
                    ],
                    total=1,
                    query="Test query"
                )
            ],
            total_results=1,
            search_strategy="hybrid",
            processing_time=0.5,
            cache_hit=False
        ))
        mock_instance.bge_m3_multivector_search = AsyncMock(return_value=SearchResponse(
            query="Test query",
            session_id="test-session",
            results=[
                SearchResult(
                    items=[],
                    total=0,
                    query="Test query"
                )
            ],
            total_results=0,
            search_mode=BGE_M3_SearchMode.MULTIVECTOR,
            processing_time=0.5
        ))
        mock_instance.health_check = AsyncMock(return_value={
            "status": "healthy",
            "bge_m3_service": {"status": "healthy"},
            "qdrant_service": {"status": "healthy"}
        })
        mock_service.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_qdrant_utils():
    """Mock Qdrant utils"""
    with patch('src.app.api.endpoints.ingest.BGE_M3_QdrantUtils') as mock_utils:
        mock_instance = Mock(spec=BGE_M3_QdrantUtils)
        mock_instance.create_collection = AsyncMock(return_value=True)
        mock_instance.hybrid_search = AsyncMock(return_value={
            "results": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "payload": {
                        "content": "Test document 1",
                        "metadata": {"source": "test1.pdf", "page": 1}
                    }
                }
            ]
        })
        mock_instance.prepare_query_embeddings = AsyncMock(return_value={
            "dense": [0.1, 0.2, 0.3],
            "sparse": {"0": 0.5, "1": 0.3},
            "multi_vector": [[0.1, 0.2], [0.3, 0.4]]
        })
        mock_instance.scroll_collection = AsyncMock(return_value=([
            {
                "id": 1,
                "payload": {"content": "Test document 1", "metadata": {"source": "test1.pdf"}}
            }
        ], None))
        mock_instance.health_check = AsyncMock(return_value={"status": "healthy"})
        mock_utils.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_ingest_service():
    """Mock ingest service"""
    with patch('src.app.api.endpoints.ingest.ingest_bge_m3') as mock_service:
        mock_service.return_value = IngestResponse(
            results=[
                IngestResult(
                    filename="test.pdf",
                    num_pages=5,
                    status="success",
                    embeddings_generated={
                        "dense": 10,
                        "sparse": 10,
                        "multi_vector": 10
                    },
                    processing_time=2.5,
                    cache_hits=5,
                    bge_m3_used=True,
                    embedding_types=["dense", "sparse", "multi_vector"]
                )
            ]
        )
        yield mock_service


class TestBGE_M3_APIEndpoints:
    """Test BGE-M3 API endpoints"""


class TestBGE_M3QueryEndpoints:
    """Test BGE-M3 query endpoints"""

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_success(self, client, mock_search_service):
        """Test successful /query-bge-m3 endpoint"""
        # Mock the search service to be available
        mock_search_service.bge_m3_service = Mock()
        mock_search_service.bge_m3_service.generate_embeddings = AsyncMock(return_value={
            "dense": [0.1] * 1024,
            "sparse": {"0": 0.5, "1": 0.3},
            "multi_vector": [[0.1] * 768 for _ in range(10)]
        })
        
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "hybrid",
                "top_k": 10,
                "alpha": 0.5,
                "beta": 0.3,
                "gamma": 0.2
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "session_id" in data
        assert "results" in data
        assert "total_results" in data
        assert "search_mode" in data
        assert "processing_time" in data

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_dense_only(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with dense only mode"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "dense",
                "alpha": 1.0,
                "beta": 0.0,
                "gamma": 0.0
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "dense"

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_sparse_only(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with sparse only mode"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "sparse",
                "alpha": 0.0,
                "beta": 1.0,
                "gamma": 0.0
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "sparse"

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_multivector_only(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with multivector only mode"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "multivector",
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": 1.0
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "multivector"

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_with_filters(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with metadata filters"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "hybrid",
                "metadata_filters": {
                    "must": [
                        {"key": "source", "match": {"value": "test.pdf"}}
                    ]
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_with_score_threshold(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with score threshold"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "hybrid",
                "score_threshold": 0.8
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_with_pagination(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with pagination"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "hybrid",
                "page": 2,
                "page_size": 20
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["page"] == 2
        assert data["page_size"] == 20

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_with_multivector_strategy(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with multivector strategy"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "multivector",
                "multivector_strategy": "max_sim"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_with_cache(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with cache"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "hybrid",
                "use_cache": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_with_normalization(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with normalization"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "hybrid",
                "normalize": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_empty_query(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with empty query"""
        # Mock the search service to be available
        mock_search_service.bge_m3_service = Mock()
        
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "",
                "session_id": "test-session"
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_invalid_mode(self, client):
        """Test /query-bge-m3 endpoint with invalid mode"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "invalid"
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_missing_session_id(self, client):
        """Test /query-bge-m3 endpoint with missing session_id"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query"
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_service_error(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with service error"""
        mock_search_service.bge_m3_hybrid_search = AsyncMock(side_effect=Exception("Service error"))
        
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session"
            }
        )
        
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_query_bge_m3_endpoint_no_results(self, client, mock_search_service):
        """Test /query-bge-m3 endpoint with no results"""
        mock_search_service.bge_m3_hybrid_search = AsyncMock(return_value=SearchResponse(
            query="Test query",
            session_id="test-session",
            results=[
                SearchResult(
                    items=[],
                    total=0,
                    query="Test query"
                )
            ],
            total_results=0,
            search_mode=BGE_M3_SearchMode.HYBRID,
            processing_time=0.5
        ))
        
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 0
        assert data["total_results"] == 0


class TestBGE_M3IngestEndpoints:
    """Test BGE-M3 ingest endpoints"""

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_success(self, client, mock_ingest_service):
        """Test successful /ingest-bge-m3 endpoint"""
        # Create test PDF file
        pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n"
            b"3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n183\n%%EOF"
        )
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": "true",
                "embedding_types": "dense,sparse,multi_vector",
                "include_dense": "true",
                "include_sparse": "true",
                "include_multivector": "true",
                "batch_size": "32",
                "cache_embeddings": "true"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_dense_only(self, client, mock_ingest_service):
        """Test /ingest-bge-m3 endpoint with dense only embeddings"""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": "true",
                "embedding_types": "dense"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_sparse_only(self, client, mock_ingest_service):
        """Test /ingest-bge-m3 endpoint with sparse only embeddings"""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": "true",
                "embedding_types": "sparse"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_multivector_only(self, client, mock_ingest_service):
        """Test /ingest-bge-m3 endpoint with multivector only embeddings"""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": "true",
                "embedding_types": "multi_vector"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_with_batch_processing(self, client, mock_ingest_service):
        """Test /ingest-bge-m3 endpoint with batch processing"""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files=[
                ("files", ("test1.pdf", io.BytesIO(pdf_content), "application/pdf")),
                ("files", ("test2.pdf", io.BytesIO(pdf_content), "application/pdf"))
            ],
            data={
                "session_id": "test-session",
                "use_bge_m3": True,
                "embedding_types": ["dense", "sparse", "multi_vector"],
                "batch_size": 2
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_with_chunking(self, client, mock_ingest_service):
        """Test /ingest-bge-m3 endpoint with chunking"""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": "true",
                "embedding_types": "dense,sparse,multi_vector",
                "chunk_size": "512",
                "chunk_overlap": "50"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_with_metadata(self, client, mock_ingest_service):
        """Test /ingest-bge-m3 endpoint with metadata"""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": True,
                "embedding_types": ["dense", "sparse", "multi_vector"],
                "metadata": '{"author": "Test Author", "title": "Test Document", "subject": "Test Subject"}'
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_without_bge_m3(self, client, mock_ingest_service):
        """Test /ingest-bge-m3 endpoint without BGE-M3"""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_no_file(self, client):
        """Test /ingest-bge-m3 endpoint without file"""
        response = client.post("/api/v1/ingest-bge-m3")
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_invalid_file_type(self, client):
        """Test /ingest-bge-m3 endpoint with invalid file type"""
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.txt", io.BytesIO(b"test content"), "text/plain")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": True
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_service_error(self, client, mock_ingest_service):
        """Test /ingest-bge-m3 endpoint with service error"""
        mock_ingest_service.side_effect = Exception("Service error")
        
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": "true"
            }
        )
        
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_ingest_bge_m3_endpoint_processing_error(self, client, mock_ingest_service):
        """Test /ingest-bge-m3 endpoint with processing error"""
        mock_ingest_service.return_value = IngestResponse(
            results=[
                IngestResult(
                    filename="test.pdf",
                    num_pages=5,
                    status="error",
                    error="Processing failed",
                    embeddings_generated={},
                    processing_time=1.0,
                    cache_hits=0,
                    bge_m3_used=True,
                    embedding_types=[]
                )
            ]
        )
        
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest-bge-m3",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["results"][0]["status"] == "error"


class TestBGE_M3ExtendedEndpoints:
    """Test BGE-M3 extended endpoints"""

    @pytest.mark.asyncio
    async def test_query_endpoint_with_bge_m3(self, client, mock_search_service):
        """Test extended /query endpoint with BGE-M3"""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "hybrid",
                "use_bge_m3": True,
                "embedding_types": ["dense", "sparse", "multi_vector"],
                "alpha": 0.5,
                "beta": 0.3,
                "gamma": 0.2
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "search_mode" in data

    @pytest.mark.asyncio
    async def test_query_endpoint_without_bge_m3(self, client, mock_search_service):
        """Test extended /query endpoint without BGE-M3"""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "use_bge_m3": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_query_endpoint_with_bge_m3_multivector(self, client, mock_search_service):
        """Test extended /query endpoint with BGE-M3 multivector"""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "multivector",
                "use_bge_m3": True,
                "multivector_strategy": "max_sim"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_ingest_endpoint_with_bge_m3(self, client, mock_ingest_service):
        """Test extended /ingest endpoint with BGE-M3"""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": True,
                "embedding_types": ["dense", "sparse", "multi_vector"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_ingest_endpoint_without_bge_m3(self, client, mock_ingest_service):
        """Test extended /ingest endpoint without BGE-M3"""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        response = client.post(
            "/api/v1/ingest",
            files={
                "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
            },
            data={
                "session_id": "test-session",
                "use_bge_m3": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_health_endpoint_with_bge_m3(self, client, mock_search_service):
        """Test /health endpoint with BGE-M3 status"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "bge_m3_service" in data["services"]

    @pytest.mark.asyncio
    async def test_health_endpoint_bge_m3_unhealthy(self, client, mock_search_service):
        """Test /health endpoint with BGE-M3 service unhealthy"""
        mock_search_service.health_check = AsyncMock(return_value={
            "status": "unhealthy",
            "bge_m3_service": {"status": "unhealthy", "error": "Service error"},
            "qdrant_service": {"status": "healthy"}
        })
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["services"]["bge_m3_service"]["status"] == "unhealthy"


class TestBGE_M3SessionEndpoints:
    """Test BGE-M3 session management endpoints"""

    @pytest.mark.asyncio
    async def test_get_session_documents(self, client, mock_search_service):
        """Test /sessions/{session_id}/documents endpoint"""
        mock_search_service.get_session_documents = AsyncMock(return_value=["test1.pdf", "test2.pdf"])
        
        response = client.get("/api/v1/sessions/test-session/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "test1.pdf" in data
        assert "test2.pdf" in data

    @pytest.mark.asyncio
    async def test_get_session_documents_empty(self, client, mock_search_service):
        """Test /sessions/{session_id}/documents endpoint with empty session"""
        mock_search_service.get_session_documents = AsyncMock(return_value=[])
        
        response = client.get("/api/v1/sessions/nonexistent-session/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_get_session_documents_error(self, client, mock_search_service):
        """Test /sessions/{session_id}/documents endpoint with error"""
        mock_search_service.get_session_documents = AsyncMock(side_effect=Exception("Service error"))
        
        response = client.get("/api/v1/sessions/test-session/documents")
        
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_delete_session(self, client, mock_search_service):
        """Test DELETE /sessions/{session_id} endpoint"""
        mock_search_service.delete_session = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/sessions/test-session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, client, mock_search_service):
        """Test DELETE /sessions/{session_id} endpoint with non-existent session"""
        mock_search_service.delete_session = AsyncMock(return_value=False)
        
        response = client.delete("/api/v1/sessions/nonexistent-session")
        
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session_error(self, client, mock_search_service):
        """Test DELETE /sessions/{session_id} endpoint with error"""
        mock_search_service.delete_session = AsyncMock(side_effect=Exception("Service error"))
        
        response = client.delete("/api/v1/sessions/test-session")
        
        assert response.status_code == 500


class TestBGE_M3CollectionEndpoints:
    """Test BGE-M3 collection management endpoints"""

    @pytest.mark.asyncio
    async def test_create_collection(self, client, mock_search_service):
        """Test POST /collections endpoint"""
        mock_search_service.create_collection = AsyncMock(return_value=True)
        
        response = client.post("/api/v1/collections", json={"name": "test_collection"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_create_collection_error(self, client, mock_search_service):
        """Test POST /collections endpoint with error"""
        mock_search_service.create_collection = AsyncMock(return_value=False)
        
        response = client.post("/api/v1/collections", json={"name": "test_collection"})
        
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_delete_collection(self, client, mock_search_service):
        """Test DELETE /collections/{collection_name} endpoint"""
        mock_search_service.delete_collection = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/collections/test_collection")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_delete_collection_not_found(self, client, mock_search_service):
        """Test DELETE /collections/{collection_name} endpoint with non-existent collection"""
        mock_search_service.delete_collection = AsyncMock(return_value=False)
        
        response = client.delete("/api/v1/collections/nonexistent_collection")
        
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, client, mock_search_service):
        """Test GET /collections/{collection_name}/stats endpoint"""
        mock_search_service.get_collection_stats = AsyncMock(return_value={
            "status": "ok",
            "result": {
                "vector_count": 1000,
                "segment_count": 2
            }
        })
        
        response = client.get("/api/v1/collections/test_collection/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["result"]["vector_count"] == 1000

    @pytest.mark.asyncio
    async def test_get_collection_stats_error(self, client, mock_search_service):
        """Test GET /collections/{collection_name}/stats endpoint with error"""
        mock_search_service.get_collection_stats = AsyncMock(side_effect=Exception("Service error"))
        
        response = client.get("/api/v1/collections/test_collection/stats")
        
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_optimize_collection(self, client, mock_search_service):
        """Test POST /collections/{collection_name}/optimize endpoint"""
        mock_search_service.optimize_collection = AsyncMock(return_value=True)
        
        response = client.post("/api/v1/collections/test_collection/optimize")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_optimize_collection_error(self, client, mock_search_service):
        """Test POST /collections/{collection_name}/optimize endpoint with error"""
        mock_search_service.optimize_collection = AsyncMock(return_value=False)
        
        response = client.post("/api/v1/collections/test_collection/optimize")
        
        assert response.status_code == 500


class TestBGE_M3StreamingEndpoints:
    """Test BGE-M3 streaming endpoints"""

    @pytest.mark.asyncio
    async def test_query_stream_endpoint_success(self, client, mock_search_service):
        """Test successful /query-bge-m3/stream endpoint"""
        response = client.post(
            "/api/v1/query-bge-m3/stream",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "search_mode": "hybrid"
            }
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"

    @pytest.mark.asyncio
    async def test_query_stream_endpoint_empty_query(self, client):
        """Test /query-bge-m3/stream endpoint with empty query"""
        response = client.post(
            "/api/v1/query-bge-m3/stream",
            json={
                "query": "",
                "session_id": "test-session"
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_query_stream_endpoint_service_error(self, client, mock_search_service):
        """Test /query-bge-m3/stream endpoint with service error"""
        mock_search_service.bge_m3_hybrid_search = AsyncMock(side_effect=Exception("Service error"))
        
        response = client.post(
            "/api/v1/query-bge-m3/stream",
            json={
                "query": "Test query",
                "session_id": "test-session"
            }
        )
        
        assert response.status_code == 500


class TestBGE_M3Validation:
    """Test BGE-M3 request validation"""

    @pytest.mark.asyncio
    async def test_invalid_embedding_types(self, client):
        """Test validation of invalid embedding types"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "embedding_types": ["invalid_type"]
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_invalid_alpha_beta_gamma(self, client):
        """Test validation of invalid alpha, beta, gamma values"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "alpha": 1.5,  # Invalid value
                "beta": -0.1,  # Invalid value
                "gamma": 0.5
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_invalid_top_k(self, client):
        """Test validation of invalid top_k value"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "top_k": 0  # Invalid value
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_invalid_score_threshold(self, client):
        """Test validation of invalid score threshold"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "score_threshold": 1.5  # Invalid value
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_invalid_pagination(self, client):
        """Test validation of invalid pagination values"""
        response = client.post(
            "/api/v1/query-bge-m3",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "page": 0,  # Invalid value
                "page_size": 0  # Invalid value
            }
        )
        
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])
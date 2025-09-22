"""
Pytest fixtures for BGE-M3 tests
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock

from src.app.services.bge_m3_service import BGE_M3_Service, CacheManager, ErrorHandler
from src.app.services.search_service import SearchService
from src.app.utils.qdrant_utils import BGE_M3_QdrantUtils
from src.app.models.schemas import (
    SearchRequest,
    SearchResponse,
    BGE_M3_SearchMode,
    BGE_M3_MultivectorStrategy,
    SearchResult,
    IngestRequest,
    IngestResponse,
    IngestResult,
    SessionInfo
)
from src.app.settings import Settings, BGE_M3_Settings
from .mocks import (
    MockBGE_M3Service,
    MockQdrantClient,
    MockRedisClient,
    MockSearchService,
    MockIngestService,
    MockQdrantUtils,
    mock_bge_m3_service,
    mock_qdrant_client,
    mock_redis_client,
    mock_search_service,
    mock_ingest_service,
    mock_qdrant_utils,
    mock_settings,
    mock_bge_m3_settings,
    test_documents,
    test_queries,
    test_embeddings
)


# Core fixtures
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for testing"""
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("Test content for BGE-M3 testing")
    yield temp_file


# Service fixtures
@pytest.fixture
def bge_m3_service_factory(mock_settings, mock_bge_m3_settings):
    """Factory for creating BGE-M3 service instances"""
    def create_service():
        with patch('src.app.services.bge_m3_service.redis.Redis.from_url', return_value=MockRedisClient()):
            service = BGE_M3_Service(mock_settings)
            service.model_client = Mock()
            service.model_client.generate_embeddings = AsyncMock(return_value={
                "dense": [0.1] * mock_bge_m3_settings.dense_dimension,
                "sparse": {str(i): 0.5 for i in range(0, 100, 10)},
                "multi_vector": [[0.1] * mock_bge_m3_settings.multi_vector_dimension 
                               for _ in range(mock_bge_m3_settings.multi_vector_count)]
            })
            return service
    return create_service


@pytest.fixture
def search_service_factory(bge_m3_service_factory):
    """Factory for creating SearchService instances"""
    def create_service():
        settings = Mock()
        bge_m3_service = bge_m3_service_factory()
        qdrant_utils = MockQdrantUtils()
        return SearchService(settings, bge_m3_service, qdrant_utils)
    return create_service


@pytest.fixture
def qdrant_utils_factory():
    """Factory for creating QdrantUtils instances"""
    def create_utils(collection_name: str = "test_collection"):
        return MockQdrantUtils(collection_name)
    return create_utils


# Data fixtures
@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "id": "doc1",
            "content": "Dies ist ein Testdokument über Zollbestimmungen und Importregeln in der EU.",
            "metadata": {
                "source": "zollbestimmungen.pdf",
                "page": 1,
                "type": "text",
                "author": "Zollamt",
                "created_at": "2023-01-01",
                "keywords": ["Zoll", "Import", "EU", "Bestimmungen"]
            }
        },
        {
            "id": "doc2",
            "content": "Die Warenklassifikation erfolgt nach dem harmonisierten System und bestimmt die Zollsätze.",
            "metadata": {
                "source": "warenklassifikation.pdf",
                "page": 2,
                "type": "text",
                "author": "EU-Kommission",
                "created_at": "2023-01-02",
                "keywords": ["Klassifikation", "Zollsätze", "Harmonisiertes System"]
            }
        },
        {
            "id": "doc3",
            "content": "Ursprungserklärungen sind erforderlich, um Präferenzzölle in Anspruch nehmen zu können.",
            "metadata": {
                "source": "ursprungserklarungen.pdf",
                "page": 3,
                "type": "text",
                "author": "Bundesfinanzministerium",
                "created_at": "2023-01-03",
                "keywords": ["Ursprung", "Präferenzzölle", "Erklärungen"]
            }
        }
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return [
        "Zollbestimmungen Import EU",
        "Warenklassifikation Zollsätze",
        "Ursprungserklärungen Präferenzzölle",
        "Zollabfertigung Verfahren",
        "Einfuhr Dokumente Anforderungen"
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    return {
        "dense": [0.1] * 1024,
        "sparse": {str(i): 0.5 for i in range(0, 100, 10)},
        "multi_vector": [[0.1] * 768 for _ in range(10)]
    }


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        SearchResult(
            id="doc1",
            document="test1.pdf",
            page=1,
            score=0.95,
            content="Dies ist ein Testdokument über Zollbestimmungen.",
            metadata={"source": "test1.pdf", "page": 1, "type": "text"},
            search_type="hybrid",
            embedding_type="dense"
        ),
        SearchResult(
            id="doc2",
            document="test2.pdf",
            page=2,
            score=0.85,
            content="Dies ist ein weiteres Testdokument.",
            metadata={"source": "test2.pdf", "page": 2, "type": "text"},
            search_type="hybrid",
            embedding_type="sparse"
        )
    ]


@pytest.fixture
def sample_ingest_results():
    """Sample ingest results for testing"""
    return [
        IngestResult(
            filename="test1.pdf",
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
        ),
        IngestResult(
            filename="test2.pdf",
            num_pages=3,
            status="success",
            embeddings_generated={
                "dense": 6,
                "sparse": 6,
                "multi_vector": 6
            },
            processing_time=1.8,
            cache_hits=3,
            bge_m3_used=True,
            embedding_types=["dense", "sparse", "multi_vector"]
        )
    ]


# Configuration fixtures
@pytest.fixture
def test_settings():
    """Test settings configuration"""
    return Settings(
        bge_m3=BGE_M3_Settings(
            model_name="BGE-M3",
            model_device="cpu",
            max_length=8192,
            dense_dimension=1024,
            sparse_dimension=10000,
            multi_vector_count=10,
            multi_vector_dimension=768,
            dense_normalize=True,
            sparse_normalize=True,
            cache_enabled=True,
            cache_redis_url="redis://localhost:6379",
            cache_ttl=3600,
            max_retries=3,
            retry_delay=1.0,
            circuit_breaker_threshold=5
        )
    )


@pytest.fixture
def test_search_request():
    """Test search request"""
    return SearchRequest(
        query="Test query",
        session_id="test-session",
        search_mode=BGE_M3_SearchMode.HYBRID,
        top_k=10,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        use_cache=True,
        normalize=True
    )


@pytest.fixture
def test_ingest_request():
    """Test ingest request"""
    return IngestRequest(
        session_id="test-session",
        use_bge_m3=True,
        embedding_types=["dense", "sparse", "multi_vector"],
        chunk_size=512,
        chunk_overlap=50,
        batch_size=10
    )


# Performance test fixtures
@pytest.fixture
def large_document_set(sample_documents):
    """Large set of documents for performance testing"""
    large_set = []
    for i in range(100):
        for doc in sample_documents:
            large_set.append({
                **doc,
                "id": f"{doc['id']}_{i}",
                "content": f"{doc['content']} - Variante {i}",
                "metadata": {
                    **doc["metadata"],
                    "page": (i % 10) + 1,
                    "created_at": f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
                }
            })
    return large_set


@pytest.fixture
def large_query_set(sample_queries):
    """Large set of queries for performance testing"""
    large_set = []
    for i in range(100):
        for query in sample_queries:
            large_set.append(f"{query} - Variante {i}")
    return large_set


# Error test fixtures
@pytest.fixture
def error_scenarios():
    """Error scenarios for testing"""
    return [
        {
            "name": "empty_query",
            "query": "",
            "expected_error": "ValueError"
        },
        {
            "name": "invalid_search_mode",
            "query": "Test query",
            "search_mode": "invalid",
            "expected_error": "ValueError"
        },
        {
            "name": "invalid_alpha_beta_gamma",
            "query": "Test query",
            "alpha": 1.5,
            "beta": -0.1,
            "gamma": 0.5,
            "expected_error": "ValueError"
        },
        {
            "name": "service_unavailable",
            "query": "Test query",
            "service_error": "Service unavailable",
            "expected_error": "Exception"
        },
        {
            "name": "timeout",
            "query": "Test query",
            "timeout": True,
            "expected_error": "asyncio.TimeoutError"
        }
    ]


# Cache test fixtures
@pytest.fixture
def cache_test_data():
    """Cache test data"""
    return {
        "texts": [
            "Dies ist ein Testtext für Caching-Tests.",
            "Ein weiterer Text mit ähnlichem Inhalt.",
            "Vollständig anderer Textinhalt für Vergleich."
        ],
        "keys": ["test_key_1", "test_key_2", "test_key_3"]
    }


# Integration test fixtures
@pytest.fixture
def integration_test_setup(bge_m3_service_factory, qdrant_utils_factory):
    """Integration test setup"""
    def setup():
        # Create services
        bge_m3_service = bge_m3_service_factory()
        qdrant_utils = qdrant_utils_factory()
        
        # Setup mock data
        test_documents = sample_documents()
        test_queries = sample_queries()
        
        return {
            "bge_m3_service": bge_m3_service,
            "qdrant_utils": qdrant_utils,
            "test_documents": test_documents,
            "test_queries": test_queries
        }
    return setup


# Benchmark fixtures
@pytest.fixture
def benchmark_config():
    """Benchmark configuration"""
    return {
        "iterations": 10,
        "warmup_iterations": 3,
        "document_sizes": [10, 50, 100, 500],
        "query_sizes": [10, 50, 100, 500],
        "batch_sizes": [1, 5, 10, 20]
    }


# Mock service fixtures with specific configurations
@pytest.fixture
def mock_bge_m3_service_with_errors():
    """Mock BGE-M3 service with error scenarios"""
    service = MockBGE_M3Service()
    
    # Configure error scenarios
    service.generate_embeddings = AsyncMock(side_effect=Exception("Service error"))
    service.health_check = AsyncMock(return_value={
        "status": "unhealthy",
        "cache_status": "unhealthy",
        "model_status": "unhealthy",
        "error": "Service unavailable"
    })
    
    return service


@pytest.fixture
def mock_qdrant_client_with_errors():
    """Mock Qdrant client with error scenarios"""
    client = MockQdrantClient()
    
    # Configure error scenarios
    client.create_collection = Mock(side_effect=Exception("Collection creation failed"))
    client.search = Mock(return_value=[])
    client.scroll = Mock(side_effect=Exception("Scroll operation failed"))
    client.delete = Mock(side_effect=Exception("Delete operation failed"))
    
    return client


@pytest.fixture
def mock_redis_client_with_errors():
    """Mock Redis client with error scenarios"""
    client = MockRedisClient()
    
    # Configure error scenarios
    client.ping = Mock(return_value=False)
    client.get = Mock(side_effect=Exception("Redis connection failed"))
    client.setex = Mock(side_effect=Exception("Redis write failed"))
    client.mget = Mock(side_effect=Exception("Redis batch read failed"))
    
    return client


# Test data generators
@pytest.fixture
def document_generator():
    """Document generator for creating test documents"""
    class DocumentGenerator:
        @staticmethod
        def create_document(id: str, content: str, metadata: Dict[str, Any] = None):
            return {
                "id": id,
                "content": content,
                "metadata": metadata or {}
            }
        
        @staticmethod
        def create_document_batch(count: int, base_content: str = "Test content"):
            return [
                DocumentGenerator.create_document(
                    id=f"doc_{i}",
                    content=f"{base_content} - Document {i}",
                    metadata={"page": i % 10 + 1, "type": "text"}
                )
                for i in range(count)
            ]
    
    return DocumentGenerator()


@pytest.fixture
def query_generator():
    """Query generator for creating test queries"""
    class QueryGenerator:
        @staticmethod
        def create_query(base: str, variation: str = ""):
            return f"{base} {variation}".strip()
        
        @staticmethod
        def create_query_batch(count: int, base_queries: List[str]):
            queries = []
            for i in range(count):
                base_query = base_queries[i % len(base_queries)]
                variation = f"query_{i}"
                queries.append(QueryGenerator.create_query(base_query, variation))
            return queries
    
    return QueryGenerator()


# Test scenario fixtures
@pytest.fixture
def test_scenarios():
    """Test scenarios for comprehensive testing"""
    return {
        "unit_tests": {
            "bge_m3_service": [
                "generate_dense_embedding",
                "generate_sparse_embedding", 
                "generate_multivector_embedding",
                "generate_embeddings",
                "batch_generate_embeddings",
                "health_check",
                "error_handling"
            ],
            "qdrant_utils": [
                "create_collection",
                "hybrid_search",
                "prepare_query_embeddings",
                "scroll_collection",
                "health_check"
            ],
            "search_service": [
                "bge_m3_hybrid_search",
                "bge_m3_multivector_search",
                "get_bge_m3_embeddings",
                "health_check",
                "batch_search"
            ]
        },
        "integration_tests": [
            "complete_workflow",
            "workflow_with_caching",
            "error_handling_workflow",
            "performance_workflow"
        ],
        "performance_tests": [
            "batch_processing_performance",
            "cache_performance",
            "search_performance",
            "memory_usage"
        ],
        "error_handling_tests": [
            "service_unavailable_handling",
            "timeout_handling",
            "circuit_breaker_integration",
            "retry_mechanism_integration"
        ]
    }


# Session management fixtures
@pytest.fixture
def test_sessions():
    """Test sessions for testing"""
    return [
        {
            "session_id": "session_1",
            "documents": ["doc1.pdf", "doc2.pdf"],
            "created_at": "2023-01-01T10:00:00",
            "last_accessed": "2023-01-01T10:30:00"
        },
        {
            "session_id": "session_2", 
            "documents": ["doc3.pdf"],
            "created_at": "2023-01-02T14:00:00",
            "last_accessed": "2023-01-02T14:15:00"
        }
    ]


# Collection management fixtures
@pytest.fixture
def test_collections():
    """Test collections for testing"""
    return [
        {
            "name": "zoll_documents",
            "description": "Zoll-related documents",
            "document_count": 100,
            "created_at": "2023-01-01T10:00:00"
        },
        {
            "name": "import_regulations",
            "description": "Import regulations and procedures",
            "document_count": 50,
            "created_at": "2023-01-02T14:00:00"
        }
    ]


# Helper fixtures
@pytest.fixture
def assertion_helpers():
    """Helper functions for assertions"""
    class AssertionHelpers:
        @staticmethod
        def assert_embedding_structure(embedding: Dict[str, Any]):
            """Assert that embedding has correct structure"""
            assert "dense" in embedding
            assert "sparse" in embedding
            assert "multi_vector" in embedding
            assert isinstance(embedding["dense"], list)
            assert isinstance(embedding["sparse"], dict)
            assert isinstance(embedding["multi_vector"], list)
        
        @staticmethod
        def assert_search_response_structure(response: SearchResponse):
            """Assert that search response has correct structure"""
            assert hasattr(response, 'query')
            assert hasattr(response, 'session_id')
            assert hasattr(response, 'results')
            assert hasattr(response, 'total_results')
            assert hasattr(response, 'search_mode')
            assert hasattr(response, 'processing_time')
        
        @staticmethod
        def assert_ingest_response_structure(response: IngestResponse):
            """Assert that ingest response has correct structure"""
            assert hasattr(response, 'results')
            assert isinstance(response.results, list)
            for result in response.results:
                assert hasattr(result, 'filename')
                assert hasattr(result, 'status')
        
        @staticmethod
        def assert_performance_metrics(metrics: Dict[str, float]):
            """Assert that performance metrics are reasonable"""
            assert "time" in metrics
            assert "memory" in metrics
            assert metrics["time"] >= 0
            assert metrics["memory"] >= 0
    
    return AssertionHelpers()


# Cleanup fixtures
@pytest.fixture
def cleanup_temp_files():
    """Fixture for cleaning up temporary files"""
    temp_files = []
    
    def add_temp_file(file_path):
        temp_files.append(file_path)
    
    yield add_temp_file
    
    # Cleanup
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


# Async test helpers
@pytest.fixture
def async_test_helpers():
    """Helper functions for async testing"""
    class AsyncTestHelpers:
        @staticmethod
        async def run_async_test(func, *args, **kwargs):
            """Run async test function"""
            return await func(*args, **kwargs)
        
        @staticmethod
        async def run_async_test_with_timeout(func, timeout: float = 10.0, *args, **kwargs):
            """Run async test function with timeout"""
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                raise Exception(f"Test timed out after {timeout} seconds")
        
        @staticmethod
        async def run_multiple_async_tests(funcs, *args, **kwargs):
            """Run multiple async test functions"""
            tasks = [func(*args, **kwargs) for func in funcs]
            return await asyncio.gather(*tasks)
    
    return AsyncTestHelpers()


# All fixtures combined
@pytest.fixture
def complete_test_setup(
    bge_m3_service_factory,
    qdrant_utils_factory,
    search_service_factory,
    sample_documents,
    sample_queries,
    sample_embeddings,
    test_settings,
    test_search_request,
    test_ingest_request,
    large_document_set,
    large_query_set,
    error_scenarios,
    cache_test_data,
    integration_test_setup,
    benchmark_config,
    mock_bge_m3_service_with_errors,
    mock_qdrant_client_with_errors,
    mock_redis_client_with_errors,
    document_generator,
    query_generator,
    test_scenarios,
    test_sessions,
    test_collections,
    assertion_helpers,
    cleanup_temp_files,
    async_test_helpers
):
    """Complete test setup with all fixtures"""
    return {
        "services": {
            "bge_m3_service": bge_m3_service_factory(),
            "qdrant_utils": qdrant_utils_factory(),
            "search_service": search_service_factory()
        },
        "data": {
            "documents": sample_documents,
            "queries": sample_queries,
            "embeddings": sample_embeddings,
            "large_document_set": large_document_set,
            "large_query_set": large_query_set
        },
        "requests": {
            "search": test_search_request,
            "ingest": test_ingest_request
        },
        "settings": test_settings,
        "test_configs": {
            "error_scenarios": error_scenarios,
            "cache_test_data": cache_test_data,
            "benchmark_config": benchmark_config,
            "scenarios": test_scenarios,
            "sessions": test_sessions,
            "collections": test_collections
        },
        "mock_services": {
            "bge_m3_service_with_errors": mock_bge_m3_service_with_errors,
            "qdrant_client_with_errors": mock_qdrant_client_with_errors,
            "redis_client_with_errors": mock_redis_client_with_errors
        },
        "generators": {
            "document": document_generator,
            "query": query_generator
        },
        "helpers": {
            "assertion": assertion_helpers,
            "async": async_test_helpers
        },
        "cleanup": cleanup_temp_files
    }


if __name__ == "__main__":
    # Test fixtures
    print("Pytest fixtures for BGE-M3 tests created successfully")
    print("Available fixtures:")
    print("- mock_bge_m3_service")
    print("- mock_qdrant_client")
    print("- mock_redis_client")
    print("- mock_search_service")
    print("- mock_ingest_service")
    print("- mock_qdrant_utils")
    print("- sample_documents")
    print("- sample_queries")
    print("- sample_embeddings")
    print("- test_settings")
    print("- complete_test_setup")
"""
Mock objects and fixtures for BGE-M3 tests
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional, Union
import redis
import qdrant_client
from qdrant_client.http import models as http_models
from qdrant_client.qdrant_client_base import QdrantBase

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


class MockBGE_M3Service:
    """Mock BGE-M3 Service for testing"""
    
    def __init__(self):
        self.settings = Mock()
        self.bge_m3_settings = Mock()
        self.cache_manager = Mock()
        self.error_handler = Mock()
        self.model_client = Mock()
        
        # Mock methods
        self.generate_dense_embedding = AsyncMock(return_value=[0.1] * 1024)
        self.generate_sparse_embedding = AsyncMock(return_value={str(i): 0.5 for i in range(0, 100, 10)})
        self.generate_multivector_embedding = AsyncMock(return_value=[[0.1] * 768 for _ in range(10)])
        self.generate_embeddings = AsyncMock(return_value={
            "embeddings": {
                "dense": [0.1] * 1024,
                "sparse": {str(i): 0.5 for i in range(0, 100, 10)},
                "multi_vector": [[0.1] * 768 for _ in range(10)]
            },
            "errors": [],
            "text": "Test query"
        })
        self.batch_generate_embeddings = AsyncMock(return_value=[{
            "embeddings": {"dense": [0.1] * 1024, "sparse": {}, "multi_vector": []},
            "errors": [],
            "text": f"Test document {i}"
        } for i in range(5)])
        self.health_check = AsyncMock(return_value={
            "status": "healthy",
            "cache_status": "healthy",
            "model_status": "healthy"
        })
        
        # Mock error handling
        self.error_handler.handle_errors = lambda func: func
        self.error_handler.handle_embedding_error = AsyncMock(return_value={
            "error": "Test error",
            "fallback": True,
            "embedding": [0.0] * 1024,
            "text": "Test text"
        })
        self.error_handler.is_circuit_open = Mock(return_value=False)
        self.error_handler.reset_circuit = Mock(return_value=True)


class MockQdrantClient:
    """Mock Qdrant client for testing"""
    
    def __init__(self):
        self.collections = {}
        self.points = {}
        
    def create_collection(self, collection_name: str, vectors_config: Dict[str, Any]):
        """Mock create collection"""
        self.collections[collection_name] = {
            "vectors_config": vectors_config,
            "points": []
        }
        return True
    
    def upsert(self, collection_name: str, points: List[Dict[str, Any]]):
        """Mock upsert operation"""
        if collection_name not in self.collections:
            self.create_collection(collection_name, {})
        
        for point in points:
            point_id = point.get("id", len(self.collections[collection_name]["points"]))
            self.collections[collection_name]["points"].append(point)
            self.points[point_id] = point
        
        return True
    
    def search(self, collection_name: str, query_vector: List[float], limit: int = 10):
        """Mock search operation"""
        if collection_name not in self.collections:
            return []
        
        # Simple mock search - return points with random scores
        results = []
        for point in self.collections[collection_name]["points"][:limit]:
            score = sum(a * b for a, b in zip(query_vector, point.get("vector", [0] * len(query_vector))))
            results.append({
                "id": point.get("id"),
                "score": score,
                "payload": point.get("payload", {}),
                "vector": point.get("vector", [])
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def scroll(self, collection_name: str, scroll_filter: Optional[Dict[str, Any]] = None, 
               limit: int = 10, with_payload: bool = True):
        """Mock scroll operation"""
        if collection_name not in self.collections:
            return [], None
        
        points = self.collections[collection_name]["points"][:limit]
        return points, None
    
    def delete(self, collection_name: str, points_selector: Any):
        """Mock delete operation"""
        if collection_name in self.collections:
            self.collections[collection_name]["points"] = []
        return True
    
    def health_check(self):
        """Mock health check"""
        return {"status": "ok"}


class MockRedisClient:
    """Mock Redis client for testing"""
    
    def __init__(self):
        self.data = {}
        self.connected = True
        
    def ping(self):
        """Mock ping"""
        return True
    
    def get(self, key: str):
        """Mock get operation"""
        return self.data.get(key)
    
    def setex(self, key: str, time: int, value: str):
        """Mock setex operation"""
        self.data[key] = value
        return True
    
    def mget(self, keys: List[str]):
        """Mock mget operation"""
        return [self.data.get(key) for key in keys]
    
    def pipeline(self):
        """Mock pipeline"""
        return MockPipeline(self)
    
    def keys(self, pattern: str):
        """Mock keys operation"""
        return list(self.data.keys())
    
    def delete(self, *keys):
        """Mock delete operation"""
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
        return count


class MockPipeline:
    """Mock Redis pipeline"""
    
    def __init__(self, redis_client: MockRedisClient):
        self.redis_client = redis_client
        self.commands = []
    
    def setex(self, key: str, time: int, value: str):
        """Mock pipeline setex"""
        self.commands.append(("setex", key, time, value))
        return self
    
    def execute(self):
        """Mock pipeline execute"""
        for command in self.commands:
            if command[0] == "setex":
                self.redis_client.setex(command[1], command[2], command[3])
        self.commands = []
        return True


class MockSearchService:
    """Mock Search Service for testing"""
    
    def __init__(self):
        self.settings = Mock()
        self.bge_m3_service = MockBGE_M3Service()
        self.qdrant_utils = Mock()
        
        # Mock methods
        self.bge_m3_hybrid_search = AsyncMock(return_value=SearchResponse(
            query="Test query",
            session_id="test-session",
            results=[
                SearchResult(
                    id="doc1",
                    document="test1.pdf",
                    page=1,
                    score=0.95,
                    content="Test document content",
                    metadata={"source": "test1.pdf", "page": 1},
                    search_type="hybrid",
                    embedding_type="dense"
                )
            ],
            total_results=1,
            search_mode=BGE_M3_SearchMode.HYBRID,
            processing_time=0.5,
            cache_hits=0,
            embedding_types=["dense", "sparse", "multi_vector"]
        ))
        
        self.bge_m3_multivector_search = AsyncMock(return_value=SearchResponse(
            query="Test query",
            session_id="test-session",
            results=[],
            total_results=0,
            search_mode=BGE_M3_SearchMode.MULTIVECTOR
        ))
        
        self.get_bge_m3_embeddings = AsyncMock(return_value={
            "dense": [0.1] * 1024,
            "sparse": {str(i): 0.5 for i in range(0, 100, 10)},
            "multi_vector": [[0.1] * 768 for _ in range(10)]
        })
        
        self.health_check = AsyncMock(return_value={
            "status": "healthy",
            "bge_m3_service": {"status": "healthy"},
            "qdrant_service": {"status": "healthy"}
        })
        
        self.batch_search = AsyncMock(return_value=[
            SearchResponse(
                query=f"Test query {i}",
                session_id="test-session",
                results=[],
                total_results=0,
                search_mode=BGE_M3_SearchMode.HYBRID
            ) for i in range(3)
        ])


class MockIngestService:
    """Mock Ingest Service for testing"""
    
    def __init__(self):
        self.settings = Mock()
        
        # Mock methods
        self.ingest_pdf = AsyncMock(return_value=IngestResponse(
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
        ))


class MockQdrantUtils:
    """Mock Qdrant Utils for testing"""
    
    def __init__(self, collection_name: str = "test_collection"):
        self.collection_name = collection_name
        self.client = MockQdrantClient()
        
        # Mock methods
        self.create_collection = AsyncMock(return_value=True)
        self.hybrid_search = AsyncMock(return_value={
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
        self.prepare_query_embeddings = AsyncMock(return_value={
            "dense": [0.1] * 1024,
            "sparse": {str(i): 0.5 for i in range(0, 100, 10)},
            "multi_vector": [[0.1] * 768 for _ in range(10)]
        })
        self.scroll_collection = AsyncMock(return_value=([
            {"id": 1, "payload": {"content": "Test document 1"}},
            {"id": 2, "payload": {"content": "Test document 2"}}
        ], None))
        self.health_check = AsyncMock(return_value={"status": "healthy"})
        self.upsert_points = AsyncMock(return_value=True)


# Pytest fixtures
@pytest.fixture
def mock_bge_m3_service():
    """Mock BGE-M3 service fixture"""
    return MockBGE_M3Service()


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client fixture"""
    return MockQdrantClient()


@pytest.fixture
def mock_redis_client():
    """Mock Redis client fixture"""
    return MockRedisClient()


@pytest.fixture
def mock_search_service():
    """Mock Search Service fixture"""
    return MockSearchService()


@pytest.fixture
def mock_ingest_service():
    """Mock Ingest Service fixture"""
    return MockIngestService()


@pytest.fixture
def mock_qdrant_utils():
    """Mock Qdrant Utils fixture"""
    return MockQdrantUtils()


@pytest.fixture
def mock_settings():
    """Mock settings fixture"""
    settings = Mock()
    settings.bge_m3 = Mock()
    settings.bge_m3.model_name = "BGE-M3"
    settings.bge_m3.model_device = "cpu"
    settings.bge_m3.max_length = 8192
    settings.bge_m3.dense_dimension = 1024
    settings.bge_m3.sparse_dimension = 10000
    settings.bge_m3.multi_vector_count = 10
    settings.bge_m3.multi_vector_dimension = 768
    settings.bge_m3.dense_normalize = True
    settings.bge_m3.sparse_normalize = True
    settings.bge_m3.cache_enabled = True
    settings.bge_m3.cache_redis_url = "redis://localhost:6379"
    settings.bge_m3.cache_ttl = 3600
    settings.bge_m3.max_retries = 3
    settings.bge_m3.retry_delay = 1.0
    settings.bge_m3.circuit_breaker_threshold = 5
    return settings


@pytest.fixture
def mock_bge_m3_settings():
    """Mock BGE-M3 settings fixture"""
    return BGE_M3_Settings(
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


@pytest.fixture
def test_documents():
    """Test documents fixture"""
    return [
        {
            "id": "doc1",
            "content": "Dies ist ein Testdokument über Zollbestimmungen und Importregeln.",
            "metadata": {
                "source": "zollbestimmungen.pdf",
                "page": 1,
                "type": "text",
                "author": "Zollamt",
                "created_at": "2023-01-01"
            }
        },
        {
            "id": "doc2",
            "content": "Import von Waren in die EU erfordert bestimmte Dokumente und Zollformalitäten.",
            "metadata": {
                "source": "import_regeln.pdf",
                "page": 2,
                "type": "text",
                "author": "EU-Kommission",
                "created_at": "2023-01-02"
            }
        }
    ]


@pytest.fixture
def test_queries():
    """Test queries fixture"""
    return [
        "Zollbestimmungen Import",
        "Dokumente für EU Import",
        "Zolltarif Warenklassifikation"
    ]


@pytest.fixture
def test_embeddings():
    """Test embeddings fixture"""
    return {
        "dense": [0.1] * 1024,
        "sparse": {str(i): 0.5 for i in range(0, 100, 10)},
        "multi_vector": [[0.1] * 768 for _ in range(10)]
    }


# Context managers for mocking
class MockContextManager:
    """Context manager for mocking"""
    
    def __init__(self, mock_obj):
        self.mock_obj = mock_obj
    
    def __enter__(self):
        return self.mock_obj
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def mock_qdrant_client_context():
    """Context manager for mocking Qdrant client"""
    return MockContextManager(MockQdrantClient())


def mock_redis_client_context():
    """Context manager for mocking Redis client"""
    return MockContextManager(MockRedisClient())


def mock_bge_m3_service_context():
    """Context manager for mocking BGE-M3 service"""
    return MockContextManager(MockBGE_M3Service())


# Patch decorators for common mocking patterns
def mock_bge_m3_service_patch():
    """Patch decorator for mocking BGE-M3 service"""
    return patch('src.app.services.bge_m3_service.BGE_M3_Service', MockBGE_M3Service)


def mock_qdrant_client_patch():
    """Patch decorator for mocking Qdrant client"""
    return patch('qdrant_client.QdrantClient', MockQdrantClient)


def mock_redis_client_patch():
    """Patch decorator for mocking Redis client"""
    return patch('redis.Redis', MockRedisClient)


def mock_search_service_patch():
    """Patch decorator for mocking Search Service"""
    return patch('src.app.services.search_service.SearchService', MockSearchService)


def mock_ingest_service_patch():
    """Patch decorator for mocking Ingest Service"""
    return patch('src.app.api.endpoints.ingest.IngestService', MockIngestService)


def mock_qdrant_utils_patch():
    """Patch decorator for mocking Qdrant Utils"""
    return patch('src.app.utils.qdrant_utils.BGE_M3_QdrantUtils', MockQdrantUtils)


# Helper functions for test setup
def setup_mock_bge_m3_service(service: BGE_M3_Service):
    """Setup mock BGE-M3 service"""
    service.model_client = Mock()
    service.model_client.generate_embeddings = AsyncMock(return_value={
        "dense": [0.1] * 1024,
        "sparse": {str(i): 0.5 for i in range(0, 100, 10)},
        "multi_vector": [[0.1] * 768 for _ in range(10)]
    })
    service.cache_manager = Mock()
    service.cache_manager.get_embedding = AsyncMock(return_value=None)
    service.cache_manager.set_embedding = AsyncMock(return_value=True)
    service.error_handler = Mock()
    service.error_handler.handle_errors = lambda func: func


def setup_mock_qdrant_client(client: QdrantBase):
    """Setup mock Qdrant client"""
    client.create_collection = Mock(return_value=True)
    client.upsert = Mock(return_value=True)
    client.search = Mock(return_value=[])
    client.scroll = Mock(return_value=([], None))
    client.delete = Mock(return_value=True)
    client.health_check = Mock(return_value={"status": "ok"})


def setup_mock_redis_client(client: redis.Redis):
    """Setup mock Redis client"""
    client.ping = Mock(return_value=True)
    client.get = Mock(return_value=None)
    client.setex = Mock(return_value=True)
    client.mget = Mock(return_value=[None])
    client.pipeline = Mock(return_value=MockPipeline(MockRedisClient()))
    client.keys = Mock(return_value=[])
    client.delete = Mock(return_value=0)


# Performance test helpers
class PerformanceTestHelper:
    """Helper class for performance testing"""
    
    @staticmethod
    def measure_time(func, *args, **kwargs):
        """Measure execution time of a function"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def measure_memory_usage(func, *args, **kwargs):
        """Measure memory usage of a function"""
        import tracemalloc
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, current, peak
    
    @staticmethod
    async def measure_async_time(func, *args, **kwargs):
        """Measure execution time of an async function"""
        import time
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def generate_test_data(size: int):
        """Generate test data of specified size"""
        return {
            "documents": [{"id": f"doc_{i}", "content": f"Test content {i}"} for i in range(size)],
            "queries": [f"Test query {i}" for i in range(size)]
        }


# Benchmark helpers
class BenchmarkHelper:
    """Helper class for benchmarking"""
    
    @staticmethod
    def run_benchmark(func, iterations: int = 10, *args, **kwargs):
        """Run benchmark function multiple times"""
        import time
        times = []
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
            results.append(result)
        
        return {
            "times": times,
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "results": results
        }
    
    @staticmethod
    async def run_async_benchmark(func, iterations: int = 10, *args, **kwargs):
        """Run async benchmark function multiple times"""
        import time
        times = []
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
            results.append(result)
        
        return {
            "times": times,
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "results": results
        }


if __name__ == "__main__":
    # Test mock objects
    mock_service = MockBGE_M3Service()
    mock_qdrant = MockQdrantClient()
    mock_redis = MockRedisClient()
    
    print("Mock objects created successfully")
    print(f"BGE-M3 Service methods: {[method for method in dir(mock_service) if not method.startswith('_')]}")
    print(f"Qdrant Client methods: {[method for method in dir(mock_qdrant) if not method.startswith('_')]}")
    print(f"Redis Client methods: {[method for method in dir(mock_redis) if not method.startswith('_')]}")
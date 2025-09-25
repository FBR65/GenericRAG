"""
Tests for BGE-M3 Service
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

from src.app.services.bge_m3_service import BGE_M3_Service, CacheManager, ErrorHandler
from src.app.settings import Settings, BGE_M3_Settings


class TestCacheManager:
    """Test Cache Manager functionality"""

    @pytest.fixture
    def mock_settings(self):
        """Mock BGE-M3 settings"""
        return BGE_M3_Settings(
            cache_enabled=True,
            cache_redis_url="redis://localhost:6379",
            cache_ttl=3600
        )

    @pytest.fixture
    def cache_manager(self, mock_settings):
        """Create CacheManager instance"""
        with patch('redis.Redis') as mock_redis:
            mock_redis_client = Mock()
            mock_redis.from_url.return_value = mock_redis_client
            mock_redis_client.ping.return_value = True
            return CacheManager(mock_settings)

    def test_cache_manager_initialization(self, mock_settings):
        """Test CacheManager initialization"""
        with patch('redis.Redis') as mock_redis:
            mock_redis_client = Mock()
            mock_redis.from_url.return_value = mock_redis_client
            mock_redis_client.ping.return_value = True
            
            cache_manager = CacheManager(mock_settings)
            assert cache_manager.settings == mock_settings
            assert cache_manager.redis_client is not None

    def test_cache_manager_initialization_failure(self, mock_settings):
        """Test CacheManager initialization with Redis failure"""
        with patch('redis.Redis') as mock_redis:
            mock_redis.from_url.side_effect = Exception("Connection failed")
            
            cache_manager = CacheManager(mock_settings)
            assert cache_manager.redis_client is None

    def test_generate_cache_key(self, cache_manager):
        """Test cache key generation"""
        text = "Test text for embedding"
        mode = "dense"
        
        key = cache_manager._generate_cache_key(text, mode)
        # Calculate expected hash
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        expected_key = f"bge_m3:{mode}:{text_hash}"
        
        assert key == expected_key

    @pytest.mark.asyncio
    async def test_get_embedding_success(self, cache_manager):
        """Test successful cache retrieval"""
        cache_key = "test_key"
        cached_data = {"dense": [0.1, 0.2, 0.3]}
        
        cache_manager.redis_client.get.return_value = '{"dense": [0.1, 0.2, 0.3]}'
        
        result = await cache_manager.get_embedding(cache_key)
        
        assert result == cached_data
        cache_manager.redis_client.get.assert_called_once_with(cache_key)

    @pytest.mark.asyncio
    async def test_get_embedding_no_cache(self, cache_manager):
        """Test cache retrieval when no cache exists"""
        cache_key = "nonexistent_key"
        
        cache_manager.redis_client.get.return_value = None
        
        result = await cache_manager.get_embedding(cache_key)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_embedding_exception(self, cache_manager):
        """Test cache retrieval with exception"""
        cache_key = "test_key"
        
        cache_manager.redis_client.get.side_effect = Exception("Redis error")
        
        result = await cache_manager.get_embedding(cache_key)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_set_embedding_success(self, cache_manager):
        """Test successful cache storage"""
        cache_key = "test_key"
        value = {"dense": [0.1, 0.2, 0.3]}
        
        result = await cache_manager.set_embedding(cache_key, value)
        
        assert result is True
        cache_manager.redis_client.setex.assert_called_once()
        args = cache_manager.redis_client.setex.call_args
        assert args[0][0] == cache_key
        assert args[0][1] == cache_manager.settings.cache_ttl
        assert args[0][2] == '{"dense": [0.1, 0.2, 0.3]}'

    @pytest.mark.asyncio
    async def test_set_embedding_no_redis(self, mock_settings):
        """Test cache storage when Redis is not available"""
        with patch('redis.Redis') as mock_redis:
            mock_redis.from_url.side_effect = Exception("Connection failed")
            cache_manager = CacheManager(mock_settings)
            
            cache_key = "test_key"
            value = {"dense": [0.1, 0.2, 0.3]}
            
            result = await cache_manager.set_embedding(cache_key, value)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_batch_get_embeddings(self, cache_manager):
        """Test batch cache retrieval"""
        keys = ["key1", "key2", "key3"]
        cached_values = ['{"dense": [0.1]}', '{"dense": [0.2]}', None]
        
        cache_manager.redis_client.mget.return_value = cached_values
        
        results = await cache_manager.batch_get_embeddings(keys)
        
        assert len(results) == 3
        assert results[0] == {"dense": [0.1]}
        assert results[1] == {"dense": [0.2]}
        assert results[2] is None

    @pytest.mark.asyncio
    async def test_batch_set_embeddings(self, cache_manager):
        """Test batch cache storage"""
        key_values = {
            "key1": {"dense": [0.1]},
            "key2": {"dense": [0.2]}
        }
        
        result = await cache_manager.batch_set_embeddings(key_values)
        
        assert result is True
        cache_manager.redis_client.pipeline.assert_called_once()
        pipeline = cache_manager.redis_client.pipeline.return_value
        assert pipeline.setex.call_count == 2
        assert pipeline.execute.called

    @pytest.mark.asyncio
    async def test_clear_cache(self, cache_manager):
        """Test cache clearing"""
        cache_manager.redis_client.keys.return_value = ["key1", "key2"]
        cache_manager.redis_client.delete.return_value = 2
        
        result = await cache_manager.clear_cache("test_pattern")
        
        assert result == 2
        cache_manager.redis_client.keys.assert_called_once_with("test_pattern")
        cache_manager.redis_client.delete.assert_called_once_with("key1", "key2")

    @pytest.mark.asyncio
    async def test_clear_cache_no_keys(self, cache_manager):
        """Test cache clearing when no keys match"""
        cache_manager.redis_client.keys.return_value = []
        
        result = await cache_manager.clear_cache("test_pattern")
        
        assert result == 0


class TestErrorHandler:
    """Test Error Handler functionality"""

    @pytest.fixture
    def mock_settings(self):
        """Mock BGE-M3 settings"""
        return BGE_M3_Settings(
            max_retries=3,
            retry_delay=1.0,
            circuit_breaker_threshold=3
        )

    @pytest.fixture
    def error_handler(self, mock_settings):
        """Create ErrorHandler instance"""
        return ErrorHandler(mock_settings)

    def test_error_handler_initialization(self, mock_settings):
        """Test ErrorHandler initialization"""
        error_handler = ErrorHandler(mock_settings)
        assert error_handler.settings == mock_settings
        assert error_handler.circuit_open is False
        assert error_handler.failure_count == 0
        assert error_handler.last_failure_time == 0

    def test_handle_errors_decorator_success(self, error_handler):
        """Test error handling decorator with successful execution"""
        @error_handler.handle_errors
        async def test_function():
            return "success"
        
        result = asyncio.run(test_function())
        assert result == "success"
        assert error_handler.failure_count == 0
        assert error_handler.circuit_open is False

    def test_handle_errors_decorator_retry_success(self, error_handler):
        """Test error handling decorator with retry success"""
        call_count = 0
        
        @error_handler.handle_errors
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return "success"
        
        result = asyncio.run(test_function())
        assert result == "success"
        assert error_handler.failure_count == 0
        assert error_handler.circuit_open is False

    def test_handle_errors_decorator_failure(self, error_handler):
        """Test error handling decorator with all attempts failing"""
        @error_handler.handle_errors
        async def test_function():
            raise Exception("Persistent error")
        
        with pytest.raises(Exception, match="Persistent error"):
            asyncio.run(test_function())
        
        assert error_handler.failure_count == 3
        assert error_handler.circuit_open is True

    def test_handle_errors_decorator_circuit_breaker_open(self, error_handler):
        """Test error handling decorator when circuit breaker is open"""
        error_handler.circuit_open = True
        
        @error_handler.handle_errors
        async def test_function():
            return "should not reach here"
        
        with pytest.raises(Exception, match="Circuit breaker is open"):
            asyncio.run(test_function())

    def test_handle_embedding_error(self, error_handler):
        """Test embedding error handling"""
        error = Exception("Model error")
        text = "Test text"
        mode = "dense"
        
        result = asyncio.run(error_handler.handle_embedding_error(error, text, mode))
        
        assert result["error"] == "Model error"
        assert result["fallback"] is True
        assert result["text"] == text
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) == error_handler.settings.dense_dimension

    def test_get_fallback_embedding_dense(self, error_handler):
        """Test fallback embedding generation for dense mode"""
        embedding = error_handler._get_fallback_embedding("dense")
        
        assert isinstance(embedding, list)
        assert len(embedding) == error_handler.settings.dense_dimension
        assert all(x == 0.0 for x in embedding)

    def test_get_fallback_embedding_sparse(self, error_handler):
        """Test fallback embedding generation for sparse mode"""
        embedding = error_handler._get_fallback_embedding("sparse")
        
        assert isinstance(embedding, dict)
        assert embedding == {}

    def test_get_fallback_embedding_multivector(self, error_handler):
        """Test fallback embedding generation for multivector mode"""
        embedding = error_handler._get_fallback_embedding("multi_vector")
        
        assert isinstance(embedding, list)
        assert len(embedding) == error_handler.settings.multi_vector_count
        for vector in embedding:
            assert len(vector) == error_handler.settings.multi_vector_dimension
            assert all(x == 0.0 for x in vector)

    def test_get_fallback_embedding_unknown(self, error_handler):
        """Test fallback embedding generation for unknown mode"""
        embedding = error_handler._get_fallback_embedding("unknown")
        
        assert embedding == []

    def test_is_circuit_open_false(self, error_handler):
        """Test circuit breaker status when closed"""
        result = error_handler.is_circuit_open()
        assert result is False

    def test_is_circuit_open_true(self, error_handler):
        """Test circuit breaker status when open"""
        error_handler.circuit_open = True
        error_handler.last_failure_time = time.time() - 30  # 30 seconds ago
        
        result = error_handler.is_circuit_open()
        assert result is True

    def test_is_circuit_open_cooldown_expired(self, error_handler):
        """Test circuit breaker status when cooldown has expired"""
        error_handler.circuit_open = True
        error_handler.last_failure_time = time.time() - 120  # 2 minutes ago
        
        result = error_handler.is_circuit_open()
        assert result is False
        assert error_handler.circuit_open is False
        assert error_handler.failure_count == 0

    def test_reset_circuit(self, error_handler):
        """Test circuit breaker reset"""
        error_handler.circuit_open = True
        error_handler.failure_count = 5
        
        result = error_handler.reset_circuit()
        
        assert result is True
        assert error_handler.circuit_open is False
        assert error_handler.failure_count == 0


class TestBGE_M3_Service:
    """Test BGE-M3 Service functionality"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings"""
        return Settings()

    @pytest.fixture
    def bge_m3_service(self, mock_settings):
        """Create BGE-M3 Service instance"""
        with patch('src.app.services.bge_m3_service.BGE_M3_Service.__init__', return_value=None):
            service = BGE_M3_Service.__new__(BGE_M3_Service)
            service.settings = mock_settings
            service.bge_m3_settings = mock_settings.bge_m3
            service.cache_manager = Mock()
            service.error_handler = Mock()
            service.model_client = Mock()
            return service

    def test_bge_m3_service_initialization(self, mock_settings):
        """Test BGE-M3 Service initialization"""
        with patch('src.app.services.bge_m3_service.BGE_M3_Service.__init__', return_value=None):
            service = BGE_M3_Service.__new__(BGE_M3_Service)
            service.settings = mock_settings
            service.bge_m3_settings = mock_settings.bge_m3
            service.cache_manager = Mock()
            service.error_handler = Mock()
            service.model_client = Mock()
            
            assert service.settings == mock_settings
            assert service.bge_m3_settings == mock_settings.bge_m3
            assert service.cache_manager is not None
            assert service.error_handler is not None
            assert service.model_client is not None

    def test_validate_text_success(self, bge_m3_service):
        """Test text validation with valid input"""
        text = "  This is a test text with extra spaces  "
        
        result = bge_m3_service._validate_text(text)
        
        assert result == "This is a test text with extra spaces"

    def test_validate_text_empty(self, bge_m3_service):
        """Test text validation with empty input"""
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            bge_m3_service._validate_text("")

    def test_validate_text_none(self, bge_m3_service):
        """Test text validation with None input"""
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            bge_m3_service._validate_text(None)

    def test_validate_text_not_string(self, bge_m3_service):
        """Test text validation with non-string input"""
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            bge_m3_service._validate_text(123)

    def test_validate_text_long_text(self, bge_m3_service):
        """Test text validation with long text"""
        long_text = "x" * 10000
        expected_text = "x" * 8192
        
        result = bge_m3_service._validate_text(long_text)
        
        assert len(result) == 8192
        assert result == expected_text

    @pytest.mark.asyncio
    async def test_make_embedding_request_dense(self, bge_m3_service):
        """Test embedding request generation for dense mode"""
        text = "Test text"
        mode = "dense"
        
        result = await bge_m3_service._make_embedding_request(text, mode)
        
        assert "dense" in result
        assert isinstance(result["dense"], list)
        assert len(result["dense"]) == bge_m3_service.bge_m3_settings.dense_dimension

    @pytest.mark.asyncio
    async def test_make_embedding_request_sparse(self, bge_m3_service):
        """Test embedding request generation for sparse mode"""
        text = "Test text"
        mode = "sparse"
        
        result = await bge_m3_service._make_embedding_request(text, mode)
        
        assert "sparse" in result
        assert isinstance(result["sparse"], dict)
        assert len(result["sparse"]) > 0

    @pytest.mark.asyncio
    async def test_make_embedding_request_multivector(self, bge_m3_service):
        """Test embedding request generation for multivector mode"""
        text = "Test text"
        mode = "multi_vector"
        
        result = await bge_m3_service._make_embedding_request(text, mode)
        
        assert "multi_vector" in result
        assert isinstance(result["multi_vector"], list)
        assert len(result["multi_vector"]) == bge_m3_service.bge_m3_settings.multi_vector_count
        for vector in result["multi_vector"]:
            assert len(vector) == bge_m3_service.bge_m3_settings.multi_vector_dimension

    @pytest.mark.asyncio
    async def test_make_embedding_request_unknown_mode(self, bge_m3_service):
        """Test embedding request generation with unknown mode"""
        text = "Test text"
        mode = "unknown"
        
        with pytest.raises(ValueError, match="Unknown embedding mode: unknown"):
            await bge_m3_service._make_embedding_request(text, mode)

    @pytest.mark.asyncio
    async def test_generate_dense_embedding_success(self, bge_m3_service):
        """Test successful dense embedding generation"""
        text = "Test text for dense embedding"
        
        result = await bge_m3_service.generate_dense_embedding(text)
        
        assert isinstance(result, list)
        assert len(result) == bge_m3_service.bge_m3_settings.dense_dimension

    @pytest.mark.asyncio
    async def test_generate_dense_embedding_cache_hit(self, bge_m3_service):
        """Test dense embedding generation with cache hit"""
        text = "Test text for dense embedding"
        
        # Mock cache hit
        bge_m3_service.cache_manager.get_embedding = AsyncMock(return_value={"dense": [0.1, 0.2, 0.3]})
        
        result = await bge_m3_service.generate_dense_embedding(text)
        
        assert result == [0.1, 0.2, 0.3]
        bge_m3_service.cache_manager.get_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_dense_embedding_error_handling(self, bge_m3_service):
        """Test dense embedding generation with error handling"""
        text = "Test text"
        
        # Mock embedding request to raise error
        bge_m3_service._make_embedding_request = AsyncMock(side_effect=Exception("Model error"))
        
        result = await bge_m3_service.generate_dense_embedding(text)
        
        assert "error" in result
        assert result["fallback"] is True
        assert isinstance(result["embedding"], list)

    @pytest.mark.asyncio
    async def test_generate_sparse_embedding_success(self, bge_m3_service):
        """Test successful sparse embedding generation"""
        text = "Test text for sparse embedding"
        
        result = await bge_m3_service.generate_sparse_embedding(text)
        
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_multivector_embedding_success(self, bge_m3_service):
        """Test successful multivector embedding generation"""
        text = "Test text for multivector embedding"
        
        result = await bge_m3_service.generate_multivector_embedding(text)
        
        assert isinstance(result, list)
        assert len(result) == bge_m3_service.bge_m3_settings.multi_vector_count
        for vector in result:
            assert len(vector) == bge_m3_service.bge_m3_settings.multi_vector_dimension

    @pytest.mark.asyncio
    async def test_generate_embeddings_all_types(self, bge_m3_service):
        """Test generation of all embedding types"""
        text = "Test text for all embeddings"
        
        result = await bge_m3_service.generate_embeddings(text)
        
        assert "embeddings" in result
        assert "errors" in result
        assert "text" in result
        
        embeddings = result["embeddings"]
        assert "dense" in embeddings
        assert "sparse" in embeddings
        assert "multi_vector" in embeddings
        
        assert isinstance(embeddings["dense"], list)
        assert isinstance(embeddings["sparse"], dict)
        assert isinstance(embeddings["multi_vector"], list)

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_errors(self, bge_m3_service):
        """Test embedding generation with some errors"""
        text = "Test text"
        
        # Mock one of the embedding methods to raise an error
        original_sparse = bge_m3_service.generate_sparse_embedding
        bge_m3_service.generate_sparse_embedding = AsyncMock(side_effect=Exception("Sparse error"))
        
        result = await bge_m3_service.generate_embeddings(text)
        
        assert len(result["errors"]) == 1
        assert result["errors"][0][0] == "sparse"
        assert "Sparse error" in result["errors"][0][1]
        
        # Restore original method
        bge_m3_service.generate_sparse_embedding = original_sparse

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings(self, bge_m3_service):
        """Test batch embedding generation"""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        result = await bge_m3_service.batch_generate_embeddings(texts)
        
        assert len(result) == 3
        for i, text_result in enumerate(result):
            assert text_result["text"] == texts[i]
            assert "embeddings" in text_result
            assert "errors" in text_result

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_empty(self, bge_m3_service):
        """Test batch embedding generation with empty input"""
        texts = []
        
        result = await bge_m3_service.batch_generate_embeddings(texts)
        
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_with_errors(self, bge_m3_service):
        """Test batch embedding generation with errors"""
        texts = ["Text 1", "Text 2"]
        
        # Mock one of the texts to cause an error
        original_generate = bge_m3_service.generate_embeddings
        bge_m3_service.generate_embeddings = AsyncMock(side_effect=Exception("Batch error"))
        
        result = await bge_m3_service.batch_generate_embeddings(texts)
        
        assert len(result) == 2
        for text_result in result:
            assert len(text_result["errors"]) > 0
            assert "Batch error" in text_result["errors"][0][1]
        
        # Restore original method
        bge_m3_service.generate_embeddings = original_generate

    @pytest.mark.asyncio
    async def test_cache_embeddings_all_types(self, bge_m3_service):
        """Test caching of all embedding types"""
        texts = ["Text 1", "Text 2"]
        
        # Mock successful embedding generation
        bge_m3_service.generate_embeddings = AsyncMock(return_value={
            "embeddings": {"dense": [0.1], "sparse": {"0": 0.5}, "multi_vector": [[0.1]]},
            "errors": [],
            "text": "Text 1"
        })
        
        result = await bge_m3_service.cache_embeddings(texts)
        
        assert result is True
        bge_m3_service.cache_manager.batch_set_embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_embeddings_empty(self, bge_m3_service):
        """Test caching with empty input"""
        texts = []
        
        result = await bge_m3_service.cache_embeddings(texts)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_success(self, bge_m3_service):
        """Test health check with all services healthy"""
        # Mock successful Redis connection
        bge_m3_service.cache_manager.redis_client = Mock()
        bge_m3_service.cache_manager.redis_client.ping.return_value = True
        bge_m3_service.cache_manager.redis_client.info.return_value = {
            "used_memory_human": "1.5M",
            "keyspace_hits": 100,
            "keyspace_misses": 50,
            "total_commands_processed": 1000
        }
        
        result = await bge_m3_service.health_check()
        
        assert result["status"] == "healthy"
        assert result["cache_available"] is True
        assert result["model_available"] is True

    @pytest.mark.asyncio
    async def test_health_check_cache_unavailable(self, bge_m3_service):
        """Test health check with cache unavailable"""
        bge_m3_service.cache_manager.redis_client = None
        
        result = await bge_m3_service.health_check()
        
        assert result["status"] == "degraded"
        assert result["cache_available"] is False
        assert result["model_available"] is True

    @pytest.mark.asyncio
    async def test_health_check_model_unavailable(self, bge_m3_service):
        """Test health check with model unavailable"""
        bge_m3_service.model_client = None
        
        result = await bge_m3_service.health_check()
        
        assert result["status"] == "unhealthy"
        assert result["cache_available"] is True
        assert result["model_available"] is False

    @pytest.mark.asyncio
    async def test_health_check_all_unavailable(self, bge_m3_service):
        """Test health check with all services unavailable"""
        bge_m3_service.cache_manager.redis_client = None
        bge_m3_service.model_client = None
        
        result = await bge_m3_service.health_check()
        
        assert result["status"] == "unhealthy"
        assert result["cache_available"] is False
        assert result["model_available"] is False

    @pytest.mark.asyncio
    async def test_process_batch_with_performance(self, bge_m3_service):
        """Test batch processing with performance metrics"""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Mock successful embedding generation
        bge_m3_service.generate_embeddings = AsyncMock(return_value={
            "embeddings": {"dense": [0.1], "sparse": {"0": 0.5}, "multi_vector": [[0.1]]},
            "errors": [],
            "text": "Text 1"
        })
        
        result = await bge_m3_service.process_batch(texts)
        
        assert "results" in result
        assert "performance" in result
        assert "total_time" in result["performance"]
        assert "texts_per_second" in result["performance"]
        assert "cache_hits" in result["performance"]

    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, bge_m3_service):
        """Test batch processing with errors"""
        texts = ["Text 1", "Text 2"]
        
        # Mock one text to cause an error
        bge_m3_service.generate_embeddings = AsyncMock(side_effect=Exception("Processing error"))
        
        result = await bge_m3_service.process_batch(texts)
        
        assert "results" in result
        assert "performance" in result
        assert "errors" in result
        assert len(result["errors"]) == 2

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, bge_m3_service):
        """Test cache statistics retrieval"""
        # Mock Redis info
        bge_m3_service.cache_manager.redis_client = Mock()
        bge_m3_service.cache_manager.redis_client.info.return_value = {
            "used_memory_human": "1.5M",
            "keyspace_hits": 100,
            "keyspace_misses": 50,
            "total_commands_processed": 1000
        }
        
        result = await bge_m3_service.get_cache_stats()
        
        assert "cache_enabled" in result
        assert "used_memory" in result
        assert "connected_clients" in result
        assert "total_commands_processed" in result
        assert "keyspace_hits" in result
        assert "keyspace_misses" in result
        assert "cache_hit_rate" in result
        assert result["cache_hit_rate"] == 66.66666666666667  # (100/(100+50)) * 100

    @pytest.mark.asyncio
    async def test_get_cache_stats_no_redis(self, bge_m3_service):
        """Test cache statistics when Redis is unavailable"""
        bge_m3_service.cache_manager.redis_client = None
        
        result = await bge_m3_service.get_cache_stats()
        
        assert result["cache_enabled"] is False

    @pytest.mark.asyncio
    async def test_clear_all_cache(self, bge_m3_service):
        """Test clearing all cache"""
        bge_m3_service.cache_manager.clear_cache = AsyncMock(return_value=10)
        
        result = await bge_m3_service.clear_all_cache()
        
        assert result["success"] is True
        assert result["cleared_count"] == 10

    @pytest.mark.asyncio
    async def test_clear_all_cache_no_redis(self, bge_m3_service):
        """Test clearing all cache when Redis is unavailable"""
        bge_m3_service.cache_manager.redis_client = None
        
        result = await bge_m3_service.clear_all_cache()
        
        assert result["success"] is False
        assert result["cleared_count"] == 0

    @pytest.mark.asyncio
    async def test_get_embedding_modes(self, bge_m3_service):
        """Test retrieval of available embedding modes"""
        result = bge_m3_service.get_embedding_modes()
        
        assert "dense" in result
        assert "sparse" in result
        assert "multi_vector" in result
        assert "all" in result

    @pytest.mark.asyncio
    async def test_get_embedding_config(self, bge_m3_service):
        """Test retrieval of embedding configuration"""
        result = bge_m3_service.get_embedding_config()
        
        assert "dense_dimension" in result
        assert "sparse_dimension" in result
        assert "multi_vector_dimension" in result
        assert "multi_vector_count" in result
        assert "cache_enabled" in result
        assert "max_retries" in result

    @pytest.mark.asyncio
    async def test_is_cache_enabled(self, bge_m3_service):
        """Test cache availability check"""
        # Test with cache enabled
        bge_m3_service.cache_manager.redis_client = Mock()
        result = bge_m3_service.is_cache_enabled()
        assert result is True
        
        # Test with cache disabled
        bge_m3_service.cache_manager.redis_client = None
        result = bge_m3_service.is_cache_enabled()
        assert result is False

    @pytest.mark.asyncio
    async def test_is_model_ready(self, bge_m3_service):
        """Test model readiness check"""
        # Test with model ready
        bge_m3_service.model_client = "BGE_M3_MODEL_CLIENT"
        result = bge_m3_service.is_model_ready()
        assert result is True
        
        # Test with model not ready
        bge_m3_service.model_client = None
        result = bge_m3_service.is_model_ready()
        assert result is False

    @pytest.mark.asyncio
    async def test_get_service_info(self, bge_m3_service):
        """Test service information retrieval"""
        result = await bge_m3_service.get_service_info()
        
        assert "service_name" in result
        assert "version" in result
        assert "embedding_modes" in result
        assert "cache_enabled" in result
        assert "model_ready" in result
        assert "health_status" in result

    @pytest.mark.asyncio
    async def test_get_service_info_with_errors(self, bge_m3_service):
        """Test service information retrieval with errors"""
        # Mock health check to raise error
        bge_m3_service.health_check = AsyncMock(side_effect=Exception("Health check error"))
        
        result = await bge_m3_service.get_service_info()
        
        assert "service_name" in result
        assert "version" in result
        assert "health_status" in result
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__])
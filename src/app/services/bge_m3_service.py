"""
BGE-M3 Embedding Service for generating Dense, Sparse, and Multi-Vector embeddings
"""

import asyncio
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Union
from functools import wraps
import aiohttp
import redis
from loguru import logger

from src.app.settings import Settings, BGE_M3_Settings


class CacheManager:
    """Manages caching for embeddings to improve performance"""
    
    def __init__(self, settings: BGE_M3_Settings):
        self.settings = settings
        self.redis_client = None
        if settings.cache_enabled:
            try:
                self.redis_client = redis.Redis.from_url(settings.cache_redis_url)
                self.redis_client.ping()
                logger.info("Cache Manager initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.redis_client = None
    
    def _generate_cache_key(self, text: str, mode: str) -> str:
        """Generate cache key for embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"bge_m3:{mode}:{text_hash}"
    
    async def get_embedding(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached embedding"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
        
        return None
    
    async def set_embedding(self, key: str, value: Dict[str, Any]) -> bool:
        """Set cached embedding"""
        if not self.redis_client:
            return False
        
        try:
            serialized_value = json.dumps(value)
            self.redis_client.setex(
                key, 
                self.settings.cache_ttl, 
                serialized_value
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to set cached embedding: {e}")
            return False
    
    async def batch_get_embeddings(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Batch get cached embeddings"""
        if not self.redis_client:
            return [None] * len(keys)
        
        try:
            cached_data = self.redis_client.mget(keys)
            results = []
            for data in cached_data:
                if data:
                    results.append(json.loads(data))
                else:
                    results.append(None)
            return results
        except Exception as e:
            logger.warning(f"Failed to batch get cached embeddings: {e}")
            return [None] * len(keys)
    
    async def batch_set_embeddings(self, key_values: Dict[str, Dict[str, Any]]) -> bool:
        """Batch set cached embeddings"""
        if not self.redis_client:
            return False
        
        try:
            pipeline = self.redis_client.pipeline()
            for key, value in key_values.items():
                serialized_value = json.dumps(value)
                pipeline.setex(key, self.settings.cache_ttl, serialized_value)
            pipeline.execute()
            return True
        except Exception as e:
            logger.warning(f"Failed to batch set cached embeddings: {e}")
            return False
    
    async def clear_cache(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return 0


class ErrorHandler:
    """Handles errors with retry logic and circuit breaker pattern"""
    
    def __init__(self, settings: BGE_M3_Settings):
        self.settings = settings
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0
    
    def handle_errors(self, func):
        """Decorator for error handling with retry logic"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.circuit_open:
                logger.warning("Circuit breaker is open, skipping request")
                raise Exception("Circuit breaker is open")
            
            last_error = None
            for attempt in range(self.settings.max_retries):
                try:
                    result = await func(*args, **kwargs)
                    self.failure_count = 0
                    self.circuit_open = False
                    return result
                except Exception as e:
                    last_error = e
                    self.failure_count += 1
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    
                    if attempt < self.settings.max_retries - 1:
                        await asyncio.sleep(self.settings.retry_delay)
            
            # Check if we should open the circuit breaker
            if self.failure_count >= self.settings.circuit_breaker_threshold:
                self.circuit_open = True
                self.last_failure_time = time.time()
                logger.error("Circuit breaker opened due to too many failures")
            
            raise last_error if last_error else Exception("Unknown error occurred")
        
        return wrapper
    
    async def handle_embedding_error(self, error: Exception, text: str, mode: str) -> Dict[str, Any]:
        """Handle embedding errors with fallback logic"""
        logger.error(f"Error generating {mode} embedding for text: {error}")
        
        # Return fallback embedding
        fallback_embedding = self._get_fallback_embedding(mode)
        
        return {
            "error": str(error),
            "fallback": True,
            "embedding": fallback_embedding,
            "text": text[:100] + "..." if len(text) > 100 else text
        }
    
    def _get_fallback_embedding(self, mode: str) -> Union[List[float], Dict[str, float], List[List[float]]]:
        """Get fallback embedding based on mode"""
        if mode == "dense":
            return [0.0] * self.settings.dense_dimension
        elif mode == "sparse":
            return {}
        elif mode == "multi_vector":
            return [[0.0] * self.settings.multi_vector_dimension for _ in range(self.settings.multi_vector_count)]
        else:
            return []
    
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_open:
            # Check if enough time has passed to reset
            if time.time() - self.last_failure_time > 60:  # 1 minute cooldown
                self.circuit_open = False
                self.failure_count = 0
                return False
            return True
        return False
    
    def reset_circuit(self) -> bool:
        """Reset circuit breaker"""
        self.circuit_open = False
        self.failure_count = 0
        logger.info("Circuit breaker reset")
        return True


class BGE_M3_Service:
    """
    Hauptservice für BGE-M3 Embedding-Generierung
    Unterstützt Dense, Sparse und Multi-Vector Modi
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bge_m3_settings = settings.bge_m3
        self.cache_manager = CacheManager(self.bge_m3_settings)
        self.error_handler = ErrorHandler(self.bge_m3_settings)
        
        # Initialize model client (simulated for now)
        self.model_client = None
        self._initialize_model()
        
        logger.info("BGE-M3 Service initialized successfully")
    
    def _initialize_model(self):
        """Initialize the BGE-M3 model client"""
        try:
            # In a real implementation, this would load the actual BGE-M3 model
            # For now, we'll simulate it with a placeholder
            logger.info(f"Initializing BGE-M3 model: {self.bge_m3_settings.model_name}")
            logger.info(f"Model device: {self.bge_m3_settings.model_device}")
            logger.info(f"Max length: {self.bge_m3_settings.max_length}")
            
            # Simulate model loading
            self.model_client = "BGE_M3_MODEL_CLIENT"
            
        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 model: {e}")
            self.model_client = None
    
    @staticmethod
    def _validate_text(text: str) -> str:
        """Validate and clean input text"""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Limit text length
        if len(text) > 8192:
            logger.warning(f"Text too long, truncating to 8192 characters")
            text = text[:8192]
        
        return text
    
    async def _make_embedding_request(self, text: str, mode: str) -> Dict[str, Any]:
        """Make request to embedding model (simulated)"""
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # Simulate different embedding types
        if mode == "dense":
            embedding = [0.1] * self.bge_m3_settings.dense_dimension
            if self.bge_m3_settings.dense_normalize:
                # Simple normalization
                magnitude = sum(x**2 for x in embedding) ** 0.5
                embedding = [x / magnitude for x in embedding]
            
            return {"dense": embedding}
        
        elif mode == "sparse":
            # Simulate sparse embedding
            embedding = {}
            for i in range(0, min(100, self.bge_m3_settings.sparse_dimension), 10):
                embedding[str(i)] = 0.5 + (i % 5) * 0.1
            
            if self.bge_m3_settings.sparse_normalize:
                max_val = max(embedding.values()) if embedding else 1
                embedding = {k: v / max_val for k, v in embedding.items()}
            
            return {"sparse": embedding}
        
        elif mode == "multi_vector":
            # Simulate multi-vector embedding
            embedding = []
            for _ in range(self.bge_m3_settings.multi_vector_count):
                vector = [0.1] * self.bge_m3_settings.multi_vector_dimension
                embedding.append(vector)
            
            return {"multi_vector": embedding}
        
        else:
            raise ValueError(f"Unknown embedding mode: {mode}")
    
    @ErrorHandler.handle_errors
    async def generate_dense_embedding(self, text: str) -> List[float]:
        """Generiert Dense Embeddings"""
        text = self._validate_text(text)
        
        # Check cache first
        cache_key = self.cache_manager._generate_cache_key(text, "dense")
        cached_result = await self.cache_manager.get_embedding(cache_key)
        
        if cached_result and "dense" in cached_result:
            logger.info(f"Cache hit for dense embedding")
            return cached_result["dense"]
        
        # Generate embedding
        logger.info(f"Generating dense embedding for text (length: {len(text)})")
        start_time = time.time()
        
        try:
            result = await self._make_embedding_request(text, "dense")
            dense_embedding = result["dense"]
            
            # Cache the result
            await self.cache_manager.set_embedding(cache_key, result)
            
            end_time = time.time()
            logger.info(f"Generated dense embedding in {end_time - start_time:.2f}s")
            
            return dense_embedding
            
        except Exception as e:
            return await self.error_handler.handle_embedding_error(e, text, "dense")
    
    @ErrorHandler.handle_errors
    async def generate_sparse_embedding(self, text: str) -> Dict[str, float]:
        """Generiert Sparse Embeddings"""
        text = self._validate_text(text)
        
        # Check cache first
        cache_key = self.cache_manager._generate_cache_key(text, "sparse")
        cached_result = await self.cache_manager.get_embedding(cache_key)
        
        if cached_result and "sparse" in cached_result:
            logger.info(f"Cache hit for sparse embedding")
            return cached_result["sparse"]
        
        # Generate embedding
        logger.info(f"Generating sparse embedding for text (length: {len(text)})")
        start_time = time.time()
        
        try:
            result = await self._make_embedding_request(text, "sparse")
            sparse_embedding = result["sparse"]
            
            # Cache the result
            await self.cache_manager.set_embedding(cache_key, result)
            
            end_time = time.time()
            logger.info(f"Generated sparse embedding in {end_time - start_time:.2f}s")
            
            return sparse_embedding
            
        except Exception as e:
            return await self.error_handler.handle_embedding_error(e, text, "sparse")
    
    @ErrorHandler.handle_errors
    async def generate_multivector_embedding(self, text: str) -> List[List[float]]:
        """Generiert Multi-Vector Embeddings (ColBERT)"""
        text = self._validate_text(text)
        
        # Check cache first
        cache_key = self.cache_manager._generate_cache_key(text, "multi_vector")
        cached_result = await self.cache_manager.get_embedding(cache_key)
        
        if cached_result and "multi_vector" in cached_result:
            logger.info(f"Cache hit for multi-vector embedding")
            return cached_result["multi_vector"]
        
        # Generate embedding
        logger.info(f"Generating multi-vector embedding for text (length: {len(text)})")
        start_time = time.time()
        
        try:
            result = await self._make_embedding_request(text, "multi_vector")
            multivector_embedding = result["multi_vector"]
            
            # Cache the result
            await self.cache_manager.set_embedding(cache_key, result)
            
            end_time = time.time()
            logger.info(f"Generated multi-vector embedding in {end_time - start_time:.2f}s")
            
            return multivector_embedding
            
        except Exception as e:
            fallback = await self.error_handler.handle_embedding_error(e, text, "multi_vector")
            return fallback["embedding"]
    
    async def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """Generiert alle drei Embedding-Typen"""
        logger.info(f"Generating all embeddings for text (length: {len(text)})")
        
        try:
            # Generate all embeddings concurrently
            tasks = [
                self.generate_dense_embedding(text),
                self.generate_sparse_embedding(text),
                self.generate_multivector_embedding(text)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            embeddings = {}
            errors = []
            
            if isinstance(results[0], Exception):
                errors.append(("dense", str(results[0])))
            else:
                embeddings["dense"] = results[0]
            
            if isinstance(results[1], Exception):
                errors.append(("sparse", str(results[1])))
            else:
                embeddings["sparse"] = results[1]
            
            if isinstance(results[2], Exception):
                errors.append(("multi_vector", str(results[2])))
            else:
                embeddings["multi_vector"] = results[2]
            
            # Log errors if any
            if errors:
                for mode, error in errors:
                    logger.error(f"Error generating {mode} embedding: {error}")
            
            return {
                "embeddings": embeddings,
                "errors": errors,
                "text": text[:100] + "..." if len(text) > 100 else text
            }
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {
                "embeddings": {},
                "errors": [("all", str(e))],
                "text": text[:100] + "..." if len(text) > 100 else text
            }
    
    async def batch_generate_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch-Verarbeitung für Effizienz"""
        logger.info(f"Processing batch of {len(texts)} texts for embeddings")
        
        if not texts:
            return []
        
        # Limit batch size
        batch_size = min(len(texts), self.bge_m3_settings.batch_size)
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_texts)} texts")
            
            try:
                # Process batch concurrently
                tasks = [self.generate_embeddings(text) for text in batch_texts]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing text {i + j}: {result}")
                        results.append({
                            "embeddings": {},
                            "errors": [("all", str(result))],
                            "text": batch_texts[j][:100] + "..." if len(batch_texts[j]) > 100 else batch_texts[j]
                        })
                    else:
                        results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add error entries for all texts in this batch
                for text in batch_texts:
                    results.append({
                        "embeddings": {},
                        "errors": [("batch", str(e))],
                        "text": text[:100] + "..." if len(text) > 100 else text
                    })
        
        logger.info(f"Completed batch processing: {len(results)} results")
        return results
    
    async def cache_embeddings(self, texts: List[str], mode: str = "all") -> Dict[str, bool]:
        """Cache embeddings for multiple texts"""
        logger.info(f"Caching embeddings for {len(texts)} texts (mode: {mode})")
        
        if not self.bge_m3_settings.cache_enabled:
            logger.info("Caching is disabled")
            return {}
        
        results = {}
        
        try:
            for text in texts:
                cache_key = self.cache_manager._generate_cache_key(text, mode)
                
                if mode == "all":
                    # Generate and cache all embeddings
                    embedding_result = await self.generate_embeddings(text)
                    await self.cache_manager.set_embedding(cache_key, embedding_result["embeddings"])
                    results[text] = True
                else:
                    # Generate and cache specific embedding type
                    if mode == "dense":
                        embedding = await self.generate_dense_embedding(text)
                    elif mode == "sparse":
                        embedding = await self.generate_sparse_embedding(text)
                    elif mode == "multi_vector":
                        embedding = await self.generate_multivector_embedding(text)
                    else:
                        logger.error(f"Unknown embedding mode: {mode}")
                        results[text] = False
                        continue
                    
                    await self.cache_manager.set_embedding(cache_key, {mode: embedding})
                    results[text] = True
                
        except Exception as e:
            logger.error(f"Error caching embeddings: {e}")
            # Mark all as failed
            for text in texts:
                results[text] = False
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Cached {success_count}/{len(texts)} embeddings successfully")
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Überprüft die Verfügbarkeit des Modells"""
        logger.info("Performing health check")
        
        health_status = {
            "service": "BGE-M3 Service",
            "status": "healthy",
            "timestamp": time.time(),
            "cache_enabled": self.bge_m3_settings.cache_enabled,
            "cache_available": self.cache_manager.redis_client is not None,
            "model_available": self.model_client is not None,
            "circuit_breaker": {
                "open": self.error_handler.is_circuit_open(),
                "failure_count": self.error_handler.failure_count,
                "last_failure": self.error_handler.last_failure_time
            }
        }
        
        try:
            # Test model availability
            if self.model_client:
                # Simple test embedding
                test_text = "health check"
                test_embedding = await self.generate_dense_embedding(test_text)
                
                if test_embedding and len(test_embedding) == self.bge_m3_settings.dense_dimension:
                    health_status["model_test"] = "passed"
                else:
                    health_status["model_test"] = "failed"
                    health_status["status"] = "degraded"
            else:
                health_status["model_test"] = "not_available"
                health_status["status"] = "degraded"
            
            # Test cache availability
            if self.cache_manager.redis_client:
                test_key = "health_check_test"
                test_value = {"test": True}
                cache_success = await self.cache_manager.set_embedding(test_key, test_value)
                
                if cache_success:
                    health_status["cache_test"] = "passed"
                    # Clean up test key
                    await self.cache_manager.clear_cache("health_check_test")
                else:
                    health_status["cache_test"] = "failed"
                    health_status["status"] = "degraded"
            else:
                health_status["cache_test"] = "not_available"
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        logger.info(f"Health check completed: {health_status['status']}")
        return health_status
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache_manager.redis_client:
            return {"cache_enabled": False}
        
        try:
            info = self.cache_manager.redis_client.info()
            return {
                "cache_enabled": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "cache_hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"cache_enabled": True, "error": str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate"""
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100
    
    async def clear_cache(self, pattern: str = "*") -> Dict[str, Any]:
        """Clear cache entries"""
        if not self.cache_manager.redis_client:
            return {"success": False, "message": "Cache not available"}
        
        try:
            cleared_count = await self.cache_manager.clear_cache(pattern)
            return {
                "success": True,
                "cleared_count": cleared_count,
                "pattern": pattern
            }
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return {
                "success": False,
                "error": str(e),
                "pattern": pattern
            }
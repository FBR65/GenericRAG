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
        self.failure_count = 0
        self.last_failure_time = 0
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
    
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        return self.failure_count >= self.settings.circuit_breaker_threshold
    
    def record_failure(self):
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Reset failure count if enough time has passed
        if self.failure_count > 0 and (time.time() - self.last_failure_time) > 60:  # 1 minute
            self.failure_count = 0
    
    def record_success(self):
        """Record a success"""
        # Reset failure count on success
        self.failure_count = 0
    
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
        self.model_client = "BGE_M3_MODEL_CLIENT"  # Always set to simulate model availability
        
        logger.info("BGE-M3 Service initialized successfully")
    
    
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
    
    async def generate_dense_embedding(self, text: str) -> List[float]:
        """Generate dense embedding for text"""
        text = self._validate_text(text)
        
        # Check cache first
        if self.bge_m3_settings.cache_enabled:
            cache_key = self.cache_manager._generate_cache_key(text, "dense")
            cached_result = await self.cache_manager.get_embedding(cache_key)
            if cached_result and "dense" in cached_result:
                logger.info(f"Cache hit for dense embedding: {text[:50]}...")
                return cached_result["dense"]
        
        # Generate embedding
        result = await self._make_embedding_request(text, "dense")
        
        # Cache result
        if self.bge_m3_settings.cache_enabled and "dense" in result:
            await self.cache_manager.set_embedding(cache_key, result)
        
        return result.get("dense", [])
    
    async def generate_sparse_embedding(self, text: str) -> Dict[str, float]:
        """Generate sparse embedding for text"""
        text = self._validate_text(text)
        
        # Check cache first
        if self.bge_m3_settings.cache_enabled:
            cache_key = self.cache_manager._generate_cache_key(text, "sparse")
            cached_result = await self.cache_manager.get_embedding(cache_key)
            if cached_result and "sparse" in cached_result:
                logger.info(f"Cache hit for sparse embedding: {text[:50]}...")
                return cached_result["sparse"]
        
        # Generate embedding
        result = await self._make_embedding_request(text, "sparse")
        
        # Cache result
        if self.bge_m3_settings.cache_enabled and "sparse" in result:
            await self.cache_manager.set_embedding(cache_key, result)
        
        return result.get("sparse", {})
    
    async def generate_multivector_embedding(self, text: str) -> List[List[float]]:
        """Generate multivector embedding for text"""
        text = self._validate_text(text)
        
        # Check cache first
        if self.bge_m3_settings.cache_enabled:
            cache_key = self.cache_manager._generate_cache_key(text, "multi_vector")
            cached_result = await self.cache_manager.get_embedding(cache_key)
            if cached_result and "multi_vector" in cached_result:
                logger.info(f"Cache hit for multivector embedding: {text[:50]}...")
                return cached_result["multi_vector"]
        
        # Generate embedding
        result = await self._make_embedding_request(text, "multi_vector")
        
        # Cache result
        if self.bge_m3_settings.cache_enabled and "multi_vector" in result:
            await self.cache_manager.set_embedding(cache_key, result)
        
        return result.get("multi_vector", [])
    
    async def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """Generate all types of embeddings for text"""
        text = self._validate_text(text)
        
        result = {
            "embeddings": {},
            "errors": [],
            "text": text
        }
        
        try:
            # Generate dense embedding
            try:
                dense_embedding = await self.generate_dense_embedding(text)
                result["embeddings"]["dense"] = dense_embedding
            except Exception as e:
                error_msg = f"Dense embedding generation failed: {str(e)}"
                result["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Generate sparse embedding
            try:
                sparse_embedding = await self.generate_sparse_embedding(text)
                result["embeddings"]["sparse"] = sparse_embedding
            except Exception as e:
                error_msg = f"Sparse embedding generation failed: {str(e)}"
                result["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Generate multivector embedding
            try:
                multivector_embedding = await self.generate_multivector_embedding(text)
                result["embeddings"]["multi_vector"] = multivector_embedding
            except Exception as e:
                error_msg = f"Multivector embedding generation failed: {str(e)}"
                result["errors"].append(error_msg)
                logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
        
        return result
    
    async def batch_generate_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate embeddings for multiple texts"""
        results = []
        
        for text in texts:
            try:
                result = await self.generate_embeddings(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error generating embeddings for text '{text[:50]}...': {e}")
                results.append({
                    "embeddings": {},
                    "errors": [str(e)],
                    "text": text
                })
        
        return results
    
    def is_model_ready(self) -> bool:
        """Check if model is ready"""
        return self.model_client is not None
    
    def get_embedding_modes(self) -> List[str]:
        """Get available embedding modes"""
        return ["dense", "sparse", "multi_vector"]
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return {
            "dense": {
                "dimension": self.bge_m3_settings.dense_dimension,
                "normalize": self.bge_m3_settings.dense_normalize
            },
            "sparse": {
                "dimension": self.bge_m3_settings.sparse_dimension,
                "normalize": self.bge_m3_settings.sparse_normalize
            },
            "multi_vector": {
                "count": self.bge_m3_settings.multi_vector_count,
                "dimension": self.bge_m3_settings.multi_vector_dimension
            }
        }
    
    def is_cache_enabled(self) -> bool:
        """Check if cache is enabled"""
        return self.bge_m3_settings.cache_enabled
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache_manager.redis_client:
            return {
                "status": "disabled",
                "error": "Redis not available"
            }
        
        try:
            info = self.cache_manager.redis_client.info()
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            hit_rate = self._calculate_hit_rate(hits, misses)
            
            return {
                "status": "enabled",
                "hit_rate": hit_rate,
                "hits": hits,
                "misses": misses,
                "total_commands": info.get("total_commands_processed", 0),
                "memory_usage": info.get("used_memory_human", "N/A")
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def clear_all_cache(self) -> int:
        """Clear all cache entries"""
        if not self.cache_manager.redis_client:
            return 0
        
        try:
            return await self.cache_manager.clear_cache()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    async def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process batch of texts for embeddings"""
        return await self.batch_generate_embeddings(texts)
    
    
    
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
                    health_status["status"] = "unhealthy"
            else:
                health_status["model_test"] = "not_available"
                health_status["status"] = "unhealthy"
            
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
                    health_status["status"] = "unhealthy"
            else:
                health_status["cache_test"] = "not_available"
                health_status["cache_status"] = "unavailable"
            
            # Determine overall status based on availability
            if health_status["model_available"] and health_status["cache_available"]:
                health_status["status"] = "healthy"
            elif not health_status["model_available"] and not health_status["cache_available"]:
                health_status["status"] = "unhealthy"
            else:
                health_status["status"] = "degraded"
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        logger.info(f"Health check completed: {health_status['status']}")
        return health_status
    
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate"""
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100
    
    
    
    
    
    
    
    
    
    
    
    
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        try:
            health = await self.health_check()
            cache_stats = await self.get_cache_stats()
            
            return {
                "service_name": "BGE-M3 Service",
                "version": "1.0.0",
                "embedding_modes": self.get_embedding_modes(),
                "cache_enabled": self.bge_m3_settings.cache_enabled,
                "model_ready": self.is_model_ready(),
                "health_status": health["status"],
                "cache_status": cache_stats["status"],
                "config": self.get_embedding_config()
            }
        except Exception as e:
            logger.error(f"Error getting service info: {e}")
            return {
                "service_name": "BGE-M3 Service",
                "version": "1.0.0",
                "error": str(e),
                "health_status": "error"
            }
    
    
    
    
    
    
    
    
    
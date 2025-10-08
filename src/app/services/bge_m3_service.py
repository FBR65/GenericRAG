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
import numpy as np

from FlagEmbedding import BGEM3FlagModel
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
        
        # Initialize real BGE-M3 model
        self.model_client = None
        self._initialize_model()
        
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
    
    def _initialize_model(self):
        """Initialize the BGE-M3 model"""
        try:
            # Use FlagEmbedding for BGE-M3
            logger.info(f"Loading BGE-M3 model: {self.bge_m3_settings.model_name}")
            
            # Load the model with FlagEmbedding
            self.model_client = BGEM3FlagModel(
                self.bge_m3_settings.model_name,
                device=self.bge_m3_settings.model_device,
                use_fp16=True  # Use half precision for better performance
            )
            
            logger.info("BGE-M3 model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import FlagEmbedding: {e}")
            logger.error("Please install FlagEmbedding: pip install FlagEmbedding")
            self.model_client = None
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            self.model_client = None
    
    async def _make_embedding_request(self, text: str, mode: str) -> Dict[str, Any]:
        """Make request to embedding model"""
        if not self.model_client:
            raise Exception("BGE-M3 model is not available")
        
        # Simulate API call delay for async consistency
        await asyncio.sleep(0.01)
        
        try:
            if mode == "dense":
                # Generate dense embedding using BGE-M3
                # Use the correct method for BGEM3FlagModel
                output = self.model_client.encode(
                    [text],
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False
                )
                
                embedding = output['dense_vecs'][0]
                
                # Ensure correct dimension
                if len(embedding) != self.bge_m3_settings.dense_dimension:
                    logger.warning(f"Dense embedding dimension mismatch: got {len(embedding)}, expected {self.bge_m3_settings.dense_dimension}")
                    # Pad or truncate to match expected dimension
                    if len(embedding) < self.bge_m3_settings.dense_dimension:
                        embedding = np.pad(embedding, (0, self.bge_m3_settings.dense_dimension - len(embedding)))
                    else:
                        embedding = embedding[:self.bge_m3_settings.dense_dimension]
                
                return {"dense": embedding.tolist()}
            
            elif mode == "sparse":
                # Generate sparse embedding using BGE-M3
                # Use the correct method for BGEM3FlagModel
                output = self.model_client.encode(
                    [text],
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False
                )
                
                sparse_embedding = output['lexical_weights'][0]
                
                # Convert to dictionary format
                if isinstance(sparse_embedding, dict):
                    # Already in dictionary format
                    pass
                else:
                    # Fallback: create sparse representation from dense
                    sparse_embedding = {}
                    for i, value in enumerate(sparse_embedding):
                        if abs(value) > 0.01:
                            sparse_embedding[str(i)] = float(value)
                
                # Normalize if requested
                if self.bge_m3_settings.sparse_normalize and sparse_embedding:
                    max_val = max(abs(v) for v in sparse_embedding.values())
                    if max_val > 0:
                        sparse_embedding = {k: v / max_val for k, v in sparse_embedding.items()}
                
                return {"sparse": sparse_embedding}
            
            elif mode == "multi_vector":
                # Generate multi-vector embedding using BGE-M3
                # Use the correct method for BGEM3FlagModel
                output = self.model_client.encode(
                    [text],
                    return_dense=False,
                    return_sparse=False,
                    return_colbert_vecs=True
                )
                
                multi_embedding = output['colbert_vecs'][0]
                
                # Ensure we have the right format
                if not isinstance(multi_embedding, list):
                    # If not a list, try to convert or create from dense
                    dense_output = self.model_client.encode(
                        [text],
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False
                    )
                    
                    dense_embedding = dense_output['dense_vecs'][0]
                    
                    # Split dense embedding into multiple vectors
                    vector_dimension = self.bge_m3_settings.multi_vector_dimension
                    vector_count = self.bge_m3_settings.multi_vector_count
                    
                    multi_embedding = []
                    for i in range(vector_count):
                        start_idx = i * vector_dimension
                        end_idx = start_idx + vector_dimension
                        
                        # Extract slice or pad with zeros
                        if end_idx <= len(dense_embedding):
                            vector = dense_embedding[start_idx:end_idx]
                        else:
                            # Pad with zeros
                            vector = list(dense_embedding[start_idx:]) + [0.0] * (end_idx - len(dense_embedding))
                        
                        multi_embedding.append(vector)
                
                # Ensure we have the right number of vectors
                if len(multi_embedding) < self.bge_m3_settings.multi_vector_count:
                    # Pad with zero vectors
                    for _ in range(self.bge_m3_settings.multi_vector_count - len(multi_embedding)):
                        zero_vector = [0.0] * self.bge_m3_settings.multi_vector_dimension
                        multi_embedding.append(zero_vector)
                elif len(multi_embedding) > self.bge_m3_settings.multi_vector_count:
                    # Truncate
                    multi_embedding = multi_embedding[:self.bge_m3_settings.multi_vector_count]
                
                return {"multi_vector": multi_embedding}
            
            else:
                raise ValueError(f"Unknown embedding mode: {mode}")
                
        except Exception as e:
            logger.error(f"Error generating {mode} embedding: {e}")
            # Return fallback embedding in correct format
            fallback = self.error_handler._get_fallback_embedding(mode)
            return {mode: fallback}
    
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
    
    async def generate_embeddings(
        self,
        text: str,
        include_dense: bool = True,
        include_sparse: bool = True,
        include_multivector: bool = True,
        cache_embeddings: bool = True,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate all types of embeddings for text using BGE-M3"""
        text = self._validate_text(text)
        
        result = {
            "embeddings": {},
            "errors": [],
            "text": text
        }
        
        # Debug: Log input parameters
        logger.debug(f"DEBUG: generate_embeddings called with text: '{text[:50]}...'")
        logger.debug(f"DEBUG: include_dense: {include_dense}, include_sparse: {include_sparse}, include_multivector: {include_multivector}")
        
        try:
            # Check if model is available
            if not self.model_client:
                error_msg = "BGE-M3 model is not available"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                # Set fallback embeddings
                if include_dense:
                    result["embeddings"]["dense"] = [0.0] * self.bge_m3_settings.dense_dimension
                if include_sparse:
                    result["embeddings"]["sparse"] = {}
                if include_multivector:
                    result["embeddings"]["multi_vector"] = []
                return result
            
            # Generate all embeddings at once using the correct BGE-M3 method
            logger.debug(f"DEBUG: Calling model_client.encode with return_dense={include_dense}, return_sparse={include_sparse}, return_colbert_vecs={include_multivector}")
            
            output = self.model_client.encode(
                [text],
                return_dense=include_dense,
                return_sparse=include_sparse,
                return_colbert_vecs=include_multivector
            )
            
            logger.debug(f"DEBUG: Model encode completed successfully")
            logger.debug(f"DEBUG: Output keys: {list(output.keys())}")
            
            # Extract the embeddings
            if include_dense:
                if 'dense_vecs' not in output:
                    error_msg = "Dense vectors not found in output"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
                    result["embeddings"]["dense"] = [0.0] * self.bge_m3_settings.dense_dimension
                else:
                    dense_embedding = output['dense_vecs'][0]
                    logger.debug(f"DEBUG: Raw dense embedding: type={type(dense_embedding)}, length={len(dense_embedding) if hasattr(dense_embedding, '__len__') else 'N/A'}")
                    
                    # Ensure correct dimension
                    if len(dense_embedding) != self.bge_m3_settings.dense_dimension:
                        logger.warning(f"Dense embedding dimension mismatch: got {len(dense_embedding)}, expected {self.bge_m3_settings.dense_dimension}")
                        # Pad or truncate to match expected dimension
                        if len(dense_embedding) < self.bge_m3_settings.dense_dimension:
                            dense_embedding = np.pad(dense_embedding, (0, self.bge_m3_settings.dense_dimension - len(dense_embedding)))
                        else:
                            dense_embedding = dense_embedding[:self.bge_m3_settings.dense_dimension]
                    
                    result["embeddings"]["dense"] = dense_embedding.tolist()
                    logger.info(f"Dense embedding generated: {len(dense_embedding)} dimensions, sample: {dense_embedding[:5]}")
            
            if include_sparse:
                if 'lexical_weights' not in output:
                    error_msg = "Sparse vectors not found in output"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
                    result["embeddings"]["sparse"] = {}
                else:
                    sparse_embedding = output['lexical_weights'][0]
                    logger.debug(f"DEBUG: Raw sparse embedding: type={type(sparse_embedding)}, length={len(sparse_embedding) if hasattr(sparse_embedding, '__len__') else 'N/A'}")
                    
                    # Convert to dictionary format if it's not already
                    if isinstance(sparse_embedding, dict):
                        result["embeddings"]["sparse"] = sparse_embedding
                        logger.info(f"Sparse embedding generated: {len(sparse_embedding)} items (dict format)")
                    else:
                        # Convert list to dict - this is likely the issue
                        sparse_dict = {}
                        non_zero_count = 0
                        for i, value in enumerate(sparse_embedding):
                            if abs(value) > 0.001:  # Reduced threshold to include more values
                                sparse_dict[str(i)] = float(value)
                                non_zero_count += 1
                        
                        result["embeddings"]["sparse"] = sparse_dict
                        logger.info(f"Sparse embedding generated: {len(sparse_dict)} non-zero items from {len(sparse_embedding)} total (threshold: 0.001)")
                        
                        # Debug: Log sparse vector details
                        if len(sparse_dict) == 0:
                            logger.warning(f"Sparse vector is empty! This might indicate an issue with the content or threshold")
                            logger.debug(f"Sample values from sparse_embedding: {sparse_embedding[:10]}")
                            logger.debug(f"Absolute values: {[abs(v) for v in sparse_embedding[:10]]}")
                        elif len(sparse_dict) < 5:  # Log if we have very few entries
                            logger.warning(f"Sparse vector has only {len(sparse_dict)} entries, which might indicate an issue")
                            logger.debug(f"Sparse vector sample: {dict(list(sparse_dict.items())[:5])}")
            
            if include_multivector:
                if 'colbert_vecs' not in output:
                    error_msg = "ColBERT vectors not found in output"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
                    result["embeddings"]["multi_vector"] = []
                else:
                    colbert_embedding = output['colbert_vecs'][0]
                    logger.debug(f"DEBUG: Raw ColBERT embedding: type={type(colbert_embedding)}, length={len(colbert_embedding) if hasattr(colbert_embedding, '__len__') else 'N/A'}")
                    
                    # Ensure we have the right format (list of lists)
                    if isinstance(colbert_embedding, np.ndarray):
                        colbert_embedding = colbert_embedding.tolist()
                    
                    # Ensure we have the right number of ColBERT vectors
                    if len(colbert_embedding) < self.bge_m3_settings.multi_vector_count:
                        # Pad with zero vectors
                        for _ in range(self.bge_m3_settings.multi_vector_count - len(colbert_embedding)):
                            zero_vector = [0.0] * self.bge_m3_settings.multi_vector_dimension
                            colbert_embedding.append(zero_vector)
                    elif len(colbert_embedding) > self.bge_m3_settings.multi_vector_count:
                        # Truncate
                        colbert_embedding = colbert_embedding[:self.bge_m3_settings.multi_vector_count]
                    
                    result["embeddings"]["multi_vector"] = colbert_embedding
                    logger.info(f"Multi-vector embedding generated: {len(colbert_embedding)} vectors")
            
            # Debug: Log final result
            logger.debug(f"DEBUG: Final embeddings: {list(result['embeddings'].keys())}")
            for key, value in result['embeddings'].items():
                if key == 'dense':
                    logger.debug(f"DEBUG: {key}: length={len(value)}, sample={value[:3]}")
                elif key == 'sparse':
                    logger.debug(f"DEBUG: {key}: length={len(value)}, sample={dict(list(value.items())[:3])}")
                elif key == 'multi_vector':
                    logger.debug(f"DEBUG: {key}: length={len(value)}, first_vector_length={len(value[0]) if value else 0}")
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
            logger.error(f"DEBUG: Error type: {type(e)}")
            logger.error(f"DEBUG: Error details: {e}")
            
            # Set fallback embeddings
            if include_dense:
                result["embeddings"]["dense"] = [0.0] * self.bge_m3_settings.dense_dimension
            if include_sparse:
                result["embeddings"]["sparse"] = {}
            if include_multivector:
                result["embeddings"]["multi_vector"] = []
        
        return result
    
    async def batch_generate_embeddings(
        self,
        texts: List[str],
        include_dense: bool = True,
        include_sparse: bool = True,
        include_multivector: bool = True,
        cache_embeddings: bool = True,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for multiple texts"""
        results = []
        
        for text in texts:
            try:
                result = await self.generate_embeddings(
                    text,
                    include_dense=include_dense,
                    include_sparse=include_sparse,
                    include_multivector=include_multivector,
                    cache_embeddings=cache_embeddings,
                    session_id=session_id
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error generating embeddings for text '{text[:50]}...': {e}")
                # Ensure we always return a valid structure
                results.append({
                    "embeddings": {
                        "dense": [0.0] * self.bge_m3_settings.dense_dimension if include_dense else [],
                        "sparse": {} if include_sparse else {},
                        "multi_vector": [] if include_multivector else []
                    },
                    "errors": [str(e)],
                    "text": text
                })
        
        return results
    
    def is_available(self) -> bool:
        """Check if service is available"""
        # Check if model is available
        if self.model_client is None:
            return False
        
        # Try a simple test to ensure the model works
        try:
            test_text = "test"
            test_embedding = self.model_client.encode([test_text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
            return len(test_embedding['dense_vecs'][0]) > 0
        except Exception:
            return False
    
    def is_model_ready(self) -> bool:
        """Check if model is ready"""
        if self.model_client is None:
            return False
        
        # Try a simple test to ensure the model works
        try:
            test_text = "test"
            test_embedding = self.model_client.encode(test_text, convert_to_tensor=False)
            return len(test_embedding) > 0
        except Exception:
            return False
    
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
    
    
    
    
    
    
    
    
    
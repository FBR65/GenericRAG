import os
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    collection_name: str = "generic_rag_collection"
    image_collection_name: str = "generic_rag_images"
    qdrant_url: str = "http://10.84.0.7:6333"
    qdrant_api_key: str = ""

    # Dense embedding configuration
    dense_model: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    dense_dimension: int = 1024
    dense_base_url: str = "http://10.84.0.10:8100"
    dense_api_key: str = ""

    # Sparse embedding configuration
    sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    sparse_max_features: int = 1000

    # CLIP embedding configuration for images
    clip_model: str = "clip-vit-base-patch32"
    clip_dimension: int = 512
    clip_local: bool = True
    clip_device: str = "auto"
    clip_ollama_endpoint: str = "http://localhost:11434"

    # Hybrid search configuration
    hybrid_search_alpha: float = 0.5  # Weight for dense vs sparse in hybrid search
    hybrid_search_limit: int = 10

    model_config = {"env_prefix": "QDRANT_"}


class LLMSettings(BaseSettings):
    # Student model configuration
    student_model: str = "google/gemma-3-27b-it"
    student_base_url: str = "http://10.78.0.5:8114/v1"
    student_api_key: str = ""

    # Teacher model configuration
    teacher_model: str = "google/gemma-3-27b-it"
    teacher_base_url: str = "http://10.78.0.5:8114/v1"
    teacher_api_key: str = ""

    # Legacy compatibility
    gemma_base_url: str = "http://10.78.0.5:8114/v1"
    gemma_api_key: str = "your_gemma_api_key_here"

    model_config = {"env_prefix": "LLM_"}


class DSPySettings(BaseSettings):
    max_full_evals: int = 3
    num_threads: int = 4
    track_stats: bool = True
    reflection_lm: str = "teacher"
    judge_lm: str = "student"

    model_config = {"env_prefix": "DSPY_"}


class StorageSettings(BaseSettings):
    image_storage_path: str = "./data/images"
    temp_storage_path: str = "./data/temp"
    dspy_cache_path: str = "./data/dspy_cache"

    model_config = {"env_prefix": "IMAGE_"}


class AppSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8001

    model_config = {"env_prefix": "APP_"}


class GradioSettings(BaseSettings):
    port: int = 7860

    model_config = {"env_prefix": "GRADIO_"}


class BGE_M3_Settings(BaseSettings):
    # BGE-M3 Modell Konfiguration
    model_name: str = "BAAI/bge-m3"
    model_device: str = "auto"
    max_length: int = 8192
    
    # Dense Embedding Konfiguration
    dense_dimension: int = 1024
    dense_normalize: bool = True
    
    # Sparse Embedding Konfiguration
    sparse_dimension: int = 49152  # BGE-M3 Vocabulary Size
    sparse_normalize: bool = True
    
    # Multi-Vector Embedding Konfiguration
    multi_vector_dimension: int = 1024
    multi_vector_count: int = 32
    
    # Cache Konfiguration
    cache_enabled: bool = True
    cache_redis_url: str = "redis://10.84.0.4:6379"
    cache_ttl: int = 3600  # 1 Stunde
    cache_max_size: int = 10000
    
    # Error Handling Konfiguration
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    
    # Performance Konfiguration
    batch_size: int = 32
    max_workers: int = 4
    timeout: int = 30
    
    model_config = {"env_prefix": "BGE_M3_"}


class Settings(BaseSettings):
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    dspy: DSPySettings = Field(default_factory=DSPySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    gradio: GradioSettings = Field(default_factory=GradioSettings)
    bge_m3: BGE_M3_Settings = Field(default_factory=BGE_M3_Settings)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        Path(self.storage.image_storage_path).mkdir(parents=True, exist_ok=True)
        Path(self.storage.temp_storage_path).mkdir(parents=True, exist_ok=True)
        Path(self.storage.dspy_cache_path).mkdir(parents=True, exist_ok=True)

        # Only update nested settings with environment variables if the values are still default
        self._update_nested_settings_if_default()

    def _update_nested_settings_if_default(self):
        """Update nested settings with environment variables only if values are still default"""
        # Update Qdrant settings only if environment variable is set
        if "QDRANT_URL" in os.environ:
            self.qdrant.qdrant_url = os.environ["QDRANT_URL"]
        if "QDRANT_COLLECTION_NAME" in os.environ:
            self.qdrant.collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        if "QDRANT_API_KEY" in os.environ:
            self.qdrant.qdrant_api_key = os.environ["QDRANT_API_KEY"]
        if "QDRANT_DENSE_MODEL" in os.environ:
            self.qdrant.dense_model = os.environ["QDRANT_DENSE_MODEL"]
        if "QDRANT_DENSE_DIMENSION" in os.environ:
            self.qdrant.dense_dimension = int(os.environ["QDRANT_DENSE_DIMENSION"])
        if "QDRANT_SPARSE_MODEL" in os.environ:
            self.qdrant.sparse_model = os.environ["QDRANT_SPARSE_MODEL"]
        if "QDRANT_SPARSE_MAX_FEATURES" in os.environ:
            self.qdrant.sparse_max_features = int(
                os.environ["QDRANT_SPARSE_MAX_FEATURES"]
            )
        if "QDRANT_HYBRID_SEARCH_ALPHA" in os.environ:
            self.qdrant.hybrid_search_alpha = float(
                os.environ["QDRANT_HYBRID_SEARCH_ALPHA"]
            )
        if "QDRANT_HYBRID_SEARCH_LIMIT" in os.environ:
            self.qdrant.hybrid_search_limit = int(
                os.environ["QDRANT_HYBRID_SEARCH_LIMIT"]
            )
        if "QDRANT_IMAGE_COLLECTION_NAME" in os.environ:
            self.qdrant.image_collection_name = os.environ[
                "QDRANT_IMAGE_COLLECTION_NAME"
            ]
        if "QDRANT_CLIP_MODEL" in os.environ:
            self.qdrant.clip_model = os.environ["QDRANT_CLIP_MODEL"]
        if "QDRANT_CLIP_DIMENSION" in os.environ:
            self.qdrant.clip_dimension = int(os.environ["QDRANT_CLIP_DIMENSION"])
        if "QDRANT_CLIP_LOCAL" in os.environ:
            self.qdrant.clip_local = os.environ["QDRANT_CLIP_LOCAL"].lower() == "true"
        if "QDRANT_CLIP_DEVICE" in os.environ:
            self.qdrant.clip_device = os.environ["QDRANT_CLIP_DEVICE"]
        if "QDRANT_CLIP_OLLAMA_ENDPOINT" in os.environ:
            self.qdrant.clip_ollama_endpoint = os.environ["QDRANT_CLIP_OLLAMA_ENDPOINT"]
        if "QDRANT_DENSE_BASE_URL" in os.environ:
            self.qdrant.dense_base_url = os.environ["QDRANT_DENSE_BASE_URL"]
        if "QDRANT_DENSE_API_KEY" in os.environ:
            self.qdrant.dense_api_key = os.environ["QDRANT_DENSE_API_KEY"]

        # Update LLM settings only if environment variables are set
        if "STUDENT_MODEL" in os.environ:
            self.llm.student_model = os.environ["STUDENT_MODEL"]
        if "TEACHER_MODEL" in os.environ:
            self.llm.teacher_model = os.environ["TEACHER_MODEL"]
        if "GEMMA_BASE_URL" in os.environ:
            self.llm.gemma_base_url = os.environ["GEMMA_BASE_URL"]
        if "GEMMA_API_KEY" in os.environ:
            self.llm.gemma_api_key = os.environ["GEMMA_API_KEY"]

        # Update DSPy settings only if environment variables are set
        if "MAX_FULL_EVALS" in os.environ:
            self.dspy.max_full_evals = int(os.environ["MAX_FULL_EVALS"])
        if "NUM_THREADS" in os.environ:
            self.dspy.num_threads = int(os.environ["NUM_THREADS"])
        if "TRACK_STATS" in os.environ:
            self.dspy.track_stats = os.environ["TRACK_STATS"].lower() == "true"

        # Update Storage settings only if environment variable is set
        if "IMAGE_STORAGE_PATH" in os.environ:
            self.storage.image_storage_path = os.environ["IMAGE_STORAGE_PATH"]

        # Update App settings only if environment variables are set
        if "APP_HOST" in os.environ:
            self.app.host = os.environ["APP_HOST"]
        if "APP_PORT" in os.environ:
            self.app.port = int(os.environ["APP_PORT"])

        # Update Gradio settings only if environment variable is set
        if "GRADIO_PORT" in os.environ:
            self.gradio.port = int(os.environ["GRADIO_PORT"])

    def _update_nested_settings(self):
        """Update nested settings with environment variables"""
        # Update Qdrant settings
        if "QDRANT_URL" in os.environ:
            self.qdrant.qdrant_url = os.environ["QDRANT_URL"]
        if "QDRANT_COLLECTION_NAME" in os.environ:
            self.qdrant.collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        if "QDRANT_API_KEY" in os.environ:
            self.qdrant.qdrant_api_key = os.environ["QDRANT_API_KEY"]
        if "QDRANT_DENSE_MODEL" in os.environ:
            self.qdrant.dense_model = os.environ["QDRANT_DENSE_MODEL"]
        if "QDRANT_DENSE_DIMENSION" in os.environ:
            self.qdrant.dense_dimension = int(os.environ["QDRANT_DENSE_DIMENSION"])
        if "QDRANT_SPARSE_MODEL" in os.environ:
            self.qdrant.sparse_model = os.environ["QDRANT_SPARSE_MODEL"]
        if "QDRANT_SPARSE_MAX_FEATURES" in os.environ:
            self.qdrant.sparse_max_features = int(
                os.environ["QDRANT_SPARSE_MAX_FEATURES"]
            )
        if "QDRANT_HYBRID_SEARCH_ALPHA" in os.environ:
            self.qdrant.hybrid_search_alpha = float(
                os.environ["QDRANT_HYBRID_SEARCH_ALPHA"]
            )
        if "QDRANT_HYBRID_SEARCH_LIMIT" in os.environ:
            self.qdrant.hybrid_search_limit = int(
                os.environ["QDRANT_HYBRID_SEARCH_LIMIT"]
            )
        if "QDRANT_IMAGE_COLLECTION_NAME" in os.environ:
            self.qdrant.image_collection_name = os.environ[
                "QDRANT_IMAGE_COLLECTION_NAME"
            ]
        if "QDRANT_CLIP_MODEL" in os.environ:
            self.qdrant.clip_model = os.environ["QDRANT_CLIP_MODEL"]
        if "QDRANT_CLIP_DIMENSION" in os.environ:
            self.qdrant.clip_dimension = int(os.environ["QDRANT_CLIP_DIMENSION"])
        if "QDRANT_CLIP_LOCAL" in os.environ:
            self.qdrant.clip_local = os.environ["QDRANT_CLIP_LOCAL"].lower() == "true"
        if "QDRANT_CLIP_DEVICE" in os.environ:
            self.qdrant.clip_device = os.environ["QDRANT_CLIP_DEVICE"]
        if "QDRANT_CLIP_OLLAMA_ENDPOINT" in os.environ:
            self.qdrant.clip_ollama_endpoint = os.environ["QDRANT_CLIP_OLLAMA_ENDPOINT"]
        if "QDRANT_DENSE_BASE_URL" in os.environ:
            self.qdrant.dense_base_url = os.environ["QDRANT_DENSE_BASE_URL"]
        if "QDRANT_DENSE_API_KEY" in os.environ:
            self.qdrant.dense_api_key = os.environ["QDRANT_DENSE_API_KEY"]

        # Update LLM settings
        if "STUDENT_MODEL" in os.environ:
            self.llm.student_model = os.environ["STUDENT_MODEL"]
        if "TEACHER_MODEL" in os.environ:
            self.llm.teacher_model = os.environ["TEACHER_MODEL"]
        if "GEMMA_BASE_URL" in os.environ:
            self.llm.gemma_base_url = os.environ["GEMMA_BASE_URL"]
        if "GEMMA_API_KEY" in os.environ:
            self.llm.gemma_api_key = os.environ["GEMMA_API_KEY"]

        # Update DSPy settings
        if "MAX_FULL_EVALS" in os.environ:
            self.dspy.max_full_evals = int(os.environ["MAX_FULL_EVALS"])
        if "NUM_THREADS" in os.environ:
            self.dspy.num_threads = int(os.environ["NUM_THREADS"])
        if "TRACK_STATS" in os.environ:
            self.dspy.track_stats = os.environ["TRACK_STATS"].lower() == "true"

        # Update Storage settings
        if "IMAGE_STORAGE_PATH" in os.environ:
            self.storage.image_storage_path = os.environ["IMAGE_STORAGE_PATH"]

        # Update App settings
        if "APP_HOST" in os.environ:
            self.app.host = os.environ["APP_HOST"]
        if "APP_PORT" in os.environ:
            self.app.port = int(os.environ["APP_PORT"])

        # Update Gradio settings
        if "GRADIO_PORT" in os.environ:
            self.gradio.port = int(os.environ["GRADIO_PORT"])


@lru_cache
def get_settings() -> Settings:
    return Settings()

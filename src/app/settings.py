import os
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    collection_name: str = "generic_rag_collection"
    qdrant_url: str = "http://10.84.0.7:6333"
    qdrant_api_key: str = ""
    
    model_config = {"env_prefix": "QDRANT_"}


class ColpaliSettings(BaseSettings):
    colpali_model_name: str = "vidore/colqwen2.5-v0.2"
    
    model_config = {"env_prefix": "COLPALI_"}


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
    port: int = 8000
    
    model_config = {"env_prefix": "APP_"}


class GradioSettings(BaseSettings):
    port: int = 7860
    
    model_config = {"env_prefix": "GRADIO_"}


class Settings(BaseSettings):
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    colpali: ColpaliSettings = Field(default_factory=ColpaliSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    dspy: DSPySettings = Field(default_factory=DSPySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    gradio: GradioSettings = Field(default_factory=GradioSettings)

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
            
        # Update Colpali settings only if environment variable is set
        if "COLPALI_MODEL_NAME" in os.environ:
            self.colpali.colpali_model_name = os.environ["COLPALI_MODEL_NAME"]
            
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
            
        # Update Colpali settings
        if "COLPALI_MODEL_NAME" in os.environ:
            self.colpali.colpali_model_name = os.environ["COLPALI_MODEL_NAME"]
            
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
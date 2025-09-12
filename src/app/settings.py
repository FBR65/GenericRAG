import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    collection_name: str = os.environ.get("QDRANT_COLLECTION_NAME", "generic_rag_collection")
    qdrant_url: str = os.environ.get("QDRANT_URL", "")
    qdrant_api_key: str = os.environ.get("QDRANT_API_KEY", "")


class ColpaliSettings(BaseSettings):
    colpali_model_name: str = "vidore/colqwen2.5-v0.2"


class LLMSettings(BaseSettings):
    # Student model configuration
    student_model: str = os.environ.get("STUDENT_MODEL", "")
    student_base_url: str = os.environ.get("STUDENT_BASE_URL", "")
    student_api_key: str = os.environ.get("STUDENT_API_KEY", "")
    
    # Teacher model configuration
    teacher_model: str = os.environ.get("TEACHER_MODEL", "")
    teacher_base_url: str = os.environ.get("TEACHER_BASE_URL", "")
    teacher_api_key: str = os.environ.get("TEACHER_API_KEY", "")


class DSPySettings(BaseSettings):
    max_full_evals: int = int(os.environ.get("DSPY_MAX_FULL_EVALS", "3"))
    num_threads: int = int(os.environ.get("DSPY_NUM_THREADS", "8"))
    track_stats: bool = os.environ.get("DSPY_TRACK_STATS", "true").lower() == "true"
    reflection_lm: str = os.environ.get("DSPY_REFLECTION_LM", "teacher")
    judge_lm: str = os.environ.get("DSPY_JUDGE_LM", "student")


class StorageSettings(BaseSettings):
    image_storage_path: Path = Path(os.environ.get("IMAGE_STORAGE_PATH", "./data/images"))
    temp_storage_path: Path = Path(os.environ.get("TEMP_STORAGE_PATH", "./data/temp"))
    dspy_cache_path: Path = Path(os.environ.get("DSPY_CACHE_PATH", "./data/dspy_cache"))


class AppSettings(BaseSettings):
    host: str = os.environ.get("APP_HOST", "0.0.0.0")
    port: int = int(os.environ.get("APP_PORT", "8000"))
    gradio_port: int = int(os.environ.get("GRADIO_PORT", "7860"))


class Settings(BaseSettings):
    qdrant: QdrantSettings = QdrantSettings()
    colpali: ColpaliSettings = ColpaliSettings()
    llm: LLMSettings = LLMSettings()
    dspy: DSPySettings = DSPySettings()
    storage: StorageSettings = StorageSettings()
    app: AppSettings = AppSettings()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.storage.image_storage_path.mkdir(parents=True, exist_ok=True)
        self.storage.temp_storage_path.mkdir(parents=True, exist_ok=True)
        self.storage.dspy_cache_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
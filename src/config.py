import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # System Configuration
    system_name: str = os.getenv("SYSTEM_NAME", "RAG System")
    system_version: str = os.getenv("SYSTEM_VERSION", "1.0.0")
    description: str = os.getenv("DESCRIPTION", "Generic RAG System for Document Processing")
    
    # Qdrant Configuration
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")
    
    # LLM Configuration
    llm_endpoint: str = os.getenv("LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "custom-model")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    
    # ColPali Configuration
    colpali_model_name: str = os.getenv("COLPALI_MODEL_NAME", "vidore/colqwen2-v1.0")
    colpali_device: str = os.getenv("COLPALI_DEVICE", "cuda:0")
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_workers: int = int(os.getenv("API_WORKERS", "1"))
    api_title: str = os.getenv("API_TITLE", "Generic RAG API")
    api_description: str = os.getenv("API_DESCRIPTION", "REST API for Generic RAG System")
    
    # File Processing Configuration
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    supported_formats: List[str] = os.getenv("SUPPORTED_FORMATS", ".pdf").split(",")
    image_dpi: int = int(os.getenv("IMAGE_DPI", "300"))
    image_quality: int = int(os.getenv("IMAGE_QUALITY", "95"))
    
    # Search Configuration
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "5"))
    
    # Frontend Configuration
    frontend_host: str = os.getenv("FRONTEND_HOST", "0.0.0.0")
    frontend_port: int = int(os.getenv("FRONTEND_PORT", "7860"))
    frontend_title: str = os.getenv("FRONTEND_TITLE", "Generic RAG System")
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE")
    
    # Processing Configuration
    temp_dir: str = os.getenv("TEMP_DIR", "temp")
    max_concurrent_uploads: int = int(os.getenv("MAX_CONCURRENT_UPLOADS", "2"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.config import Settings


class TestSettings:
    """Test cases for configuration settings."""
    
    def test_environment_variables(self):
        """Test that environment variables are properly read."""
        test_env = {
            "SYSTEM_NAME": "Test System",
            "QDRANT_HOST": "test.qdrant.com",
            "QDRANT_PORT": "1234",
            "LLM_MODEL": "test-model",
            "LLM_TEMPERATURE": "0.5",
            "API_PORT": "9000",
            "SEARCH_TOP_K": "10"
        }
        
        with patch.dict(os.environ, test_env):
            settings = Settings()
            
            assert settings.system_name == "Test System"
            assert settings.qdrant_host == "test.qdrant.com"
            assert settings.qdrant_port == 1234
            assert settings.llm_model == "test-model"
            assert settings.llm_temperature == 0.5
            assert settings.api_port == 9000
            assert settings.search_top_k == 10
    
    def test_supported_formats_parsing(self):
        """Test parsing of supported formats from environment."""
        with patch.dict(os.environ, {"SUPPORTED_FORMATS": ".pdf,.docx,.txt"}):
            settings = Settings()
            assert settings.supported_formats == [".pdf", ".docx", ".txt"]
    
    def test_logging_level_parsing(self):
        """Test parsing of logging level from environment."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            settings = Settings()
            assert settings.log_level == "DEBUG"
    
    def test_type_conversion(self):
        """Test proper type conversion of environment variables."""
        test_env = {
            "QDRANT_PORT": "6333",
            "LLM_TEMPERATURE": "0.8",
            "LLM_MAX_TOKENS": "2000",
            "API_PORT": "8001",
            "MAX_FILE_SIZE": "20971520",
            "IMAGE_DPI": "600",
            "IMAGE_QUALITY": "90",
            "SEARCH_TOP_K": "3",
            "FRONTEND_PORT": "7861",
            "API_WORKERS": "2",
            "MAX_CONCURRENT_UPLOADS": "4"
        }
        
        with patch.dict(os.environ, test_env):
            settings = Settings()
            
            # Test integer conversions
            assert isinstance(settings.qdrant_port, int)
            assert isinstance(settings.api_port, int)
            assert isinstance(settings.search_top_k, int)
            assert isinstance(settings.frontend_port, int)
            assert isinstance(settings.api_workers, int)
            assert isinstance(settings.max_concurrent_uploads, int)
            
            # Test float conversions
            assert isinstance(settings.llm_temperature, float)
            
            # Test specific values
            assert settings.qdrant_port == 6333
            assert settings.llm_temperature == 0.8
            assert settings.llm_max_tokens == 2000
            assert settings.api_port == 8001
            assert settings.max_file_size == 20971520
            assert settings.image_dpi == 600
            assert settings.image_quality == 90
            assert settings.search_top_k == 3
            assert settings.frontend_port == 7861
            assert settings.api_workers == 2
            assert settings.max_concurrent_uploads == 4
    
    def test_file_optional_settings(self):
        """Test that optional file settings work correctly."""
        with patch.dict(os.environ, {"LOG_FILE": "/var/log/rag_system.log"}):
            settings = Settings()
            assert settings.log_file == "/var/log/rag_system.log"
    
    def test_config_file_loading(self):
        """Test loading configuration from .env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("TEST_VAR=test_value\n")
            f.write("TEST_NUM=42\n")
            env_file = f.name
        
        try:
            with patch.dict(os.environ, {"ENV_FILE": env_file}):
                # This would test .env file loading if implemented
                pass
        finally:
            os.unlink(env_file)
    
    def test_default_settings(self):
        """Test default settings when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            # Test system settings
            assert settings.system_name == "Generic RAG System"
            assert settings.system_version == "1.0.0"
            assert settings.description == "Generic RAG System for Document Processing"
            
            # Test Qdrant settings
            assert settings.qdrant_host == "localhost"
            assert settings.qdrant_port == 6333
            assert settings.qdrant_collection_name == "rag_documents"
            assert settings.qdrant_api_key is None
            
            # Test LLM settings
            assert settings.llm_endpoint == "http://localhost:8000/v1/chat/completions"
            assert settings.llm_model == "custom-model"
            assert settings.llm_temperature == 0.7
            assert settings.llm_max_tokens == 1000
            
            # Test ColPali settings
            assert settings.colpali_model_name == "vidore/colqwen2-v1.0"
            assert settings.colpali_device == "cuda:0"
            
            # Test API settings
            assert settings.api_host == "0.0.0.0"
            assert settings.api_port == 8000
            assert settings.api_title == "Generic RAG API"
            assert settings.api_description == "REST API for Generic RAG System"
            
            # Test file processing settings
            assert settings.max_file_size == 10485760
            assert settings.supported_formats == [".pdf"]
            assert settings.image_dpi == 300
            assert settings.image_quality == 95
            
            # Test search settings
            assert settings.search_top_k == 5
            
            # Test frontend settings
            assert settings.frontend_host == "0.0.0.0"
            assert settings.frontend_port == 7860
            assert settings.frontend_title == "Generic RAG System"
            
            # Test logging settings
            assert settings.log_level == "INFO"
            assert settings.log_file is None
            
            # Test processing settings
            assert settings.temp_dir == "temp"
            assert settings.max_concurrent_uploads == 2
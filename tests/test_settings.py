"""
Tests for application settings
"""
import os
import pytest
from unittest.mock import patch

from src.app.settings import Settings, get_settings


class TestSettings:
    """Test application settings"""
    
    def test_settings_initialization(self):
        """Test settings initialization with default values"""
        with patch.dict(os.environ, {}):
            settings = Settings()
            
            # Test default values
            assert settings.app.host == "0.0.0.0"
            assert settings.app.port == 8000
            assert settings.gradio.port == 7860
            assert settings.qdrant.qdrant_url == "http://10.84.0.7:6333"
            assert settings.qdrant.collection_name == "generic_rag_collection"
            assert settings.colpali.colpali_model_name == "vidore/colqwen2.5-v0.2"
            assert settings.storage.image_storage_path == "./data/images"
            assert settings.storage.temp_storage_path == "./data/temp"
            assert settings.storage.dspy_cache_path == "./data/dspy_cache"
            assert settings.llm.student_model == "google/gemma-3-27b-it"
            assert settings.llm.teacher_model == "google/gemma-3-27b-it"
            assert settings.llm.gemma_base_url == "http://10.78.0.5:8114/v1"
            assert settings.dspy.max_full_evals == 3
            assert settings.dspy.num_threads == 4
            assert settings.dspy.track_stats is True
    
    def test_settings_with_environment_variables(self):
        """Test settings with environment variables"""
        env_vars = {
            "APP_HOST": "127.0.0.1",
            "APP_PORT": "9000",
            "GRADIO_PORT": "7861",
            "QDRANT_URL": "http://localhost:6333",
            "QDRANT_COLLECTION_NAME": "test_collection",
            "COLPALI_MODEL_NAME": "test/model",
            "IMAGE_STORAGE_PATH": "/custom/images",
            "GEMMA_BASE_URL": "http://localhost:8114/v1",
            "GEMMA_API_KEY": "test_key",
            "STUDENT_MODEL": "test/student",
            "TEACHER_MODEL": "test/teacher",
            "MAX_FULL_EVALS": "5",
            "NUM_THREADS": "8",
            "TRACK_STATS": "False",
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.app.host == "127.0.0.1"
            assert settings.app.port == 9000
            assert settings.gradio.port == 7861
            assert settings.qdrant.qdrant_url == "http://localhost:6333"
            assert settings.qdrant.collection_name == "test_collection"
            assert settings.colpali.colpali_model_name == "test/model"
            assert settings.storage.image_storage_path == "/custom/images"
            assert settings.llm.gemma_base_url == "http://localhost:8114/v1"
            assert settings.llm.gemma_api_key == "test_key"
            assert settings.llm.student_model == "test/student"
            assert settings.llm.teacher_model == "test/teacher"
            assert settings.dspy.max_full_evals == 5
            assert settings.dspy.num_threads == 8
            assert settings.dspy.track_stats is False
    
    def test_get_settings_caching(self):
        """Test that get_settings caches the result"""
        with patch.dict(os.environ, {}):
            settings1 = get_settings()
            settings2 = get_settings()
            
            assert settings1 is settings2


class TestQdrantSettings:
    """Test Qdrant-specific settings"""
    
    def test_qdrant_settings(self):
        """Test Qdrant settings"""
        with patch.dict(os.environ, {}):
            from src.app.settings import QdrantSettings
            
            settings = QdrantSettings()
            
            assert settings.collection_name == "generic_rag_collection"
            assert settings.qdrant_url == "http://10.84.0.7:6333"
            assert settings.qdrant_api_key == ""


class TestColpaliSettings:
    """Test ColPali-specific settings"""
    
    def test_colpali_settings(self):
        """Test ColPali settings"""
        with patch.dict(os.environ, {}):
            from src.app.settings import ColpaliSettings
            
            settings = ColpaliSettings()
            
            assert settings.colpali_model_name == "vidore/colqwen2.5-v0.2"


class TestStorageSettings:
    """Test storage-specific settings"""
    
    def test_storage_settings(self):
        """Test storage settings"""
        with patch.dict(os.environ, {}):
            from src.app.settings import StorageSettings
            
            settings = StorageSettings()
            
            assert settings.image_storage_path == "./data/images"
            assert settings.temp_storage_path == "./data/temp"
            assert settings.dspy_cache_path == "./data/dspy_cache"


class TestLLMSettings:
    """Test LLM-specific settings"""
    
    def test_llm_settings(self):
        """Test LLM settings"""
        with patch.dict(os.environ, {}):
            from src.app.settings import LLMSettings
            
            settings = LLMSettings()
            
            assert settings.student_model == "google/gemma-3-27b-it"
            assert settings.teacher_model == "google/gemma-3-27b-it"
            assert settings.gemma_base_url == "http://10.78.0.5:8114/v1"
            assert settings.gemma_api_key == "your_gemma_api_key_here"


class TestDSPySettings:
    """Test DSPy-specific settings"""
    
    def test_dspy_settings(self):
        """Test DSPy settings"""
        with patch.dict(os.environ, {}):
            from src.app.settings import DSPySettings
            
            settings = DSPySettings()
            
            assert settings.max_full_evals == 3
            assert settings.num_threads == 4
            assert settings.track_stats is True


if __name__ == "__main__":
    pytest.main([__file__])
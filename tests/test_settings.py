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
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(
                ("APP_", "GRADIO_", "QDRANT_", "COLPALI_", "LLM_", "IMAGE_", "DSPY_")
            ):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
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
        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value

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
            # Clear the cache to force fresh settings
            import src.app.settings

            src.app.settings.get_settings.cache_clear()

            # Force reload of settings to pick up environment variables
            import importlib

            importlib.reload(src.app.settings)
            from src.app.settings import Settings

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
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(("QDRANT_",)):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
            from src.app.settings import QdrantSettings

            settings = QdrantSettings()

            assert settings.collection_name == "generic_rag_collection"
            assert settings.qdrant_url == "http://10.84.0.7:6333"
            assert settings.qdrant_api_key == ""
        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value


class TestColpaliSettings:
    """Test ColPali-specific settings"""

    def test_colpali_settings(self):
        """Test ColPali settings"""
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(("COLPALI_",)):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
            from src.app.settings import ColpaliSettings

            settings = ColpaliSettings()

            assert settings.colpali_model_name == "vidore/colqwen2.5-v0.2"
        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value


class TestStorageSettings:
    """Test storage-specific settings"""

    def test_storage_settings(self):
        """Test storage settings"""
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(("IMAGE_", "DSPY_")):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
            from src.app.settings import StorageSettings

            settings = StorageSettings()

            assert settings.image_storage_path == "./data/images"
            assert settings.temp_storage_path == "./data/temp"
            assert settings.dspy_cache_path == "./data/dspy_cache"
        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value


class TestLLMSettings:
    """Test LLM-specific settings"""

    def test_llm_settings(self):
        """Test LLM settings"""
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(("LLM_", "TEACHER_", "STUDENT_", "GEMMA_")):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
            from src.app.settings import LLMSettings

            settings = LLMSettings()

            assert settings.student_model == "google/gemma-3-27b-it"
            assert settings.teacher_model == "google/gemma-3-27b-it"
            assert settings.student_base_url == "http://10.78.0.5:8114/v1"
            assert settings.teacher_base_url == "http://10.78.0.5:8114/v1"
            assert settings.gemma_base_url == "http://10.78.0.5:8114/v1"
            assert settings.gemma_api_key == "your_gemma_api_key_here"
        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value


class TestDSPySettings:
    """Test DSPy-specific settings"""

    def test_dspy_settings(self):
        """Test DSPy settings"""
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(("DSPY_",)):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
            from src.app.settings import DSPySettings

            settings = DSPySettings()

            assert settings.max_full_evals == 3
            assert settings.num_threads == 4
            assert settings.track_stats is True
        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value


class TestNewServiceSettings:
    """Test settings for new services"""

    def test_pdf_extractor_settings(self):
        """Test PDF extractor settings"""
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(("PDF_", "OUTPUT_", "IMAGE_")):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
            from src.app.settings import Settings

            settings = Settings()

            # Check that PDF extractor settings are accessible
            assert hasattr(settings, "storage")
            assert hasattr(settings.storage, "image_storage_path")
            assert hasattr(settings.storage, "temp_storage_path")
            assert hasattr(settings.storage, "dspy_cache_path")

            # Check default values
            assert settings.storage.image_storage_path == "./data/images"
            assert settings.storage.temp_storage_path == "./data/temp"
            assert settings.storage.dspy_cache_path == "./data/dspy_cache"

        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value

    def test_text_preprocessor_settings(self):
        """Test text preprocessor settings"""
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(("EMBEDDING_", "SPARSE_")):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
            from src.app.settings import Settings

            settings = Settings()

            # Check that embedding settings are accessible
            assert hasattr(settings, "qdrant")
            assert hasattr(settings.qdrant, "clip_model")
            assert hasattr(settings.qdrant, "clip_dimension")
            assert hasattr(settings.qdrant, "clip_ollama_endpoint")

            # Check default values
            assert settings.qdrant.clip_model == "clip-vit-base-patch32"
            assert settings.qdrant.clip_dimension == 512
            assert settings.qdrant.clip_ollama_endpoint == "http://localhost:11434"

        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value

    def test_vlm_service_settings(self):
        """Test VLM service settings"""
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(("VLM_", "OLLAMA_", "GEMMA_")):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
            from src.app.settings import Settings

            settings = Settings()

            # Check that LLM settings are accessible for VLM
            assert hasattr(settings, "llm")
            assert hasattr(settings.llm, "student_model")
            assert hasattr(settings.llm, "teacher_model")
            assert hasattr(settings.llm, "gemma_base_url")
            assert hasattr(settings.llm, "gemma_api_key")

            # Check default values
            assert settings.llm.student_model == "google/gemma-3-27b-it"
            assert settings.llm.teacher_model == "google/gemma-3-27b-it"
            assert settings.llm.gemma_base_url == "http://10.78.0.5:8114/v1"
            assert settings.llm.gemma_api_key == "your_gemma_api_key_here"

        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value

    def test_search_service_settings(self):
        """Test search service settings"""
        # Clear environment variables that might affect the test
        old_env = {}
        for key in list(os.environ.keys()):
            if key.startswith(("QDRANT_", "SEARCH_")):
                old_env[key] = os.environ[key]
                del os.environ[key]

        try:
            from src.app.settings import Settings

            settings = Settings()

            # Check that Qdrant settings are accessible for search
            assert hasattr(settings, "qdrant")
            assert hasattr(settings.qdrant, "qdrant_url")
            assert hasattr(settings.qdrant, "collection_name")

            # Check default values
            assert settings.qdrant.qdrant_url == "http://10.84.0.7:6333"
            assert settings.qdrant.collection_name == "generic_rag_collection"

        finally:
            # Restore environment variables
            for key, value in old_env.items():
                os.environ[key] = value


if __name__ == "__main__":
    pytest.main([__file__])

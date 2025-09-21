"""
Tests for image embedding service
"""

import pytest
import tempfile
import os
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from PIL import Image
import numpy as np
import torch

from src.app.services.image_embedding_service import ImageEmbeddingService


class TestImageEmbeddingService:
    """Test image embedding service"""

    @pytest.fixture
    def image_embedding_service(self):
        """Create image embedding service instance"""
        return ImageEmbeddingService(
            clip_model_name="clip-vit-base-patch32",
            clip_dimension=512,
            clip_ollama_endpoint="http://localhost:11434",
        )

    @pytest.fixture
    def mock_image(self):
        """Create mock image"""
        return Image.new("RGB", (224, 224), color="red")

    @pytest.fixture
    def mock_image_path(self, mock_image):
        """Create temporary image file"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            mock_image.save(temp_file.name)
            yield temp_file.name

        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    def test_image_embedding_service_initialization(self, image_embedding_service):
        """Test image embedding service initialization"""
        assert image_embedding_service.clip_model_name == "clip-vit-base-patch32"
        assert image_embedding_service.clip_dimension == 512
        assert image_embedding_service.clip_ollama_endpoint == "http://localhost:11434"
        assert image_embedding_service.logger is not None

    def test_load_image_success(self, image_embedding_service, mock_image_path):
        """Test successful image loading"""
        image = image_embedding_service.load_image(mock_image_path)

        assert image is not None
        assert isinstance(image, Image.Image)
        assert image.size == (224, 224)
        assert image.mode == "RGB"

    def test_load_image_file_not_found(self, image_embedding_service):
        """Test image loading with non-existent file"""
        with pytest.raises(Exception):
            image_embedding_service.load_image("non_existent_file.png")

    def test_load_image_invalid_format(self, image_embedding_service):
        """Test image loading with invalid format"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"This is not an image")
            temp_file_path = temp_file.name

        try:
            with pytest.raises(Exception):
                image_embedding_service.load_image(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_preprocess_image(self, image_embedding_service, mock_image):
        """Test image preprocessing"""
        processed_image = image_embedding_service.preprocess_image(mock_image)

        assert processed_image is not None
        assert isinstance(processed_image, Image.Image)
        assert processed_image.size == (224, 224)
        assert processed_image.mode == "RGB"

    def test_preprocess_image_resize(self, image_embedding_service, mock_image):
        """Test image preprocessing with resize"""
        # Create larger image
        large_image = Image.new("RGB", (512, 512), color="blue")
        processed_image = image_embedding_service.preprocess_image(
            large_image, size=224
        )

        assert processed_image.size == (224, 224)

    def test_preprocess_image_square_resize(self, image_embedding_service, mock_image):
        """Test image preprocessing with square aspect ratio"""
        # Create rectangular image
        rectangular_image = Image.new("RGB", (500, 300), color="green")
        processed_image = image_embedding_service.preprocess_image(
            rectangular_image, size=224
        )

        assert processed_image.size == (224, 224)

    def test_generate_clip_embedding_from_path_success(
        self, image_embedding_service, mock_image_path
    ):
        """Test successful CLIP embedding generation from image path"""
        with (
            patch("src.app.services.image_embedding_service.Image.open") as mock_open,
            patch(
                "src.app.services.image_embedding_service.ImageEmbeddingService.preprocess_image"
            ) as mock_preprocess,
            patch(
                "src.app.services.image_embedding_service.ImageEmbeddingService._get_clip_embedding"
            ) as mock_get_embedding,
        ):
            # Setup mocks
            mock_open.return_value = mock_image_path
            mock_preprocess.return_value = mock_image_path
            mock_get_embedding.return_value = [0.1] * 512

            embedding = image_embedding_service.generate_clip_embedding_from_path(
                mock_image_path
            )

            assert isinstance(embedding, list)
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)

    def test_generate_clip_embedding_from_path_error(self, image_embedding_service):
        """Test CLIP embedding generation with error"""
        with pytest.raises(Exception):
            image_embedding_service.generate_clip_embedding_from_path(
                "non_existent_file.png"
            )

    def test_generate_clip_embedding_from_image_success(
        self, image_embedding_service, mock_image
    ):
        """Test successful CLIP embedding generation from image object"""
        with (
            patch(
                "src.app.services.image_embedding_service.ImageEmbeddingService.preprocess_image"
            ) as mock_preprocess,
            patch(
                "src.app.services.image_embedding_service.ImageEmbeddingService._get_clip_embedding"
            ) as mock_get_embedding,
        ):
            # Setup mocks
            mock_preprocess.return_value = mock_image
            mock_get_embedding.return_value = [0.1] * 512

            embedding = image_embedding_service.generate_clip_embedding_from_image(
                mock_image
            )

            assert isinstance(embedding, list)
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)

    def test_generate_clip_embedding_from_image_error(self, image_embedding_service):
        """Test CLIP embedding generation with invalid image"""
        invalid_image = None
        with pytest.raises(Exception):
            image_embedding_service.generate_clip_embedding_from_image(invalid_image)

    def test_get_clip_embedding_success(self, image_embedding_service, mock_image):
        """Test successful CLIP embedding generation"""
        with patch(
            "src.app.services.image_embedding_service.ImageEmbeddingService._call_ollama_api"
        ) as mock_call_api:
            mock_call_api.return_value = [0.1] * 512

            embedding = image_embedding_service._get_clip_embedding(mock_image)

            assert isinstance(embedding, list)
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)

    def test_get_clip_embedding_error(self, image_embedding_service, mock_image):
        """Test CLIP embedding generation with error"""
        with patch(
            "src.app.services.image_embedding_service.ImageEmbeddingService._call_ollama_api"
        ) as mock_call_api:
            mock_call_api.side_effect = Exception("API Error")

            with pytest.raises(Exception):
                image_embedding_service._get_clip_embedding(mock_image)

    @pytest.mark.asyncio
    async def test_call_ollama_api_success(self, image_embedding_service, mock_image):
        """Test successful Ollama API call"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            embedding = await image_embedding_service._call_ollama_api(mock_image)

            assert isinstance(embedding, list)
            assert len(embedding) == 5
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_call_ollama_api_error(self, image_embedding_service, mock_image):
        """Test Ollama API call with error"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception):
                await image_embedding_service._call_ollama_api(mock_image)

    def test_validate_embedding_dimension(self, image_embedding_service):
        """Test embedding dimension validation"""
        # Valid embedding
        valid_embedding = [0.1] * 512
        assert (
            image_embedding_service._validate_embedding_dimension(valid_embedding)
            is True
        )

        # Invalid embedding - wrong dimension
        invalid_embedding = [0.1] * 256
        assert (
            image_embedding_service._validate_embedding_dimension(invalid_embedding)
            is False
        )

        # Invalid embedding - wrong type
        invalid_embedding = [0.1, 0.2, "invalid", 0.4]
        assert (
            image_embedding_service._validate_embedding_dimension(invalid_embedding)
            is False
        )

    def test_normalize_embedding(self, image_embedding_service):
        """Test embedding normalization"""
        test_embedding = [1.0, -1.0, 2.0, -2.0, 0.0]
        normalized = image_embedding_service._normalize_embedding(test_embedding)

        assert isinstance(normalized, list)
        assert len(normalized) == len(test_embedding)

        # Check that values are between -1 and 1
        for value in normalized:
            assert -1.0 <= value <= 1.0

    def test_normalize_embedding_zero_vector(self, image_embedding_service):
        """Test embedding normalization with zero vector"""
        zero_embedding = [0.0] * 512
        normalized = image_embedding_service._normalize_embedding(zero_embedding)

        assert isinstance(normalized, list)
        assert len(normalized) == len(zero_embedding)
        assert all(value == 0.0 for value in normalized)

    def test_calculate_similarity_cosine(self, image_embedding_service):
        """Test cosine similarity calculation"""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]  # Same as embedding1
        embedding3 = [0.0, 1.0, 0.0]  # Different

        similarity_same = image_embedding_service.calculate_similarity(
            embedding1, embedding2
        )
        similarity_different = image_embedding_service.calculate_similarity(
            embedding1, embedding3
        )

        assert isinstance(similarity_same, float)
        assert isinstance(similarity_different, float)
        assert similarity_same == 1.0  # Perfect match
        assert similarity_different == 0.0  # No match

    def test_calculate_similarity_euclidean(self, image_embedding_service):
        """Test Euclidean distance calculation"""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]  # Same as embedding1
        embedding3 = [0.0, 1.0, 0.0]  # Different

        distance_same = image_embedding_service.calculate_similarity(
            embedding1, embedding2, metric="euclidean"
        )
        distance_different = image_embedding_service.calculate_similarity(
            embedding1, embedding3, metric="euclidean"
        )

        assert isinstance(distance_same, float)
        assert isinstance(distance_different, float)
        assert distance_same == 0.0  # Perfect match
        assert distance_different > 0.0  # Different embeddings

    def test_calculate_similarity_invalid_metric(self, image_embedding_service):
        """Test similarity calculation with invalid metric"""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]

        with pytest.raises(ValueError):
            image_embedding_service.calculate_similarity(
                embedding1, embedding2, metric="invalid_metric"
            )

    def test_calculate_similarity_invalid_dimensions(self, image_embedding_service):
        """Test similarity calculation with invalid dimensions"""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0]  # Different dimension

        with pytest.raises(ValueError):
            image_embedding_service.calculate_similarity(embedding1, embedding2)


class TestImageEmbeddingServiceIntegration:
    """Integration tests for image embedding service"""

    @pytest.fixture
    def test_images(self):
        """Create test images for integration testing"""
        images = []

        # Create different test images
        test_images = [
            ("red_image", (224, 224), "red"),
            ("blue_image", (224, 224), "blue"),
            ("green_image", (224, 224), "green"),
            ("large_image", (512, 512), "yellow"),
            ("rectangular_image", (300, 200), "purple"),
        ]

        for name, size, color in test_images:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                image = Image.new("RGB", size, color=color)
                image.save(temp_file.name)
                images.append((name, temp_file.name))

        yield images

        # Cleanup
        for name, path in images:
            if os.path.exists(path):
                os.unlink(path)

    def test_full_embedding_workflow(self, image_embedding_service, test_images):
        """Test complete embedding generation workflow"""
        embeddings = []

        for name, image_path in test_images:
            try:
                embedding = image_embedding_service.generate_clip_embedding_from_path(
                    image_path
                )
                embeddings.append((name, embedding))

                # Validate embedding
                assert isinstance(embedding, list)
                assert len(embedding) == 512
                assert all(isinstance(x, float) for x in embedding)

            except Exception as e:
                pytest.fail(f"Failed to generate embedding for {name}: {e}")

        # Test similarity calculations
        if len(embeddings) >= 2:
            name1, embedding1 = embeddings[0]
            name2, embedding2 = embeddings[1]

            similarity = image_embedding_service.calculate_similarity(
                embedding1, embedding2
            )
            assert isinstance(similarity, float)
            assert 0.0 <= similarity <= 1.0

    def test_batch_embedding_generation(self, image_embedding_service, test_images):
        """Test batch embedding generation"""
        image_paths = [path for _, path in test_images]

        embeddings = []
        for image_path in image_paths:
            try:
                embedding = image_embedding_service.generate_clip_embedding_from_path(
                    image_path
                )
                embeddings.append(embedding)
            except Exception:
                # Skip failed embeddings for batch testing
                continue

        # Verify we got some embeddings
        assert len(embeddings) > 0

        # Test similarity matrix
        similarity_matrix = []
        for i, embedding1 in enumerate(embeddings):
            row = []
            for j, embedding2 in enumerate(embeddings):
                similarity = image_embedding_service.calculate_similarity(
                    embedding1, embedding2
                )
                row.append(similarity)
            similarity_matrix.append(row)

        # Verify matrix properties
        assert len(similarity_matrix) == len(embeddings)
        for row in similarity_matrix:
            assert len(row) == len(embeddings)
            for similarity in row:
                assert 0.0 <= similarity <= 1.0

    def test_image_preprocessing_pipeline(self, image_embedding_service, test_images):
        """Test complete image preprocessing pipeline"""
        for name, image_path in test_images:
            try:
                # Load image
                original_image = image_embedding_service.load_image(image_path)

                # Preprocess image
                processed_image = image_embedding_service.preprocess_image(
                    original_image
                )

                # Generate embedding
                embedding = image_embedding_service.generate_clip_embedding_from_image(
                    processed_image
                )

                # Validate results
                assert isinstance(processed_image, Image.Image)
                assert isinstance(embedding, list)
                assert len(embedding) == 512

            except Exception as e:
                pytest.fail(f"Failed preprocessing pipeline for {name}: {e}")

    @pytest.mark.asyncio
    async def test_async_embedding_generation(
        self, image_embedding_service, test_images
    ):
        """Test async embedding generation"""

        async def generate_embedding_async(image_path):
            return await image_embedding_service._call_ollama_api(
                image_embedding_service.load_image(image_path)
            )

        # Test async generation for first few images
        async_tasks = []
        for _, image_path in test_images[:3]:  # Test first 3 images
            task = generate_embedding_async(image_path)
            async_tasks.append(task)

        embeddings = await asyncio.gather(*async_tasks, return_exceptions=True)

        # Verify results
        successful_embeddings = [e for e in embeddings if isinstance(e, list)]
        assert len(successful_embeddings) > 0

        for embedding in successful_embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)


if __name__ == "__main__":
    pytest.main([__file__])

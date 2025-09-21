"""
Tests for VLM (Vision-Language Model) service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from PIL import Image
import tempfile
import os
from typing import List, Dict, Any

from src.app.services.vlm_service import VLMService
from src.app.models.schemas import VLMResponse, VLMRequest


class TestVLMService:
    """Test VLM service"""

    @pytest.fixture
    def vlm_service(self):
        """Create VLM service instance"""
        return VLMService(
            model_name="llava-1.6-vicuna-7b",
            api_url="http://localhost:11434/api/generate",
            max_tokens=512,
            temperature=0.7,
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

    @pytest.fixture
    def mock_vlm_request(self):
        """Create mock VLM request"""
        return VLMRequest(
            prompt="What is shown in this image?",
            image_path="test_image.png",
            context="This is a test image for VLM analysis.",
        )

    @pytest.fixture
    def mock_vlm_response(self):
        """Create mock VLM response"""
        return VLMResponse(
            response="The image shows a red square.",
            confidence_score=0.95,
            processing_time=1.5,
            model_used="llava-1.6-vicuna-7b",
        )

    def test_vlm_service_initialization(self, vlm_service):
        """Test VLM service initialization"""
        assert vlm_service.model_name == "llava-1.6-vicuna-7b"
        assert vlm_service.api_url == "http://localhost:11434/api/generate"
        assert vlm_service.max_tokens == 512
        assert vlm_service.temperature == 0.7
        assert vlm_service.logger is not None

    def test_validate_image_path_success(self, vlm_service, mock_image_path):
        """Test successful image path validation"""
        is_valid = vlm_service._validate_image_path(mock_image_path)

        assert is_valid is True

    def test_validate_image_path_file_not_found(self, vlm_service):
        """Test image path validation with non-existent file"""
        is_valid = vlm_service._validate_image_path("non_existent_file.png")

        assert is_valid is False

    def test_validate_image_path_invalid_format(self, vlm_service):
        """Test image path validation with invalid format"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"This is not an image")
            temp_file_path = temp_file.name

        try:
            is_valid = vlm_service._validate_image_path(temp_file_path)
            assert is_valid is False
        finally:
            os.unlink(temp_file_path)

    def test_validate_image_path_invalid_extension(self, vlm_service):
        """Test image path validation with invalid extension"""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as temp_file:
            temp_file.write(b"fake image content")
            temp_file_path = temp_file.name

        try:
            is_valid = vlm_service._validate_image_path(temp_file_path)
            assert is_valid is False
        finally:
            os.unlink(temp_file_path)

    def test_validate_prompt_success(self, vlm_service):
        """Test successful prompt validation"""
        valid_prompts = [
            "What is shown in this image?",
            "Describe the scene in detail.",
            "What objects are visible?",
            "Analyze the image content.",
        ]

        for prompt in valid_prompts:
            is_valid = vlm_service._validate_prompt(prompt)
            assert is_valid is True

    def test_validate_prompt_empty(self, vlm_service):
        """Test prompt validation with empty prompt"""
        is_valid = vlm_service._validate_prompt("")
        assert is_valid is False

    def test_validate_prompt_too_long(self, vlm_service):
        """Test prompt validation with too long prompt"""
        long_prompt = "x" * 1001  # Exceeds typical limit
        is_valid = vlm_service._validate_prompt(long_prompt)
        assert is_valid is False

    def test_validate_prompt_invalid_characters(self, vlm_service):
        """Test prompt validation with invalid characters"""
        invalid_prompts = [None, 123, [], {}]

        for prompt in invalid_prompts:
            is_valid = vlm_service._validate_prompt(prompt)
            assert is_valid is False

    def test_validate_request_success(self, vlm_service, mock_vlm_request):
        """Test successful request validation"""
        is_valid = vlm_service._validate_request(mock_vlm_request)
        assert is_valid is True

    def test_validate_request_invalid(self, vlm_service):
        """Test request validation with invalid request"""
        invalid_requests = [None, "not_a_request", 123, [], {}]

        for request in invalid_requests:
            is_valid = vlm_service._validate_request(request)
            assert is_valid is False

    def test_validate_request_missing_fields(self, vlm_service):
        """Test request validation with missing fields"""
        invalid_requests = [
            VLMRequest(prompt="", image_path="test.png", context=""),
            VLMRequest(prompt="test", image_path="", context=""),
            VLMRequest(prompt="test", image_path="test.png", context=""),
        ]

        for request in invalid_requests:
            is_valid = vlm_service._validate_request(request)
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_process_image_success(self, vlm_service, mock_image_path):
        """Test successful image processing"""
        with (
            patch("PIL.Image.open") as mock_open,
            patch(
                "src.app.services.vlm_service.VLMService._validate_image_path"
            ) as mock_validate,
        ):
            # Setup mocks
            mock_open.return_value = mock_image_path
            mock_validate.return_value = True

            # Test processing
            result = await vlm_service.process_image(mock_image_path)

            assert result is not None
            assert isinstance(result, dict)
            assert "image_data" in result
            assert "image_size" in result
            assert "image_mode" in result

    @pytest.mark.asyncio
    async def test_process_image_invalid_path(self, vlm_service):
        """Test image processing with invalid path"""
        with pytest.raises(Exception):
            await vlm_service.process_image("non_existent_file.png")

    @pytest.mark.asyncio
    async def test_process_image_error(self, vlm_service, mock_image_path):
        """Test image processing with error"""
        with patch("PIL.Image.open") as mock_open:
            mock_open.side_effect = Exception("Image processing failed")

            with pytest.raises(Exception):
                await vlm_service.process_image(mock_image_path)

    @pytest.mark.asyncio
    async def test_generate_response_success(self, vlm_service, mock_vlm_request):
        """Test successful response generation"""
        with patch("aiohttp.ClientSession") as mock_session:
            # Mock successful API response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "response": "The image shows a red square.",
                "confidence": 0.95,
                "processing_time": 1.5,
            }

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            # Test response generation
            result = await vlm_service.generate_response(mock_vlm_request)

            assert isinstance(result, VLMResponse)
            assert result.response == "The image shows a red square."
            assert result.confidence_score == 0.95
            assert result.processing_time == 1.5
            assert result.model_used == "llava-1.6-vicuna-7b"

    @pytest.mark.asyncio
    async def test_generate_response_api_error(self, vlm_service, mock_vlm_request):
        """Test response generation with API error"""
        with patch("aiohttp.ClientSession") as mock_session:
            # Mock API error
            mock_response = Mock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception):
                await vlm_service.generate_response(mock_vlm_request)

    @pytest.mark.asyncio
    async def test_generate_response_invalid_request(self, vlm_service):
        """Test response generation with invalid request"""
        with pytest.raises(Exception):
            await vlm_service.generate_response(None)

    @pytest.mark.asyncio
    async def test_analyze_image_success(
        self, vlm_service, mock_vlm_request, mock_vlm_response
    ):
        """Test successful image analysis"""
        with patch(
            "src.app.services.vlm_service.VLMService.generate_response"
        ) as mock_generate:
            mock_generate.return_value = mock_vlm_response

            # Test analysis
            result = await vlm_service.analyze_image(mock_vlm_request)

            assert isinstance(result, VLMResponse)
            assert result.response == "The image shows a red square."
            assert result.confidence_score == 0.95
            assert result.processing_time == 1.5
            assert result.model_used == "llava-1.6-vicuna-7b"

    @pytest.mark.asyncio
    async def test_analyze_image_error(self, vlm_service, mock_vlm_request):
        """Test image analysis with error"""
        with patch(
            "src.app.services.vlm_service.VLMService.generate_response"
        ) as mock_generate:
            mock_generate.side_effect = Exception("Analysis failed")

            with pytest.raises(Exception):
                await vlm_service.analyze_image(mock_vlm_request)

    @pytest.mark.asyncio
    async def test_batch_analyze_images_success(
        self, vlm_service, mock_vlm_request, mock_vlm_response
    ):
        """Test successful batch image analysis"""
        requests = [mock_vlm_request for _ in range(3)]

        with patch(
            "src.app.services.vlm_service.VLMService.generate_response"
        ) as mock_generate:
            mock_generate.return_value = mock_vlm_response

            # Test batch analysis
            results = await vlm_service.batch_analyze_images(requests)

            assert isinstance(results, list)
            assert len(results) == 3

            for result in results:
                assert isinstance(result, VLMResponse)
                assert result.response == "The image shows a red square."
                assert result.confidence_score == 0.95
                assert result.processing_time == 1.5
                assert result.model_used == "llava-1.6-vicuna-7b"

    @pytest.mark.asyncio
    async def test_batch_analyze_images_mixed_results(
        self, vlm_service, mock_vlm_request, mock_vlm_response
    ):
        """Test batch image analysis with mixed results"""
        requests = [mock_vlm_request for _ in range(3)]

        with patch(
            "src.app.services.vlm_service.VLMService.generate_response"
        ) as mock_generate:
            # Mock mixed results
            mock_generate.side_effect = [
                mock_vlm_response,
                Exception("Analysis failed"),
                mock_vlm_response,
            ]

            # Test batch analysis
            results = await vlm_service.batch_analyze_images(requests)

            assert isinstance(results, list)
            assert len(results) == 3

            # Check successful results
            successful_results = [r for r in results if isinstance(r, VLMResponse)]
            assert len(successful_results) == 2

            # Check failed results
            failed_results = [r for r in results if isinstance(r, Exception)]
            assert len(failed_results) == 1

    @pytest.mark.asyncio
    async def test_batch_analyze_images_empty_requests(self, vlm_service):
        """Test batch image analysis with empty requests"""
        results = await vlm_service.batch_analyze_images([])

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_batch_analyze_images_invalid_requests(self, vlm_service):
        """Test batch image analysis with invalid requests"""
        invalid_requests = [None, "not_a_request", 123]

        with patch(
            "src.app.services.vlm_service.VLMService.generate_response"
        ) as mock_generate:
            mock_generate.return_value = VLMResponse(
                response="Test response",
                confidence_score=0.95,
                processing_time=1.5,
                model_used="llava-1.6-vicuna-7b",
            )

            results = await vlm_service.batch_analyze_images(invalid_requests)

            assert isinstance(results, list)
            assert len(results) == 3

            # All results should be exceptions due to invalid requests
            for result in results:
                assert isinstance(result, Exception)

    def test_format_prompt_success(self, vlm_service):
        """Test successful prompt formatting"""
        prompt = "What is shown in this image?"
        context = "This is a test image for analysis."

        formatted_prompt = vlm_service._format_prompt(prompt, context)

        assert isinstance(formatted_prompt, str)
        assert prompt in formatted_prompt
        assert context in formatted_prompt
        assert "Please analyze the image" in formatted_prompt

    def test_format_prompt_empty_context(self, vlm_service):
        """Test prompt formatting with empty context"""
        prompt = "What is shown in this image?"
        context = ""

        formatted_prompt = vlm_service._format_prompt(prompt, context)

        assert isinstance(formatted_prompt, str)
        assert prompt in formatted_prompt
        assert "Please analyze the image" in formatted_prompt

    def test_format_prompt_empty_prompt(self, vlm_service):
        """Test prompt formatting with empty prompt"""
        prompt = ""
        context = "This is a test image for analysis."

        formatted_prompt = vlm_service._format_prompt(prompt, context)

        assert isinstance(formatted_prompt, str)
        assert context in formatted_prompt
        assert "Please analyze the image" in formatted_prompt

    def test_format_prompt_both_empty(self, vlm_service):
        """Test prompt formatting with both empty"""
        prompt = ""
        context = ""

        formatted_prompt = vlm_service._format_prompt(prompt, context)

        assert isinstance(formatted_prompt, str)
        assert "Please analyze the image" in formatted_prompt

    def test_format_prompt_invalid_inputs(self, vlm_service):
        """Test prompt formatting with invalid inputs"""
        invalid_inputs = [
            (None, "context"),
            ("prompt", None),
            (None, None),
            (123, "context"),
            ("prompt", 123),
        ]

        for prompt, context in invalid_inputs:
            formatted_prompt = vlm_service._format_prompt(prompt, context)
            assert isinstance(formatted_prompt, str)

    def test_validate_response_success(self, vlm_service, mock_vlm_response):
        """Test successful response validation"""
        is_valid = vlm_service._validate_response(mock_vlm_response)

        assert is_valid is True

    def test_validate_response_invalid(self, vlm_service):
        """Test response validation with invalid response"""
        invalid_responses = [
            None,
            "not_a_response",
            123,
            [],
            {},
            VLMResponse(
                response="",
                confidence_score=0.95,
                processing_time=1.5,
                model_used="llava-1.6-vicuna-7b",
            ),
            VLMResponse(
                response="Test response",
                confidence_score=-1.0,  # Invalid confidence
                processing_time=1.5,
                model_used="llava-1.6-vicuna-7b",
            ),
            VLMResponse(
                response="Test response",
                confidence_score=0.95,
                processing_time=-1.0,  # Invalid processing time
                model_used="llava-1.6-vicuna-7b",
            ),
        ]

        for response in invalid_responses:
            is_valid = vlm_service._validate_response(response)
            assert is_valid is False

    def test_calculate_confidence_score(self, vlm_service):
        """Test confidence score calculation"""
        # Test with valid scores
        test_scores = [0.8, 0.9, 0.95, 1.0]
        for score in test_scores:
            calculated_score = vlm_service._calculate_confidence_score(score)
            assert 0.0 <= calculated_score <= 1.0

        # Test with invalid scores
        invalid_scores = [-1.0, 1.5, "invalid", None]
        for score in invalid_scores:
            calculated_score = vlm_service._calculate_confidence_score(score)
            assert 0.0 <= calculated_score <= 1.0

    def test_extract_key_information(self, vlm_service):
        """Test key information extraction"""
        response_text = "The image shows a red square with some text. There are also some blue circles in the background."

        key_info = vlm_service._extract_key_information(response_text)

        assert isinstance(key_info, dict)
        assert "objects" in key_info
        assert "colors" in key_info
        assert "shapes" in key_info

        # Check extracted information
        assert "red square" in key_info["objects"]
        assert "blue circles" in key_info["objects"]
        assert "red" in key_info["colors"]
        assert "blue" in key_info["colors"]
        assert "square" in key_info["shapes"]
        assert "circles" in key_info["shapes"]

    def test_extract_key_information_empty(self, vlm_service):
        """Test key information extraction with empty response"""
        key_info = vlm_service._extract_key_information("")

        assert isinstance(key_info, dict)
        assert "objects" in key_info
        assert "colors" in key_info
        assert "shapes" in key_info
        assert len(key_info["objects"]) == 0
        assert len(key_info["colors"]) == 0
        assert len(key_info["shapes"]) == 0

    def test_extract_key_information_invalid(self, vlm_service):
        """Test key information extraction with invalid input"""
        invalid_inputs = [None, 123, [], {}]

        for input_text in invalid_inputs:
            key_info = vlm_service._extract_key_information(input_text)
            assert isinstance(key_info, dict)
            assert "objects" in key_info
            assert "colors" in key_info
            assert "shapes" in key_info


class TestVLMServiceIntegration:
    """Integration tests for VLM service"""

    @pytest.fixture
    def complex_vlm_request(self):
        """Create complex VLM request for integration testing"""
        return VLMRequest(
            prompt="Analyze this image in detail and describe what you see. Include information about objects, colors, shapes, and any text present.",
            image_path="test_image.png",
            context="This is a complex image with multiple elements for detailed analysis.",
        )

    @pytest.fixture
    def multiple_vlm_requests(self):
        """Create multiple VLM requests for batch testing"""
        requests = []
        for i in range(5):
            request = VLMRequest(
                prompt=f"Analyze image {i + 1} and describe what you see.",
                image_path=f"test_image_{i + 1}.png",
                context=f"This is test image {i + 1} for batch analysis.",
            )
            requests.append(request)
        return requests

    def test_full_vlm_workflow(
        self, vlm_service, complex_vlm_request, mock_vlm_response
    ):
        """Test complete VLM workflow"""
        with patch(
            "src.app.services.vlm_service.VLMService.generate_response"
        ) as mock_generate:
            mock_generate.return_value = mock_vlm_response

            # Test complete workflow
            result = await vlm_service.analyze_image(complex_vlm_request)

            # Validate result
            assert isinstance(result, VLMResponse)
            assert result.response == "The image shows a red square."
            assert result.confidence_score == 0.95
            assert result.processing_time == 1.5
            assert result.model_used == "llava-1.6-vicuna-7b"

            # Test key information extraction
            key_info = vlm_service._extract_key_information(result.response)
            assert isinstance(key_info, dict)
            assert "objects" in key_info
            assert "colors" in key_info
            assert "shapes" in key_info

    def test_batch_vlm_workflow(
        self, vlm_service, multiple_vlm_requests, mock_vlm_response
    ):
        """Test complete batch VLM workflow"""
        with patch(
            "src.app.services.vlm_service.VLMService.generate_response"
        ) as mock_generate:
            mock_generate.return_value = mock_vlm_response

            # Test batch workflow
            results = await vlm_service.batch_analyze_images(multiple_vlm_requests)

            # Validate results
            assert isinstance(results, list)
            assert len(results) == len(multiple_vlm_requests)

            for i, result in enumerate(results):
                assert isinstance(result, VLMResponse)
                assert result.response == "The image shows a red square."
                assert result.confidence_score == 0.95
                assert result.processing_time == 1.5
                assert result.model_used == "llava-1.6-vicuna-7b"

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, vlm_service, complex_vlm_request):
        """Test error handling in VLM workflow"""
        with patch(
            "src.app.services.vlm_service.VLMService.generate_response"
        ) as mock_generate:
            # Mock different error scenarios
            mock_generate.side_effect = [
                Exception("API Error"),
                Exception("Timeout Error"),
                Exception("Invalid Request Error"),
            ]

            # Test error handling
            with pytest.raises(Exception):
                await vlm_service.analyze_image(complex_vlm_request)

    def test_prompt_formatting_workflow(self, vlm_service):
        """Test prompt formatting workflow"""
        test_cases = [
            (
                "What is this?",
                "Context about the image.",
                "What is this? Context about the image. Please analyze the image:",
            ),
            (
                "Describe the scene.",
                "",
                "Describe the scene. Please analyze the image:",
            ),
            ("", "Context only.", "Context only. Please analyze the image:"),
            ("", "", "Please analyze the image:"),
        ]

        for prompt, context, expected in test_cases:
            formatted = vlm_service._format_prompt(prompt, context)
            assert isinstance(formatted, str)
            assert len(formatted) > 0

    def test_confidence_score_workflow(self, vlm_service):
        """Test confidence score calculation workflow"""
        test_scores = [0.0, 0.25, 0.5, 0.75, 1.0, -1.0, 1.5, "invalid", None]

        for score in test_scores:
            calculated_score = vlm_service._calculate_confidence_score(score)
            assert 0.0 <= calculated_score <= 1.0

    def test_key_information_extraction_workflow(self, vlm_service):
        """Test key information extraction workflow"""
        test_responses = [
            "The image shows a red car and a blue house.",
            "There are three green trees and one yellow sun.",
            "A black cat is sitting on a white mat.",
            "Multiple colorful objects are scattered across the image.",
            "",
        ]

        for response in test_responses:
            key_info = vlm_service._extract_key_information(response)
            assert isinstance(key_info, dict)
            assert "objects" in key_info
            assert "colors" in key_info
            assert "shapes" in key_info
            assert isinstance(key_info["objects"], list)
            assert isinstance(key_info["colors"], list)
            assert isinstance(key_info["shapes"], list)

    @pytest.mark.asyncio
    async def test_concurrent_vlm_operations(
        self, vlm_service, multiple_vlm_requests, mock_vlm_response
    ):
        """Test concurrent VLM operations"""
        with patch(
            "src.app.services.vlm_service.VLMService.generate_response"
        ) as mock_generate:
            mock_generate.return_value = mock_vlm_response

            # Test concurrent operations
            tasks = [
                vlm_service.analyze_image(request) for request in multiple_vlm_requests
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify results
            successful_results = [r for r in results if isinstance(r, VLMResponse)]
            assert len(successful_results) == len(multiple_vlm_requests)

            for result in successful_results:
                assert isinstance(result, VLMResponse)
                assert result.response == "The image shows a red square."
                assert result.confidence_score == 0.95
                assert result.processing_time == 1.5
                assert result.model_used == "llava-1.6-vicuna-7b"


if __name__ == "__main__":
    pytest.main([__file__])

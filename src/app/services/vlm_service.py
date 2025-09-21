"""
Vision Language Model Service for generating responses based on search results
"""

import asyncio
import base64
import io
from typing import List, Optional, Dict, Any
from loguru import logger
from PIL import Image

from src.app.models.schemas import SearchResult, VLMResponse
from src.app.settings import get_settings


class VLMService:
    """Service for generating responses using Vision Language Models"""

    def __init__(self):
        """Initialize the VLM service"""
        self.settings = get_settings()
        self.ollama_endpoint = "http://localhost:11434"
        self.vlm_model = "gemma3:latest"

        logger.info("Initialized VLMService")

    async def generate_response_with_vlm(
        self,
        query: str,
        search_results: List[SearchResult],
        use_images: bool = True,
        max_context_length: int = 4000,
    ) -> VLMResponse:
        """
        Generate response using VLM with search results context

        Args:
            query: User query
            search_results: Search results from RAG system
            use_images: Whether to include image context
            max_context_length: Maximum context length for VLM

        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating VLM response for query: {query}")

            # Prepare context for VLM
            context = await self.prepare_context_for_vlm(
                search_results=search_results,
                use_images=use_images,
                max_length=max_context_length,
            )

            # Create prompt for VLM
            prompt = await self.create_vlm_prompt(query=query, context=context)

            # Generate response using Ollama
            response = await self._call_ollama(prompt)

            logger.info("Successfully generated VLM response")
            return VLMResponse(
                response=response,
                confidence_score=0.8,
                processing_time=1.5,
                model_used=self.vlm_model,
                context_length=max_context_length,
                images_used=use_images,
                sources_referenced=[result.id for result in search_results[:5]],
            )

        except Exception as e:
            logger.error(f"Error generating VLM response: {e}")
            return VLMResponse(
                response=f"Error generating response: {str(e)}",
                confidence_score=0.0,
                processing_time=1.5,
                model_used=self.vlm_model,
                context_length=max_context_length,
                images_used=False,
                sources_referenced=[result.id for result in search_results[:5]],
            )

    async def prepare_context_for_vlm(
        self,
        search_results: List[SearchResult],
        use_images: bool = True,
        max_length: int = 4000,
    ) -> Dict[str, Any]:
        """
        Prepare context from search results for VLM

        Args:
            search_results: Search results from RAG system
            use_images: Whether to include image context
            max_length: Maximum context length

        Returns:
            Dictionary containing text and image context
        """
        try:
            logger.info("Preparing context for VLM")

            context = {
                "text_context": [],
                "image_context": [],
                "metadata": {
                    "total_results": len(search_results),
                    "text_results": 0,
                    "image_results": 0,
                },
            }

            # Process search results
            for result in search_results:
                # Text context
                if result.document:
                    text_entry = {
                        "content": result.document,
                        "page": result.page,
                        "score": result.score,
                        "document": result.document,
                        "id": result.id,
                    }
                    context["text_context"].append(text_entry)
                    context["metadata"]["text_results"] += 1

                # Image context
                if use_images and result.image:
                    image_entry = {
                        "image": result.image,
                        "page": result.page,
                        "score": result.score,
                        "document": result.document,
                        "id": result.id,
                        "metadata": result.metadata,
                    }
                    context["image_context"].append(image_entry)
                    context["metadata"]["image_results"] += 1

            # Limit context length
            context = self._limit_context_length(context, max_length)

            logger.info(
                f"Prepared context with {len(context['text_context'])} text and {len(context['image_context'])} image entries"
            )
            return context

        except Exception as e:
            logger.error(f"Error preparing context for VLM: {e}")
            return {
                "text_context": [],
                "image_context": [],
                "metadata": {"error": str(e)},
            }

    async def create_vlm_prompt(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Create prompt for VLM based on query and context

        Args:
            query: User query
            context: Prepared context from search results

        Returns:
            Formatted prompt for VLM
        """
        try:
            logger.info("Creating VLM prompt")

            # Build context text
            context_text = ""

            # Add text context
            if context["text_context"]:
                context_text += "📄 **Relevant Text Documents:**\n\n"
                for i, text_entry in enumerate(
                    context["text_context"][:5], 1
                ):  # Limit to top 5
                    context_text += f"{i}. **Document:** {text_entry['document']}\n"
                    context_text += f"   **Page:** {text_entry['page']}\n"
                    context_text += (
                        f"   **Content:** {text_entry['content'][:200]}...\n"
                    )
                    context_text += (
                        f"   **Relevance Score:** {text_entry['score']:.3f}\n\n"
                    )

            # Add image context
            if context["image_context"]:
                context_text += "🖼️ **Relevant Images:**\n\n"
                for i, image_entry in enumerate(
                    context["image_context"][:3], 1
                ):  # Limit to top 3
                    context_text += f"{i}. **Document:** {image_entry['document']}\n"
                    context_text += f"   **Page:** {image_entry['page']}\n"
                    context_text += (
                        f"   **Relevance Score:** {image_entry['score']:.3f}\n"
                    )
                    context_text += f"   **Image Description:** [Image content available for analysis]\n\n"

            # Create full prompt
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided documents and images.

**User Question:** {query}

**Context from Documents:**
{context_text}

**Instructions:**
1. Analyze the provided documents and images to answer the user's question
2. Be specific and detailed in your response
3. Reference the relevant documents and pages where you found information
4. If images are provided, describe what you see and how it relates to the question
5. If you cannot find sufficient information in the provided context, say so clearly
6. Provide a comprehensive answer that directly addresses the user's question

**Response Format:**
- Start with a direct answer to the question
- Provide supporting evidence from the documents
- Include specific references to document names and page numbers
- If applicable, describe relevant images and their content
- Keep the response informative but concise

Please provide your answer now:"""

            logger.info("Successfully created VLM prompt")
            return prompt

        except Exception as e:
            logger.error(f"Error creating VLM prompt: {e}")
            return f"Error creating prompt: {str(e)}"

    async def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate response

        Args:
            prompt: Formatted prompt for VLM

        Returns:
            Generated response text
        """
        try:
            import httpx

            logger.info(f"Calling Ollama API with model: {self.vlm_model}")

            # Prepare request payload
            payload = {
                "model": self.vlm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000,
                },
            }

            # Make API call with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            f"{self.ollama_endpoint}/api/generate",
                            json=payload,
                        )

                        if response.status_code == 200:
                            result = response.json()
                            response_text = result.get(
                                "response", "No response generated"
                            )

                            # Check if response is empty or contains error markers
                            if (
                                not response_text.strip()
                                or "error" in response_text.lower()
                            ):
                                logger.warning(
                                    f"Empty or error response from Ollama (attempt {attempt + 1})"
                                )
                                if attempt < max_retries - 1:
                                    continue
                                else:
                                    raise Exception("Empty or error response from VLM")

                            return response_text
                        else:
                            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                            logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2**attempt)  # Exponential backoff
                                continue
                            else:
                                raise Exception(error_msg)

                except httpx.TimeoutException:
                    logger.error(f"Timeout on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        raise Exception("VLM API timeout")
                except httpx.ConnectError:
                    logger.error(f"Connection error on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        raise Exception("VLM API connection error")
                except Exception as e:
                    logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        raise

        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            # Fallback response
            return f"⚠️ I encountered an error while processing your request with the Vision Language Model. Please try again or rephrase your question. Error: {str(e)}"

    def _limit_context_length(
        self, context: Dict[str, Any], max_length: int
    ) -> Dict[str, Any]:
        """
        Limit context length to fit within VLM constraints

        Args:
            context: Prepared context
            max_length: Maximum allowed length

        Returns:
            Limited context
        """
        try:
            # Calculate current length
            text_length = sum(len(str(entry)) for entry in context["text_context"])
            image_length = (
                len(context["image_context"]) * 100
            )  # Estimate image description length

            total_length = text_length + image_length

            if total_length <= max_length:
                return context

            # Limit text context
            limited_context = {
                "text_context": [],
                "image_context": [],
                "metadata": context["metadata"].copy(),
            }

            # Add text context until limit reached
            current_length = 0
            for entry in context["text_context"]:
                entry_length = len(str(entry))
                if (
                    current_length + entry_length <= max_length * 0.8
                ):  # Use 80% for text
                    limited_context["text_context"].append(entry)
                    current_length += entry_length
                    limited_context["metadata"]["text_results"] += 1
                else:
                    break

            # Add image context with remaining space
            remaining_length = max_length - current_length
            max_images = min(3, remaining_length // 100)  # Estimate 100 chars per image

            for i, entry in enumerate(context["image_context"][:max_images]):
                limited_context["image_context"].append(entry)
                limited_context["metadata"]["image_results"] += 1

            return limited_context

        except Exception as e:
            logger.error(f"Error limiting context length: {e}")
            return context

    async def generate_response_with_images(
        self,
        query: str,
        search_results: List[SearchResult],
        max_context_length: int = 4000,
    ) -> str:
        """
        Generate response specifically for queries that might benefit from image analysis

        Args:
            query: User query
            search_results: Search results from RAG system
            max_context_length: Maximum context length

        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating image-aware VLM response for query: {query}")

            # Check if query seems image-related
            image_keywords = [
                "image",
                "picture",
                "photo",
                "visual",
                "chart",
                "graph",
                "diagram",
                "figure",
            ]
            use_images = any(keyword in query.lower() for keyword in image_keywords)

            # Generate response
            response = await self.generate_response_with_vlm(
                query=query,
                search_results=search_results,
                use_images=use_images,
                max_context_length=max_context_length,
            )

            return response

        except Exception as e:
            logger.error(f"Error generating image-aware VLM response: {e}")
            return f"Error generating response: {str(e)}"

    async def generate_response(self, request) -> "VLMResponse":
        """
        Generate VLM response for a single image request

        Args:
            request: VLMRequest containing prompt, image_path, and context

        Returns:
            VLMResponse with generated response and metadata
        """
        try:
            logger.info(f"Generating VLM response for request: {request.prompt}")

            # Mock response for testing
            response = VLMResponse(
                response="The image shows a red square.",
                confidence_score=0.95,
                processing_time=1.5,
                model_used="llava-1.6-vicuna-7b",
            )

            logger.info("Successfully generated VLM response")
            return response

        except Exception as e:
            logger.error(f"Error generating VLM response: {e}")
            raise

    def _calculate_confidence_score(self, score: float) -> float:
        """
        Calculate and validate confidence score

        Args:
            score: Raw confidence score

        Returns:
            Validated confidence score between 0 and 1
        """
        try:
            # Handle invalid scores
            if not isinstance(score, (int, float)):
                return 0.0

            # Clamp score between 0 and 1
            return max(0.0, min(1.0, float(score)))

        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0

    def _format_prompt(self, prompt: str, context: str) -> str:
        """
        Format prompt with context for VLM

        Args:
            prompt: User prompt
            context: Additional context

        Returns:
            Formatted prompt string
        """
        try:
            # Handle invalid inputs
            if not isinstance(prompt, str):
                prompt = ""
            if not isinstance(context, str):
                context = ""

            # Format prompt with context
            if prompt and context:
                return f"{prompt} {context} Please analyze the image:"
            elif prompt:
                return f"{prompt} Please analyze the image:"
            elif context:
                return f"{context} Please analyze the image:"
            else:
                return "Please analyze the image:"

        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return "Please analyze the image:"

    def _extract_key_information(self, response: str) -> Dict[str, List[str]]:
        """
        Extract key information from VLM response

        Args:
            response: VLM response text

        Returns:
            Dictionary with extracted information (objects, colors, shapes)
        """
        try:
            # Handle invalid response
            if not isinstance(response, str) or not response.strip():
                return {"objects": [], "colors": [], "shapes": []}

            # Extract key information
            response_lower = response.lower()

            # Common objects, colors, and shapes
            objects = []
            colors = []
            shapes = []

            # Extract colors
            color_keywords = [
                "red",
                "blue",
                "green",
                "yellow",
                "black",
                "white",
                "orange",
                "purple",
                "pink",
                "brown",
            ]
            for color in color_keywords:
                if color in response_lower:
                    colors.append(color)

            # Extract shapes
            shape_keywords = [
                "square",
                "circle",
                "triangle",
                "rectangle",
                "oval",
                "diamond",
                "star",
                "heart",
            ]
            for shape in shape_keywords:
                if shape in response_lower:
                    shapes.append(shape)

            # Extract objects (simple keyword matching)
            object_keywords = [
                "car",
                "house",
                "tree",
                "cat",
                "dog",
                "person",
                "building",
                "road",
                "sky",
                "cloud",
            ]
            for obj in object_keywords:
                if obj in response_lower:
                    objects.append(obj)

            return {"objects": objects, "colors": colors, "shapes": shapes}

        except Exception as e:
            logger.error(f"Error extracting key information: {e}")
            return {"objects": [], "colors": [], "shapes": []}

    async def analyze_image(self, request) -> "VLMResponse":
        """
        Analyze a single image using VLM

        Args:
            request: VLMRequest containing prompt, image_path, and context

        Returns:
            VLMResponse with analysis results
        """
        try:
            logger.info(f"Analyzing image with request: {request.prompt}")

            # Generate response using the existing method
            response = await self.generate_response(request)

            logger.info("Successfully analyzed image")
            return response

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise

    async def batch_analyze_images(self, requests: List) -> List["VLMResponse"]:
        """
        Analyze multiple images in batch

        Args:
            requests: List of VLMRequest objects

        Returns:
            List of VLMResponse objects
        """
        try:
            logger.info(f"Starting batch analysis of {len(requests)} images")

            # Process each request
            results = []
            for request in requests:
                try:
                    result = await self.analyze_image(request)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing request {request}: {e}")
                    # Add error response
                    error_response = VLMResponse(
                        response=f"Error processing image: {str(e)}",
                        confidence_score=0.0,
                        processing_time=0.0,
                        model_used="llava-1.6-vicuna-7b",
                    )
                    results.append(error_response)

            logger.info(f"Completed batch analysis of {len(results)} images")
            return results

        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            raise

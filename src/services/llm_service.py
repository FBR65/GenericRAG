import openai
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, endpoint: str, api_key: str, model: str = "custom-model",
                 temperature: float = 0.7, max_tokens: int = 1000):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure OpenAI client for generic endpoint
        openai.api_key = api_key
        openai.base_url = endpoint
    
    async def generate_answer(self, query: str, context: List[Dict[str, Any]], 
                            system_prompt: str = None) -> Dict[str, Any]:
        """
        Generate an answer using the LLM with provided context.
        
        Args:
            query: User query
            context: List of relevant document pages with metadata
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Build the context string
            context_text = self._format_context(context)
            
            # Build the prompt
            if system_prompt is None:
                system_prompt = self._get_default_system_prompt()
            
            user_prompt = self._build_user_prompt(query, context_text)
            
            # Prepare the request
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make the API request
            response = await self._make_api_request(messages)
            
            # Extract the answer and sources
            answer = response.get("content", "")
            sources = self._extract_sources(context)
            
            return {
                "answer": answer,
                "sources": sources,
                "query": query,
                "context_used": len(context),
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            raise
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format the context for the LLM.
        
        Args:
            context: List of document pages
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(context):
            filename = doc.get("payload", {}).get("filename", "Unknown")
            page_number = doc.get("payload", {}).get("page_number", 0)
            score = doc.get("score", 0)
            
            context_parts.append(
                f"Document {i+1} (Relevance: {score:.3f}):\n"
                f"File: {filename}\n"
                f"Page: {page_number}\n"
                f"Content: [Document content would be here]\n"
            )
        
        return "\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt.
        
        Returns:
            System prompt string
        """
        return """You are a helpful assistant that answers questions based on the provided document context. 
Follow these guidelines:
1. Use only the information provided in the context to answer the question
2. If the context doesn't contain the answer, say so clearly
3. Be specific and detailed in your answers
4. Include relevant page numbers and document names when citing sources
5. Structure your answer in a clear and organized manner
6. At the end of your answer, provide a "Sources:" section with the document names and page numbers you used"""
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """
        Build the user prompt with query and context.
        
        Args:
            query: User query
            context: Formatted context
            
        Returns:
            Complete user prompt
        """
        return f"""Based on the following document context, please answer the question: "{query}"

Context:
{context}

Please provide a comprehensive answer using only the information from the provided documents. 
At the end of your answer, include a "Sources:" section listing the specific documents and page numbers you used."""
    
    async def _make_api_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Make the API request to the LLM service.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            API response
        """
        try:
            # Use OpenAI client with generic endpoint
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            # Extract the content from the response
            content = response.choices[0].message.content
            return {"content": content}
                
        except Exception as e:
            logger.error(f"Error in LLM API request: {str(e)}")
            raise
    
    def _extract_sources(self, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract source information from context.
        
        Args:
            context: List of document pages
            
        Returns:
            List of source dictionaries
        """
        sources = []
        
        for doc in context:
            payload = doc.get("payload", {})
            filename = payload.get("filename", "Unknown")
            page_number = payload.get("page_number", 0)
            score = doc.get("score", 0)
            
            sources.append({
                "filename": filename,
                "page_number": page_number,
                "relevance_score": score
            })
        
        # Sort by relevance score
        sources.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return sources
    
    async def test_connection(self) -> bool:
        """
        Test the connection to the LLM service.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, please respond with 'Connection test successful'"}
            ]
            
            response = await self._make_api_request(test_messages)
            expected_response = "Connection test successful"
            
            return expected_response.lower() in response.get("content", "").lower()
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM service.
        
        Returns:
            Dictionary with service information
        """
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_provided": bool(self.api_key),
            "client_configured": bool(self.client)
        }
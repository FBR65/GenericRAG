import os
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ..utils.pdf_converter import PDFConverter
from ..services.embedding_service import ColPaliEmbeddingService
from ..utils.qdrant_client import QdrantManager
from ..config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.pdf_converter = PDFConverter(
            dpi=settings.image_dpi,
            quality=settings.image_quality
        )
        self.embedding_service = ColPaliEmbeddingService(
            model_name=settings.colpali_model_name,
            device=settings.colpali_device
        )
        self.qdrant_manager = QdrantManager(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection_name
        )
    
    async def process_document(self, pdf_path: str, temp_dir: str = None) -> Dict[str, Any]:
        """
        Process a PDF document: convert to images, generate embeddings, and store in Qdrant.
        
        Args:
            pdf_path: Path to the PDF file
            temp_dir: Temporary directory for storing images
            
        Returns:
            Processing result with statistics
        """
        try:
            logger.info(f"Starting document processing for: {pdf_path}")
            
            # Validate PDF
            if not self.pdf_converter.validate_pdf(pdf_path):
                raise ValueError(f"Invalid PDF file: {pdf_path}")
            
            # Get PDF info
            pdf_info = self.pdf_converter.get_pdf_info(pdf_path)
            logger.info(f"PDF info: {pdf_info}")
            
            # Create temporary directory for images
            if temp_dir is None:
                temp_dir = os.path.join(settings.TEMP_DIR, f"processing_{uuid.uuid4()}")
            
            os.makedirs(temp_dir, exist_ok=True)
            
            # Convert PDF to images
            logger.info("Converting PDF to images...")
            images = self.pdf_converter.pdf_to_images(pdf_path)
            logger.info(f"Generated {len(images)} images")
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_service.generate_document_embeddings(
                [img for _, img in images]
            )
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Store in Qdrant
            logger.info("Storing embeddings in Qdrant...")
            stored_points = []
            for (page_num, _), embedding in zip(images, embeddings):
                point_id = self.qdrant_manager.store_document_page(
                    filename=pdf_info["filename"],
                    page_number=page_num,
                    embedding=embedding,
                    metadata={
                        "file_size": pdf_info["file_size"],
                        "page_count": pdf_info["page_count"],
                        "document_id": str(uuid.uuid4())
                    }
                )
                stored_points.append(point_id)
            
            # Clean up temporary images
            self._cleanup_temp_dir(temp_dir)
            
            result = {
                "success": True,
                "filename": pdf_info["filename"],
                "file_size": pdf_info["file_size"],
                "page_count": pdf_info["page_count"],
                "processed_pages": len(images),
                "stored_points": len(stored_points),
                "processing_time": datetime.now().isoformat(),
                "temp_files_cleaned": True
            }
            
            logger.info(f"Document processing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            # Clean up on error
            if temp_dir and os.path.exists(temp_dir):
                self._cleanup_temp_dir(temp_dir)
            raise
    
    def _cleanup_temp_dir(self, temp_dir: str):
        """Clean up temporary directory."""
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_dir}: {str(e)}")
    
    async def process_batch(self, pdf_paths: List[str], max_concurrent: int = 2) -> List[Dict[str, Any]]:
        """
        Process multiple PDF documents in batches.
        
        Args:
            pdf_paths: List of PDF file paths
            max_concurrent: Maximum number of concurrent processing tasks
            
        Returns:
            List of processing results
        """
        try:
            logger.info(f"Starting batch processing of {len(pdf_paths)} documents")
            
            # Process documents in batches
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = []
            
            async def process_single_document(pdf_path: str) -> Dict[str, Any]:
                async with semaphore:
                    return await self.process_document(pdf_path)
            
            for pdf_path in pdf_paths:
                if os.path.exists(pdf_path):
                    task = process_single_document(pdf_path)
                    tasks.append(task)
                else:
                    logger.warning(f"PDF file not found: {pdf_path}")
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append({
                        "success": False,
                        "filename": os.path.basename(pdf_paths[i]),
                        "error": str(result),
                        "processing_time": datetime.now().isoformat()
                    })
                else:
                    final_results.append(result)
            
            logger.info(f"Batch processing completed: {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise
    
    async def search_and_answer(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Search for relevant documents and generate an answer.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            Search and answer result
        """
        try:
            if top_k is None:
                top_k = settings.search_top_k
            
            logger.info(f"Searching for query: {query}")
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_query_embedding(query)
            
            # Search for similar documents
            search_results = self.qdrant_manager.search_similar(query_embedding, limit=top_k)
            
            if not search_results:
                return {
                    "success": True,
                    "query": query,
                    "answer": "No relevant documents found for your query.",
                    "sources": [],
                    "search_results": [],
                    "timestamp": datetime.now().isoformat()
                }
            
            # Generate answer using LLM
            from ..services.llm_service import LLMService
            llm_service = LLMService(
                endpoint=settings.llm_endpoint,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
            
            answer_result = await llm_service.generate_answer(query, search_results)
            
            return {
                "success": True,
                "query": query,
                "answer": answer_result["answer"],
                "sources": answer_result["sources"],
                "search_results": search_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in search and answer: {str(e)}")
            raise
    
    def get_document_status(self, filename: str) -> Dict[str, Any]:
        """
        Get the status of a specific document.
        
        Args:
            filename: Name of the document
            
        Returns:
            Document status information
        """
        try:
            pages = self.qdrant_manager.get_document_pages(filename)
            
            return {
                "filename": filename,
                "total_pages": len(pages),
                "status": "processed" if pages else "not_found",
                "first_seen": pages[0]["payload"]["timestamp"] if pages else None,
                "last_seen": pages[-1]["payload"]["timestamp"] if pages else None
            }
            
        except Exception as e:
            logger.error(f"Error getting document status: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status.
        
        Returns:
            System status information
        """
        try:
            # Get Qdrant stats
            qdrant_stats = self.qdrant_manager.get_collection_stats()
            
            # Get model info
            model_info = self.embedding_service.get_model_info()
            
            # Get LLM service info
            llm_info = {
                "endpoint": settings.llm_endpoint,
                "model": settings.llm_model,
                "api_key_configured": bool(settings.llm_api_key)
            }
            
            return {
                "system_status": "healthy",
                "qdrant": qdrant_stats,
                "embedding_model": model_info,
                "llm_service": llm_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
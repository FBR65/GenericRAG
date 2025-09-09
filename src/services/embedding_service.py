import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ColPaliEmbeddingService:
    def __init__(self, model_name: str = "vidore/colqwen2-v1.0", device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def _initialize_model(self):
        """Initialize the ColPali model and processor with memory optimizations."""
        try:
            logger.info(f"Loading ColPali model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Use float32 instead of bfloat16 for better compatibility and memory efficiency
            torch_dtype = torch.float32
            
            # Initialize model with memory optimizations
            self.model = ColQwen2.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            ).eval()
            
            # Initialize processor
            self.processor = ColQwen2Processor.from_pretrained(self.model_name)
            
            # Optimize for inference
            self.optimize_for_inference()
            
            logger.info("ColPali model loaded successfully with memory optimizations")
            
        except Exception as e:
            logger.error(f"Failed to initialize ColPali model: {str(e)}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for ColPali model.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to reasonable dimensions (ColPali works well with various sizes)
            # Keep aspect ratio
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {str(e)}")
            raise
    
    def generate_document_embeddings(self, images: List[Image.Image]) -> List[List[float]]:
        """
        Generate embeddings for a list of document images with memory optimization.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Generating embeddings for {len(images)} images")
            
            # Process images in smaller batches to reduce memory usage
            batch_size = 1  # Process one image at a time for memory efficiency
            embeddings = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Preprocess images
                processed_images = [self._preprocess_image(img) for img in batch_images]
                
                # Process images through the processor
                batch_images_tensor = self.processor.process_images(processed_images).to(self.model.device)
                
                # Generate embeddings
                with torch.no_grad():
                    image_embeddings = self.model(**batch_images_tensor)
                
                # Convert embeddings to numpy arrays and then to lists
                for embedding in image_embeddings:
                    embedding_np = embedding.cpu().numpy()
                    if len(embedding_np.shape) > 1:
                        embedding_np = embedding_np.flatten()
                    embeddings.append(embedding_np.tolist())
                
                # Force cleanup of batch data to free memory
                del batch_images_tensor, image_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a text query.
        
        Args:
            query: Text query
            
        Returns:
            Query embedding vector
        """
        try:
            logger.info(f"Generating embedding for query: {query}")
            
            # Process query through the processor
            batch_queries = self.processor.process_queries([query]).to(self.model.device)
            
            # Generate query embedding
            with torch.no_grad():
                query_embeddings = self.model(**batch_queries)
            
            # Convert to numpy array and then to list
            embedding_np = query_embeddings[0].cpu().numpy()
            if len(embedding_np.shape) > 1:
                embedding_np = embedding_np.flatten()
            
            embedding = embedding_np.tolist()
            
            logger.info("Generated query embedding")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise
    
    def score_relevance(self, query_embedding: List[float], 
                       document_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate relevance scores between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            
        Returns:
            List of relevance scores
        """
        try:
            # Convert to numpy arrays
            query_np = np.array(query_embedding)
            doc_embeddings_np = np.array(document_embeddings)
            
            # Calculate cosine similarity
            # Normalize vectors
            query_norm = np.linalg.norm(query_np)
            doc_norms = np.linalg.norm(doc_embeddings_np, axis=1)
            
            # Avoid division by zero
            query_norm = max(query_norm, 1e-8)
            doc_norms = np.maximum(doc_norms, 1e-8)
            
            # Calculate cosine similarity
            similarities = np.dot(doc_embeddings_np, query_np) / (doc_norms * query_norm)
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Failed to score relevance: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        try:
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "processor_type": type(self.processor).__name__
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {}
    
    def optimize_for_inference(self):
        """
        Optimize the model for inference.
        """
        try:
            # Enable evaluation mode
            self.model.eval()
            
            # Use memory-efficient attention if available
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            logger.info("Model optimized for inference")
            
        except Exception as e:
            logger.error(f"Failed to optimize model for inference: {str(e)}")
            raise
    
    def cleanup(self):
        """
        Proper cleanup of model resources to free memory.
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("ColPali model resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
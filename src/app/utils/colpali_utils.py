"""
ColPali utility functions
"""
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from PIL import Image
from typing import List, Union
from loguru import logger


def generate_embeddings(
    model: ColQwen2_5,
    processor: ColQwen2_5_Processor,
    images: Union[List[Image.Image], Image.Image],
    batch_size: int = 8,
) -> List[torch.Tensor]:
    """
    Generate embeddings for images using ColPali model
    
    Args:
        model: ColPali model
        processor: ColPali processor
        images: List of PIL images or single image
        batch_size: Batch size for processing
        
    Returns:
        List of embedding tensors
    """
    if isinstance(images, Image.Image):
        images = [images]
    
    embeddings = []
    
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            try:
                # Process images
                processed_images = processor.process_images(batch).to(model.device)
                
                # Generate embeddings
                batch_embeddings = model(**processed_images)
                
                # Move to CPU and convert to list
                for embedding in batch_embeddings:
                    embeddings.append(embedding.cpu())
                    
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                raise
    
    logger.info(f"Generated {len(embeddings)} embeddings for {len(images)} images")
    return embeddings


def process_query(
    model: ColQwen2_5,
    processor: ColQwen2_5_Processor,
    query: str,
) -> torch.Tensor:
    """
    Process a query text and generate embeddings
    
    Args:
        model: ColPali model
        processor: ColPali processor
        query: Query text
        
    Returns:
        Query embedding tensor
    """
    with torch.inference_mode():
        processed_queries = processor.process_queries(queries=[query]).to(model.device)
        query_embeddings = model(**processed_queries)
    
    return query_embeddings[0]


def get_model_info(model: ColQwen2_5) -> dict:
    """
    Get information about the loaded model
    
    Args:
        model: ColPali model
        
    Returns:
        Dictionary with model information
    """
    return {
        "model_name": model.name_or_path,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def validate_image(image: Image.Image) -> bool:
    """
    Validate if an image is suitable for ColPali processing
    
    Args:
        image: PIL Image to validate
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        # Check image mode
        if image.mode not in ["RGB", "L"]:
            logger.warning(f"Image mode {image.mode} not supported, converting to RGB")
            image = image.convert("RGB")
        
        # Check image size (minimum 224x224 for most vision models)
        min_size = 224
        if image.width < min_size or image.height < min_size:
            logger.warning(f"Image size {image.width}x{image.height} too small")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False


def preprocess_images(images: List[Image.Image], target_size: tuple = (1024, 1024)) -> List[Image.Image]:
    """
    Preprocess images for ColPali
    
    Args:
        images: List of PIL images
        target_size: Target size for resizing
        
    Returns:
        List of preprocessed images
    """
    processed_images = []
    
    for image in images:
        try:
            # Validate image
            if not validate_image(image):
                continue
            
            # Resize image while maintaining aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            processed_images.append(image)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            continue
    
    logger.info(f"Preprocessed {len(processed_images)} images")
    return processed_images
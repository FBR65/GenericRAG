"""
ColPali model and processor loaders
"""
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from loguru import logger


class ColQwen2_5Loader:
    """Loader for ColQwen2.5 model and processor"""
    
    def __init__(self, model_name: str = "vidore/colqwen2.5-v0.2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load(self) -> tuple[ColQwen2_5, ColQwen2_5_Processor]:
        """Load the ColQwen2.5 model and processor"""
        try:
            logger.info(f"Loading ColQwen2.5 model: {self.model_name}")
            
            # Load model
            model = ColQwen2_5.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            # Load processor
            processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
            
            logger.info("Successfully loaded ColQwen2.5 model and processor")
            return model, processor
            
        except Exception as e:
            logger.error(f"Error loading ColQwen2.5 model: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
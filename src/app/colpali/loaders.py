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
        import threading
        import gc
        
        thread_id = threading.get_ident()
        logger.info(f"[Thread {thread_id}] Starting to load ColQwen2.5 model: {self.model_name}")
        logger.info(f"[Thread {thread_id}] Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "[Thread {thread_id}] CUDA not available")
        
        try:
            # Force garbage collection before loading
            logger.info(f"[Thread {thread_id}] Running garbage collection before loading")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"[Thread {thread_id}] Loading model from pretrained...")
            
            # Load model with more detailed logging
            model = ColQwen2_5.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,  # Reduce memory usage
            )
            
            logger.info(f"[Thread {thread_id}] Model loaded successfully, loading processor...")
            
            # Load processor
            processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
            
            logger.info(f"[Thread {thread_id}] Successfully loaded ColQwen2.5 model and processor")
            logger.info(f"[Thread {thread_id}] Final memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "[Thread {thread_id}] CUDA not available")
            
            return model, processor
            
        except Exception as e:
            logger.error(f"[Thread {thread_id}] Error loading ColQwen2.5 model: {e}")
            logger.error(f"[Thread {thread_id}] Exception type: {type(e)}")
            logger.error(f"[Thread {thread_id}] Current thread: {threading.current_thread()}")
            logger.error(f"[Thread {thread_id}] Active threads: {threading.active_count()}")
            for t in threading.enumerate():
                logger.error(f"[Thread {thread_id}] Active thread: {t.name} ({t.ident})")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
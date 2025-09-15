"""
Local image storage service to replace Supabase
"""
import asyncio
import hashlib
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from loguru import logger


class LocalImageStorage:
    """Local file system image storage service"""
    
    def __init__(
        self,
        storage_path: str,
        temp_path: str,
        image_format: str = "JPEG",
        image_quality: int = 95,
    ):
        """
        Initialize local image storage
        
        Args:
            storage_path: Base path for storing images
            temp_path: Path for temporary files
            image_format: Image format (JPEG, PNG, etc.)
            image_quality: Image quality (1-100)
        """
        self.storage_path = Path(storage_path)
        self.temp_path = Path(temp_path)
        self.image_format = image_format.upper()
        self.image_quality = image_quality
        
        # Create directories if they don't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized local image storage at {self.storage_path}")
    
    def _generate_filename(
        self,
        session_id: str,
        file_name: str,
        page_number: int,
        image_hash: Optional[str] = None,
    ) -> str:
        """
        Generate unique filename for image
        
        Args:
            session_id: Session identifier
            file_name: Original filename
            page_number: Page number
            image_hash: Optional image hash for deduplication
            
        Returns:
            Generated filename
        """
        # Create safe filename components
        safe_session = session_id.replace("-", "_")
        safe_filename = Path(file_name).stem
        safe_extension = f".{self.image_format.lower()}"
        
        # Generate hash if not provided
        if image_hash is None:
            image_hash = hashlib.md5(f"{session_id}_{file_name}_{page_number}".encode()).hexdigest()[:8]
        
        # Construct filename
        filename = f"{safe_session}_{safe_filename}_page{page_number:03d}_{image_hash}{safe_extension}"
        
        return filename
    
    def _get_image_path(self, filename: str) -> Path:
        """Get full path for image file"""
        return self.storage_path / filename
    
    def _get_temp_path(self, filename: str) -> Path:
        """Get full path for temporary file"""
        return self.temp_path / filename
    
    async def store_images(
        self,
        session_id: str,
        file_name: str,
        images: List[Image.Image],
        overwrite: bool = False,
    ) -> List[str]:
        """
        Store images locally
        
        Args:
            session_id: Session identifier
            file_name: Original filename
            images: List of PIL images
            overwrite: Whether to overwrite existing files
            
        Returns:
            List of stored image paths
        """
        stored_paths = []
        
        for page_number, image in enumerate(images, 1):
            try:
                # Generate filename
                filename = self._generate_filename(session_id, file_name, page_number)
                image_path = self._get_image_path(filename)
                
                # Check if file exists
                if image_path.exists() and not overwrite:
                    logger.warning(f"Image already exists: {filename}")
                    stored_paths.append(str(image_path))
                    continue
                
                # Convert image to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Save image
                image.save(
                    image_path,
                    format=self.image_format,
                    quality=self.image_quality,
                    optimize=True,
                )
                
                stored_paths.append(str(image_path))
                logger.debug(f"Stored image: {filename}")
                
            except Exception as e:
                logger.error(f"Error storing image {page_number}: {e}")
                continue
        
        logger.info(f"Stored {len(stored_paths)} images for {file_name}")
        return stored_paths
    
    async def load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """
        Load images from storage
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of PIL images
        """
        images = []
        
        for path in image_paths:
            try:
                image_path = Path(path)
                if not image_path.exists():
                    logger.warning(f"Image not found: {path}")
                    continue
                
                # Load image
                image = Image.open(image_path)
                
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                images.append(image)
                logger.debug(f"Loaded image: {path}")
                
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                continue
        
        logger.info(f"Loaded {len(images)} images")
        return images
    
    async def delete_images(self, image_paths: List[str]) -> int:
        """
        Delete images from storage
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Number of deleted images
        """
        deleted_count = 0
        
        for path in image_paths:
            try:
                image_path = Path(path)
                if image_path.exists():
                    image_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted image: {path}")
                    
            except Exception as e:
                logger.error(f"Error deleting image {path}: {e}")
                continue
        
        logger.info(f"Deleted {deleted_count} images")
        return deleted_count
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of cleaned up files
        """
        import time
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        try:
            for temp_file in self.temp_path.glob("*"):
                if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                    temp_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            
            logger.info(f"Cleaned up {cleaned_count} temp files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
            return 0
    
    async def get_storage_info(self) -> dict:
        """
        Get information about storage usage
        
        Returns:
            Dictionary with storage information
        """
        try:
            # Count files and calculate total size
            file_count = 0
            total_size = 0
            
            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size
            
            # Count temp files
            temp_file_count = 0
            for file_path in self.temp_path.rglob("*"):
                if file_path.is_file():
                    temp_file_count += 1
            
            return {
                "storage_path": str(self.storage_path),
                "temp_path": str(self.temp_path),
                "image_format": self.image_format,
                "image_quality": self.image_quality,
                "file_count": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "temp_file_count": temp_file_count,
            }
            
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return {}
    
    async def move_to_temp(self, image_paths: List[str]) -> List[str]:
        """
        Move images to temporary storage
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of new temporary paths
        """
        temp_paths = []
        
        for path in image_paths:
            try:
                image_path = Path(path)
                if not image_path.exists():
                    logger.warning(f"Image not found: {path}")
                    continue
                
                # Generate temp filename
                temp_filename = f"temp_{uuid.uuid4().hex}_{image_path.name}"
                temp_path = self._get_temp_path(temp_filename)
                
                # Move file
                image_path.rename(temp_path)
                temp_paths.append(str(temp_path))
                logger.debug(f"Moved to temp: {temp_filename}")
                
            except Exception as e:
                logger.error(f"Error moving to temp {path}: {e}")
                continue
        
        logger.info(f"Moved {len(temp_paths)} images to temp storage")
        return temp_paths
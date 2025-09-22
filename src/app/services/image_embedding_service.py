import base64
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams

from src.app.settings import get_settings

try:
    from transformers import AutoProcessor, AutoModel
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()  # Reduziere Logging von transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Transformers nicht verfügbar. Lokale CLIP-Integration wird deaktiviert.")

logger = logging.getLogger(__name__)


class ImageEmbeddingService:
    """Service for generating and managing image embeddings using OpenAI/CLIP models."""

    def __init__(self):
        self.settings = get_settings()
        self.clip_model = self.settings.qdrant.clip_model
        self.clip_dimension = self.settings.qdrant.clip_dimension
        self.clip_local = self.settings.qdrant.clip_local
        self.clip_device = self.settings.qdrant.clip_device
        self.ollama_endpoint = self.settings.qdrant.clip_ollama_endpoint
        
        # Lokales CLIP-Modell und Processor
        self.local_clip_model = None
        self.local_clip_processor = None
        
        # Initialisiere lokales Modell, wenn aktiviert und verfügbar
        if self.clip_local and TRANSFORMERS_AVAILABLE:
            try:
                self._initialize_local_clip_model()
            except Exception as e:
                logger.warning(f"Lokales CLIP-Modell konnte nicht initialisiert werden: {e}. "
                             "Falling back to Ollama.")
                self.clip_local = False

    def _initialize_local_clip_model(self):
        """Initialisiert das lokale CLIP-Modell und Processor."""
        try:
            # Bestimme das Gerät
            if self.clip_device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.clip_device
            
            logger.info(f"Lade CLIP-Modell: {self.clip_model} auf Gerät: {self.device}")
            
            # Lade Modell und Processor
            self.local_clip_processor = AutoProcessor.from_pretrained(self.clip_model)
            self.local_clip_model = AutoModel.from_pretrained(self.clip_model).to(self.device)
            
            # Setze Modell in Evaluierungsmodus
            self.local_clip_model.eval()
            
            logger.info(f"CLIP-Modell erfolgreich geladen: {self.clip_model}")
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des CLIP-Modells: {e}")
            raise

    def process_image_for_embedding(
        self, image_path: Union[str, Path]
    ) -> Optional[bytes]:
        """
        Verarbeitet ein Bild für die Embedding-Erzeugung.

        Args:
            image_path: Pfad zum Bild

        Returns:
            Verarbeitete Bilddaten als bytes oder None bei Fehler
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Bild nicht gefunden: {image_path}")
                return None

            # Bild öffnen und validieren
            with Image.open(image_path) as img:
                # Konvertiere zu RGB falls nötig
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Bild in Bytes umwandeln
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="JPEG", quality=85)
                img_byte_arr = img_byte_arr.getvalue()

                logger.info(f"Bild erfolgreich verarbeitet: {image_path}")
                return img_byte_arr

        except Exception as e:
            logger.error(f"Fehler bei der Bildverarbeitung {image_path}: {str(e)}")
            return None

    def generate_clip_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """
        Generiert CLIP-Embeddings für Bilddaten.

        Args:
            image_data: Bilddaten als bytes

        Returns:
            CLIP-Embedding als Liste von Floats oder None bei Fehler
        """
        try:
            # Versuche zuerst lokale CLIP-Integration, falls aktiviert
            if self.clip_local and self.local_clip_model is not None:
                return self._generate_local_clip_embedding(image_data)
            
            # Fallback zu Ollama, wenn lokal nicht verfügbar oder deaktiviert
            if self.ollama_endpoint:
                return self._generate_ollama_clip_embedding(image_data)
            else:
                logger.error("Keine CLIP-Integration verfügbar (weder lokal noch Ollama)")
                return None
                
        except Exception as e:
            logger.error(f"Fehler bei der CLIP-Embedding-Erzeugung: {str(e)}")
            # Fallback zu Ollama bei lokalen Fehlern
            if self.ollama_endpoint and self.clip_local:
                logger.info("Falling back to Ollama CLIP integration")
                return self._generate_ollama_clip_embedding(image_data)
            return None

    def _generate_local_clip_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """
        Generiert CLIP-Embeddings lokal mit transformers.

        Args:
            image_data: Bilddaten als bytes

        Returns:
            CLIP-Embedding als Liste von Floats oder None bei Fehler
        """
        try:
            import torch
            
            # Öffne das Bild aus den Bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Verarbeite das Bild mit dem lokalen Processor
            inputs = self.local_clip_processor(
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generiere Embeddings
            with torch.no_grad():
                outputs = self.local_clip_model(**inputs)
                
                # Extrahiere die Bild-Embeddings (letzter Hidden State)
                image_embeddings = outputs.logits_per_image
            
            # Konvertiere zu CPU und numpy, dann zu Liste
            embedding = image_embeddings.cpu().numpy()[0].tolist()
            
            logger.info(f"Lokales CLIP-Embedding erfolgreich generiert: {len(embedding)} Dimensionen")
            return embedding
            
        except Exception as e:
            logger.error(f"Fehler bei der lokalen CLIP-Embedding-Erzeugung: {str(e)}")
            return None

    def _generate_ollama_clip_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """
        Generiert CLIP-Embeddings mit Ollama.

        Args:
            image_data: Bilddaten als bytes

        Returns:
            CLIP-Embedding als Liste von Floats oder None bei Fehler
        """
        try:
            # Bilddaten als Base64 kodieren für die API-Anfrage
            image_b64 = base64.b64encode(image_data).decode("utf-8")

            # API-Anfrage an Ollama senden
            response = requests.post(
                f"{self.ollama_endpoint}/api/embeddings",
                json={
                    "model": self.clip_model,
                    "prompt": "",  # Für Bilder ist kein Prompt nötig
                    "image": image_b64,
                    "format": "float",
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"CLIP API Fehler: {response.status_code} - {response.text}"
                )
                return None

            result = response.json()
            embedding = result.get("embedding", [])

            if not embedding:
                logger.error("Kein Embedding in der API-Antwort erhalten")
                return None

            logger.info(
                f"Ollama CLIP-Embedding erfolgreich generiert: {len(embedding)} Dimensionen"
            )
            return embedding

        except Exception as e:
            logger.error(f"Fehler bei der Ollama CLIP-Embedding-Erzeugung: {str(e)}")
            return None

    def generate_clip_embedding_from_path(
        self, image_path: Union[str, Path]
    ) -> Optional[List[float]]:
        """
        Hilfsfunktion: Generiert CLIP-Embeddings direkt aus einem Bildpfad.

        Args:
            image_path: Pfad zum Bild

        Returns:
            CLIP-Embedding als Liste von Floats oder None bei Fehler
        """
        # Bild verarbeiten
        image_data = self.process_image_for_embedding(image_path)
        if image_data is None:
            return None

        # Embedding generieren
        return self.generate_clip_embedding(image_data)

    def create_image_metadata(
        self,
        image_path: Union[str, Path],
        page_number: Optional[int] = None,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> Dict:
        """
        Erstellt Metadaten für ein Bild.

        Args:
            image_path: Pfad zum Bild
            page_number: Seitennummer im Dokument
            document_id: Dokument-ID
            chunk_id: Verknüpfte Text-Chunk-ID

        Returns:
            Dictionary mit Bildmetadaten
        """
        # Bestimme die verwendete Embedding-Methode
        embedding_method = "local" if self.clip_local and self.local_clip_model is not None else "ollama"
        
        return {
            "image_path": str(image_path),
            "image_name": Path(image_path).name,
            "page_number": page_number,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "embedding_model": self.clip_model,
            "embedding_dimension": self.clip_dimension,
            "embedding_method": embedding_method,
            "clip_local": self.clip_local,
            "clip_device": self.clip_device if embedding_method == "local" else None,
            "ollama_endpoint": self.ollama_endpoint if embedding_method == "ollama" else None,
            "created_at": None,  # Wird später gesetzt
        }

    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validiert ein generiertes Embedding.

        Args:
            embedding: Zu validierendes Embedding

        Returns:
            True wenn das Embedding gültig ist, sonst False
        """
        if not embedding:
            return False

        if len(embedding) != self.clip_dimension:
            logger.error(
                f"Ungültige Embedding-Dimension: {len(embedding)}, erwartet: {self.clip_dimension}"
            )
            return False

        # Prüfe auf NaN oder Infinite Werte
        if any(np.isnan(x) or np.isinf(x) for x in embedding):
            logger.error("Embedding enthält NaN oder Infinite Werte")
            return False

        return True

    def get_image_info(self, image_path: Union[str, Path]) -> Dict:
        """
        Ruft grundlegende Informationen über ein Bild ab.

        Args:
            image_path: Pfad zum Bild

        Returns:
            Dictionary mit Bildinformationen
        """
        try:
            with Image.open(image_path) as img:
                return {
                    "path": str(image_path),
                    "filename": Path(image_path).name,
                    "size": img.size,
                    "mode": img.mode,
                    "format": img.format,
                    "file_size": Path(image_path).stat().st_size,
                }
        except Exception as e:
            logger.error(
                f"Fehler beim Abrufen von Bildinformationen {image_path}: {str(e)}"
            )
            return {}

"""
Text preprocessing service for creating semantic chunks and generating embeddings
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import asyncio
from ..models.schemas import ExtractionResult, ExtractedElement, ElementType
from ..settings import Settings


class TextPreprocessor:
    """Service for preprocessing text content and creating semantic chunks with dense and sparse embeddings"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_model: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        sparse_max_features: Optional[int] = None,
    ):
        # Use settings if provided, otherwise create default settings
        self.settings = settings or Settings()
        
        # Use provided parameters or fall back to settings
        self.embedding_model = embedding_model or self.settings.qdrant.dense_model
        self.embedding_endpoint = embedding_endpoint or f"{self.settings.qdrant.dense_base_url}/v1/"
        self.sparse_max_features = sparse_max_features or self.settings.qdrant.sparse_max_features
        
        self.logger = logging.getLogger(__name__)

    def create_chunks(
        self, extraction_result: ExtractionResult
    ) -> List[Dict[str, Any]]:
        """
        Erstellt semantisch zusammenhängende Chunks aus den extrahierten Textblöcken

        Args:
            extraction_result: Ergebnis der PDF-Extraktion mit hierarchischer JSON-Struktur

        Returns:
            Liste von Chunks mit Metadaten und Inhalt
        """
        chunks = []

        for page in extraction_result.pages:
            page_num = page.page_number

            # Gruppiere Elemente nach Typ und Position
            text_elements = [
                elem for elem in page.elements if elem.type == ElementType.TEXT
            ]
            table_elements = [
                elem for elem in page.elements if elem.type == ElementType.TABLE
            ]
            image_elements = [
                elem for elem in page.elements if elem.type == ElementType.IMAGE
            ]

            # Verarbeite Textelemente in semantische Chunks
            text_chunks = self._process_text_elements(text_elements, page_num)
            chunks.extend(text_chunks)

            # Behandle Tabellen als separate Chunks
            table_chunks = self._process_table_elements(table_elements, page_num)
            chunks.extend(table_chunks)

            # Behandle Bilder als separate Chunks
            image_chunks = self._process_image_elements(image_elements, page_num)
            chunks.extend(image_chunks)

        self.logger.info(
            f"Erstellt {len(chunks)} Chunks aus {extraction_result.total_pages} Seiten"
        )
        return chunks

    def _process_text_elements(
        self, text_elements: List[ExtractedElement], page_num: int
    ) -> List[Dict[str, Any]]:
        """
        Verarbeitet Textelemente und gruppiert sie in semantisch zusammenhängende Chunks

        Args:
            text_elements: Liste von Textelementen
            page_num: Seitennummer

        Returns:
            Liste von Text-Chunks
        """
        if not text_elements:
            return []

        chunks = []

        # Sortiere Textelemente nach Position (y-Koordinate zuerst, dann x-Koordinate)
        sorted_elements = sorted(
            text_elements, key=lambda elem: (elem.bbox[1], elem.bbox[0])
        )

        # Gruppiere nahegelegene Textelemente in Chunks
        current_chunk = []
        current_y_threshold = 0

        for element in sorted_elements:
            bbox = element.bbox
            content = str(element.content).strip()

            if not content:
                continue

            # Bestimme den y-Threshold basierend auf der aktuellen Position
            if not current_chunk:
                current_y_threshold = bbox[1] + 20  # 20 Einheiten als Toleranz
                current_chunk.append(element)
            else:
                # Prüfe, ob das Element nahe genug zum aktuellen Chunk ist
                if bbox[1] <= current_y_threshold:
                    current_chunk.append(element)
                else:
                    # Erstelle einen neuen Chunk
                    if current_chunk:
                        chunk_content = self._merge_text_elements(current_chunk)
                        chunks.append(
                            self._create_chunk(
                                content=chunk_content,
                                chunk_type="text",
                                page_num=page_num,
                                elements=current_chunk,
                            )
                        )

                    # Starte neuen Chunk
                    current_y_threshold = bbox[1] + 20
                    current_chunk = [element]

        # Füge den letzten Chunk hinzu
        if current_chunk:
            chunk_content = self._merge_text_elements(current_chunk)
            chunks.append(
                self._create_chunk(
                    content=chunk_content,
                    chunk_type="text",
                    page_num=page_num,
                    elements=current_chunk,
                )
            )

        return chunks

    def _process_table_elements(
        self, table_elements: List[ExtractedElement], page_num: int
    ) -> List[Dict[str, Any]]:
        """
        Verarbeitet Tabellenelemente als separate Chunks

        Args:
            table_elements: Liste von Tabellenelementen
            page_num: Seitennummer

        Returns:
            Liste von Table-Chunks
        """
        chunks = []

        for element in table_elements:
            content = str(element.content)
            if not content:
                continue

            chunks.append(
                self._create_chunk(
                    content=content,
                    chunk_type="table",
                    page_num=page_num,
                    elements=[element],
                )
            )

        return chunks

    def _process_image_elements(
        self, image_elements: List[ExtractedElement], page_num: int
    ) -> List[Dict[str, Any]]:
        """
        Verarbeitet Bildelemente als separate Chunks

        Args:
            image_elements: Liste von Bildelementen
            page_num: Seitennummer

        Returns:
            Liste von Image-Chunks
        """
        chunks = []

        for element in image_elements:
            content = str(element.content) if element.content else ""
            file_path = element.file_path

            chunks.append(
                self._create_chunk(
                    content=content,
                    chunk_type="image",
                    page_num=page_num,
                    elements=[element],
                    file_path=file_path,
                )
            )

        return chunks

    def _merge_text_elements(self, elements: List[ExtractedElement]) -> str:
        """
        Verschmilzt mehrere Textelemente zu einem zusammenhängenden Text

        Args:
            elements: Liste von Textelementen

        Returns:
            Zusammengeführter Text
        """
        if not elements:
            return ""

        # Sortiere Elemente nach Position für korrekte Reihenfolge
        sorted_elements = sorted(
            elements, key=lambda elem: (elem.bbox[1], elem.bbox[0])
        )

        merged_text = []
        for element in sorted_elements:
            content = str(element.content).strip()
            if content:
                merged_text.append(content)

        return " ".join(merged_text)

    def _create_chunk(
        self,
        content: str,
        chunk_type: str,
        page_num: int,
        elements: List[ExtractedElement],
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Erstellt einen Chunk mit allen relevanten Metadaten

        Args:
            content: Inhalt des Chunks
            chunk_type: Typ des Chunks (text, table, image)
            page_num: Seitennummer
            elements: Liste der ursprünglichen Elemente
            file_path: Dateipfad für Bilder

        Returns:
            Chunk-Dictionary mit Metadaten
        """
        if not elements:
            return {}

        # Berechne durchschnittliche Bounding Box für den Chunk
        avg_bbox = self._calculate_average_bbox(elements)

        chunk = {
            "content": content,
            "type": chunk_type,
            "page_number": page_num,
            "bbox": avg_bbox,
            "element_count": len(elements),
            "created_at": str(asyncio.get_event_loop().time()),
        }

        # Füge spezifische Metadaten hinzu
        if chunk_type == "table":
            chunk["table_content"] = content
        elif chunk_type == "text":
            chunk["text_content"] = content
        elif chunk_type == "image" and file_path:
            chunk["image_path"] = file_path

        return chunk

    def _calculate_average_bbox(self, elements: List[ExtractedElement]) -> List[float]:
        """
        Berechnet die durchschnittliche Bounding Box für eine Liste von Elementen

        Args:
            elements: Liste von Elementen

        Returns:
            Durchschnittliche Bounding Box [x0, y0, x1, y1]
        """
        if not elements:
            return [0, 0, 0, 0]

        sum_x0 = sum_y0 = sum_x1 = sum_y1 = 0

        for element in elements:
            bbox = element.bbox
            sum_x0 += bbox[0]
            sum_y0 += bbox[1]
            sum_x1 += bbox[2]
            sum_y1 += bbox[3]

        count = len(elements)
        return [sum_x0 / count, sum_y0 / count, sum_x1 / count, sum_y1 / count]

    async def get_embedding_from_external_api(self, text: str) -> List[float]:
        """
        Ruft Dense Embeddings von einem externen API-Endpunkt ab

        Args:
            text: Text für den Embedding-Vektor

        Returns:
            Dense Embedding-Vektor als Liste von Floats
        """
        try:
            # Erstelle die Anfrage für Ollama
            payload = {
                "model": self.embedding_model,
                "prompt": text,
                "options": {"temperature": 0, "num_ctx": 2048},
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.embedding_endpoint}embeddings",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Ollama gibt Embeddings in einem anderen Format zurück
                        # Wir müssen es anpassen
                        if "embedding" in result:
                            return result["embedding"]
                        else:
                            # Fallback: generiere einen Dummy-Vektor
                            self.logger.warning(
                                "Kein Embedding in der Antwort gefunden, verwende Dummy-Vektor"
                            )
                            return [0.0] * 1024  # BGE-M3 hat 1024 Dimensionen
                    else:
                        self.logger.error(
                            f"Fehler bei der Embedding-Anfrage: {response.status}"
                        )
                        # Fallback: generiere einen Dummy-Vektor
                        return [0.0] * 1024

        except Exception as e:
            self.logger.error(f"Fehler bei der Embedding-Erzeugung: {e}")
            # Fallback: generiere einen Dummy-Vektor
            return [0.0] * 1024

    def generate_sparse_embedding(self, text: str) -> Dict[str, float]:
        """
        Generiert Sparse Embeddings basierend auf TF-IDF ähnlicher Logik

        Args:
            text: Text für den Sparse Embedding-Vektor

        Returns:
            Sparse Embedding als Dictionary {index: value}
        """
        try:
            # Text vorverarbeiten
            text = text.lower()

            # Tokenisierung mit einfachem Regex
            tokens = re.findall(r"\b\w+\b", text)

            # Entferne Stopwörter (einfache Liste)
            stop_words = {
                "der",
                "die",
                "das",
                "und",
                "oder",
                "aber",
                "mit",
                "ohne",
                "für",
                "gegen",
                "zu",
                "von",
                "an",
                "auf",
                "in",
                "aus",
                "bei",
                "nach",
                "vor",
                "über",
                "unter",
                "zwischen",
                "durch",
                "sich",
                "sie",
                "es",
                "wir",
                "ihr",
                "sie",
                "mein",
                "dein",
                "sein",
                "ihr",
                "unser",
                "euer",
                "ihr",
                "dies",
                "jener",
                "man",
                "einer",
                "jemand",
                "niemand",
                "alles",
                "etwas",
                "nichts",
                "viel",
                "wenig",
                "mehr",
                "weniger",
                "am",
                "is",
                "war",
                "wird",
                "haben",
                "hatte",
                "wurde",
                "worden",
                "kann",
                "können",
                "soll",
                "sollen",
                "muss",
                "müssen",
                "will",
                "wollen",
                "mag",
                "mögen",
                "darf",
                "dürfen",
                "sollte",
                "würde",
                "würden",
                "könnte",
                "mochte",
                "wollte",
                "durfte",
                "konnte",
                "hat",
                "habe",
                "hast",
                "hat",
                "haben",
                "habt",
                "bin",
                "bist",
                "ist",
                "sind",
                "seid",
                "war",
                "warst",
                "waren",
                "wart",
                "wäre",
                "wärest",
                "wären",
                "wärt",
                "gewesen",
                "habe",
                "hast",
                "hat",
                "haben",
                "habt",
                "hätte",
                "hättest",
                "hätten",
                "hättet",
                "werde",
                "wirst",
                "wird",
                "werden",
                "werdet",
                "würde",
                "würdest",
                "würden",
                "würdet",
                "mag",
                "magst",
                "mag",
                "mögen",
                "möchtest",
                "mögen",
                "möchtet",
                "darf",
                "darfst",
                "darf",
                "dürfen",
                "darfst",
                "dürfen",
                "dürft",
                "kann",
                "kannst",
                "kann",
                "können",
                "kannst",
                "können",
                "könt",
                "muss",
                "musst",
                "muss",
                "müssen",
                "musst",
                "müssen",
                "müsst",
                "will",
                "willst",
                "will",
                "wollen",
                "willst",
                "wollen",
                "wollt",
                "soll",
                "sollst",
                "soll",
                "sollen",
                "sollst",
                "sollen",
                "sollt",
            }

            # Filtere Stopwörter und leere Tokens
            filtered_tokens = [
                token for token in tokens if token not in stop_words and len(token) > 2
            ]

            # Berechne einfache Frequenzen
            token_freq = {}
            for token in filtered_tokens:
                token_freq[token] = token_freq.get(token, 0) + 1

            # Begrenze die Anzahl der Features
            if len(token_freq) > self.sparse_max_features:
                # Behalte die häufigsten Features
                sorted_tokens = sorted(
                    token_freq.items(), key=lambda x: x[1], reverse=True
                )
                token_freq = dict(sorted_tokens[: self.sparse_max_features])

            # Erstelle Sparse-Vektor als Dictionary
            sparse_vector = {}
            for idx, (token, freq) in enumerate(token_freq.items()):
                # Normalisiere die Frequenz
                normalized_freq = min(freq / max(token_freq.values()), 1.0)
                sparse_vector[str(idx)] = normalized_freq

            return sparse_vector

        except Exception as e:
            self.logger.error(f"Fehler bei der Sparse Embedding-Erzeugung: {e}")
            # Fallback: leerer Sparse-Vektor
            return {}

    async def get_hybrid_embeddings(
        self, text: str
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Generiert sowohl Dense als auch Sparse Embeddings für einen Text

        Args:
            text: Text für die Embeddings

        Returns:
            Tuple aus (Dense Embedding, Sparse Embedding)
        """
        try:
            # Generiere Dense Embedding
            dense_embedding = await self.get_embedding_from_external_api(text)

            # Generiere Sparse Embedding
            sparse_embedding = self.generate_sparse_embedding(text)

            return dense_embedding, sparse_embedding

        except Exception as e:
            self.logger.error(f"Fehler bei der Hybrid Embedding-Erzeugung: {e}")
            # Fallback: Dummy-Vektoren
            return [0.0] * 1024, {}

    async def process_chunks_with_embeddings(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Verarbeitet eine Liste von Chunks und fügt sowohl Dense als auch Sparse Embeddings hinzu

        Args:
            chunks: Liste von Chunks ohne Embeddings

        Returns:
            Liste von Chunks mit Dense und Sparse Embeddings
        """
        processed_chunks = []

        for chunk in chunks:
            try:
                # Generiere sowohl Dense als auch Sparse Embeddings
                dense_embedding, sparse_embedding = await self.get_hybrid_embeddings(
                    chunk["content"]
                )

                # Füge Embeddings zum Chunk hinzu
                chunk_with_embeddings = chunk.copy()
                chunk_with_embeddings["dense_vector"] = dense_embedding
                chunk_with_embeddings["sparse_vector"] = sparse_embedding

                processed_chunks.append(chunk_with_embeddings)

            except Exception as e:
                self.logger.error(f"Fehler bei der Verarbeitung eines Chunks: {e}")
                # Füge Chunk mit Dummy-Embeddings hinzu
                chunk_with_embeddings = chunk.copy()
                chunk_with_embeddings["dense_vector"] = [0.0] * 1024  # Dummy-Vektor
                chunk_with_embeddings["sparse_vector"] = {}  # Leerer Sparse-Vektor
                processed_chunks.append(chunk_with_embeddings)

        self.logger.info(
            f"Verarbeitet {len(processed_chunks)} Chunks mit Dense und Sparse Embeddings"
        )
        return processed_chunks

    async def process_chunks_with_bge_m3_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        include_dense: bool = True,
        include_sparse: bool = True,
        include_multivector: bool = True,
        batch_size: int = 32,
        cache_embeddings: bool = True,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Verarbeitet eine Liste von Chunks mit BGE-M3 Embeddings (dense, sparse, multivector)

        Args:
            chunks: Liste von Chunks ohne Embeddings
            include_dense: Dense Embeddings generieren
            include_sparse: Sparse Embeddings generieren
            include_multivector: Multi-Vector Embeddings generieren
            batch_size: Batch Größe für Verarbeitung
            cache_embeddings: Embeddings cachen
            session_id: Session ID für Caching

        Returns:
            Liste von Chunks mit BGE-M3 Embeddings
        """
        processed_chunks = []
        
        try:
            # Importiere BGE-M3 Service
            from src.app.services.bge_m3_service import BGE_M3_Service
            
            # Initialisiere BGE-M3 Service
            bge_m3_service = BGE_M3_Service()
            
            if not bge_m3_service.is_available():
                self.logger.warning("BGE-M3 Service nicht verfügbar, verwende Standard-Embeddings")
                return await self.process_chunks_with_embeddings(chunks)
            
            self.logger.info(f"Verarbeite {len(chunks)} Chunks mit BGE-M3 Embeddings")
            
            # Verarbeite Chunks in Batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk["content"] for chunk in batch_chunks]
                
                try:
                    # Generiere BGE-M3 Embeddings für den Batch
                    batch_embeddings = await bge_m3_service.batch_generate_embeddings(
                        texts=batch_texts,
                        include_dense=include_dense,
                        include_sparse=include_sparse,
                        include_multivector=include_multivector,
                        cache_embeddings=cache_embeddings,
                        session_id=session_id
                    )
                    
                    # Füge Embeddings zu den Chunks hinzu
                    for j, chunk in enumerate(batch_chunks):
                        chunk_with_embeddings = chunk.copy()
                        
                        # Füge Embeddings basierend auf Verfügbarkeit hinzu
                        if j < len(batch_embeddings):
                            embeddings = batch_embeddings[j]
                            
                            if include_dense and embeddings.get("dense"):
                                chunk_with_embeddings["dense_vector"] = embeddings["dense"]
                            else:
                                chunk_with_embeddings["dense_vector"] = [0.0] * 1024
                            
                            if include_sparse and embeddings.get("sparse"):
                                chunk_with_embeddings["sparse_vector"] = embeddings["sparse"]
                            else:
                                chunk_with_embeddings["sparse_vector"] = {}
                            
                            if include_multivector and embeddings.get("multivector"):
                                chunk_with_embeddings["multivector"] = embeddings["multivector"]
                            else:
                                chunk_with_embeddings["multivector"] = []
                            
                            # Füge Fehlerinformationen hinzu
                            if embeddings.get("errors"):
                                chunk_with_embeddings["embedding_errors"] = embeddings["errors"]
                        
                        processed_chunks.append(chunk_with_embeddings)
                        
                except Exception as e:
                    self.logger.error(f"Fehler bei der Batch-Verarbeitung {i//batch_size + 1}: {e}")
                    # Fallback zur Standard-Verarbeitung für diesen Batch
                    fallback_chunks = await self.process_chunks_with_embeddings(batch_chunks)
                    processed_chunks.extend(fallback_chunks)
            
            self.logger.info(
                f"Verarbeitet {len(processed_chunks)} Chunks mit BGE-M3 Embeddings "
                f"(dense={include_dense}, sparse={include_sparse}, multivector={include_multivector})"
            )
            
        except Exception as e:
            self.logger.error(f"Fehler bei der BGE-M3 Verarbeitung: {e}")
            # Fallback zur Standard-Verarbeitung
            self.logger.info("Fallback zur Standard-Embedding-Verarbeitung")
            processed_chunks = await self.process_chunks_with_embeddings(chunks)
        
        return processed_chunks

    async def get_bge_m3_embeddings_for_text(
        self,
        text: str,
        include_dense: bool = True,
        include_sparse: bool = True,
        include_multivector: bool = True,
        cache_embeddings: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generiert BGE-M3 Embeddings für einen einzelnen Text

        Args:
            text: Text für die Embeddings
            include_dense: Dense Embeddings generieren
            include_sparse: Sparse Embeddings generieren
            include_multivector: Multi-Vector Embeddings generieren
            cache_embeddings: Embeddings cachen
            session_id: Session ID für Caching

        Returns:
            Dictionary mit Embeddings und Metadaten
        """
        try:
            # Importiere BGE-M3 Service
            from src.app.services.bge_m3_service import BGE_M3_Service
            
            # Initialisiere BGE-M3 Service
            bge_m3_service = BGE_M3_Service()
            
            if not bge_m3_service.is_available():
                self.logger.warning("BGE-M3 Service nicht verfügbar")
                return {
                    "dense": [0.0] * 1024,
                    "sparse": {},
                    "multivector": [],
                    "errors": ["BGE-M3 Service nicht verfügbar"]
                }
            
            # Generiere BGE-M3 Embeddings
            embeddings = await bge_m3_service.generate_embeddings(
                text=text,
                include_dense=include_dense,
                include_sparse=include_sparse,
                include_multivector=include_multivector,
                cache_embeddings=cache_embeddings,
                session_id=session_id
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Fehler bei der BGE-M3 Embedding-Erzeugung: {e}")
            return {
                "dense": [0.0] * 1024,
                "sparse": {},
                "multivector": [],
                "errors": [str(e)]
            }

    async def get_cached_embeddings(
        self,
        text: str,
        session_id: Optional[str] = None,
        embedding_type: str = "dense"
    ) -> Optional[List[float]]:
        """
        Ruft gecachte Embeddings ab, falls verfügbar

        Args:
            text: Text für den Cache-Lookup
            session_id: Session ID für Cache
            embedding_type: Typ des Embeddings (dense, sparse, multivector)

        Returns:
            Gecachte Embeddings oder None, nicht gefunden
        """
        try:
            # Importiere Cache-Service
            from src.app.services.cache_service import CacheService
            
            cache_service = CacheService()
            cache_key = f"bge_m3_{embedding_type}_{hash(text)}"
            
            if session_id:
                cache_key = f"{session_id}_{cache_key}"
            
            cached_embedding = await cache_service.get(cache_key)
            
            if cached_embedding:
                self.logger.info(f"Cache hit für {embedding_type}-Embedding")
                return cached_embedding
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache-Lookup fehlgeschlagen: {e}")
            return None

    async def cache_embeddings(
        self,
        text: str,
        embeddings: Dict[str, Any],
        session_id: Optional[str] = None,
        ttl: int = 3600  # 1 Stunde Standard-TTL
    ) -> bool:
        """
        Speichert Embeddings im Cache

        Args:
            text: Ursprünglicher Text
            embeddings: Embeddings zum Speichern
            session_id: Session ID für Cache
            ttl: Time-to-live in Sekunden

        Returns:
            True, wenn Speicherung erfolgreich
        """
        try:
            # Importiere Cache-Service
            from src.app.services.cache_service import CacheService
            
            cache_service = CacheService()
            
            # Speichere verschiedene Embedding-Typen
            for embedding_type, embedding_data in embeddings.items():
                if embedding_data:  # Nur nicht-leere Embeddings speichern
                    cache_key = f"bge_m3_{embedding_type}_{hash(text)}"
                    
                    if session_id:
                        cache_key = f"{session_id}_{cache_key}"
                    
                    await cache_service.set(cache_key, embedding_data, ttl)
                    self.logger.debug(f"Embedding {embedding_type} gecached")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Cache-Speicherung fehlgeschlagen: {e}")
            return False

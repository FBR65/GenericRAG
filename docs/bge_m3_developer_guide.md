
# BGE-M3 Developer Guide

## Übersicht

Dieses Dokument richtet sich an Entwickler, die die BGE-M3-Funktionalität im GenericRAG erweitern, anpassen oder debuggen möchten. Es bietet eine detaillierte Erklärung der Code-Struktur, Best Practices für die Erweiterung und umfassende Informationen zum Testing und Debugging.

## Code Struktur

### Projektübersicht

Die BGE-M3-Integration ist modular aufgebaut und folgt sauberen Architekturprinzipien:

```
src/app/
├── services/
│   ├── bge_m3_service.py          # Haupt-BGE-M3 Service
│   ├── search_service.py          # Erweiterter Such-Service
│   └── ...
├── utils/
│   ├── qdrant_utils.py            # Qdrant Hilfsfunktionen
│   └── ...
├── api/
│   └── endpoints/
│       ├── ingest.py              # Ingest Endpoints
│       ├── query.py               # Query Endpoints
│       └── ...
├── models/
│   └── schemas.py                 # Pydantic Schemas
└── settings.py                    # Konfiguration
```

### Kernkomponenten im Detail

#### 1. BGE_M3_Service (`src/app/services/bge_m3_service.py`)

Der zentrale Service für alle BGE-M3-Operationen:

```python
class BGE_M3_Service:
    """
    Hauptservice für BGE-M3 Operationen mit Unterstützung für:
    - Dense Embedding Generierung
    - Sparse Embedding Generierung  
    - Multi-Vector Embedding Generierung
    - Batch-Verarbeitung
    - Caching und Fehlerbehandlung
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bge_m3_settings = settings.bge_m3
        self.cache_manager = CacheManager(settings)
        self.error_handler = ErrorHandler(settings)
        self.model_client = self._initialize_model()
        
    async def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """Generiert alle drei Embedding-Typen für einen Text"""
        
    async def batch_generate_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch-Generierung von Embeddings"""
        
    async def health_check(self) -> Dict[str, Any]:
        """Health Check für den Service"""
```

**Wichtige Methoden:**

- `generate_dense_embedding(text: str) -> List[float]`: Generiert Dense Embeddings
- `generate_sparse_embedding(text: str) -> Dict[str, float]`: Generiert Sparse Embeddings
- `generate_multivector_embedding(text: str) -> List[List[float]]`: Generiert Multi-Vektoren
- `generate_embeddings(text: str) -> Dict[str, Any]`: Generiert alle Typen
- `batch_generate_embeddings(texts: List[str]) -> List[Dict[str, Any]]`: Batch-Verarbeitung

#### 2. SearchService (`src/app/services/search_service.py`)

Erweitert für BGE-M3-Unterstützung:

```python
class SearchService:
    """
    Such-Service mit BGE-M3 Unterstützung für:
    - Hybride Suche mit drei-Phasen-Ansatz
    - Text- und Bildsuche
    - Ergebnis-Kombination und Ranking
    """
    
    def __init__(self, qdrant_client: AsyncQdrantClient, image_storage, settings: Settings):
        self.qdrant_client = qdrant_client
        self.image_storage = image_storage
        self.settings = settings
        self.bge_m3_service = BGE_M3_Service(settings)
        
    async def perform_hybrid_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Führt hybride Suche mit BGE-M3 durch"""
        
    async def bge_m3_hybrid_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Spezifische BGE-M3 hybride Suche mit drei-Phasen-Ansatz"""
```

**Wichtige Methoden:**

- `perform_hybrid_search()`: Allgemeine hybride Suche
- `bge_m3_hybrid_search()`: BGE-M3 optimierte Suche
- `get_dense_embedding()`: Generiert Dense Embeddings mit Fallback
- `get_bge_m3_embeddings()`: Generiert alle BGE-M3 Embeddings
- `get_bge_m3_embeddings_batch()`: Batch-Verarbeitung

#### 3. Qdrant Utils (`src/app/utils/qdrant_utils.py`)

Hilfsfunktionen für Qdrant-Operationen:

```python
# BGE-M3 spezifische Funktionen
async def bge_m3_hybrid_search_with_retry(...)
async def create_bge_m3_collection_if_not_exists(...)
async def upsert_hybrid_chunks_with_retry(...)
```

**Wichtige Funktionen:**

- `bge_m3_hybrid_search_with_retry()`: Drei-Phasen-Suche mit Retry
- `create_bge_m3_collection_if_not_exists()`: Collection-Erstellung
- `upsert_hybrid_chunks_with_retry()`: Upsert mit Retry-Logik
- `create_hybrid_point()`: Erstellt hybride Punkte mit allen Vektortypen

#### 4. Cache Manager (`src/app/services/bge_m3_service.py`)

Redis-basierter Caching für Embeddings:

```python
class CacheManager:
    """Redis-basierter Cache für Embeddings mit Batch-Operationen"""
    
    def __init__(self, settings: Settings):
        self.redis_client = self._initialize_redis()
        self.settings = settings
        
    async def get_embedding(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Holt ein Embedding aus dem Cache"""
        
    async def set_embedding(self, cache_key: str, value: Dict[str, Any]) -> bool:
        """Speichert ein Embedding im Cache"""
        
    async def batch_get_embeddings(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Batch-Cache Retrieval"""
        
    async def batch_set_embeddings(self, key_values: Dict[str, Dict[str, Any]]) -> bool:
        """Batch-Cache Storage"""
```

#### 5. Error Handler (`src/app/services/bge_m3_service.py`)

Robuste Fehlerbehandlung:

```python
class ErrorHandler:
    """
    Fehlerbehandlung mit:
    - Retry-Logik mit exponentiellem Backoff
    - Circuit Breaker Pattern
    - Fallback-Mechanismen
    """
    
    def __init__(self, settings: BGE_M3_Settings):
        self.settings = settings
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0
        
    @handle_errors
    async def some_operation(self, ...):
        """Operation mit automatischem Error Handling"""
        
    async def handle_embedding_error(self, error: Exception, text: str, mode: str) -> Dict[str, Any]:
        """Behandelt Embedding Fehler mit Fallback"""
```

## Erweiterung der BGE-M3 Funktionalität

### 1. Neue Embedding-Modi hinzufügen

Um neue Embedding-Modi hinzuzufügen, erweitern Sie den `BGE_M3_Service`:

```python
class BGE_M3_Service:
    # ... bestehender Code
    
    async def generate_custom_embedding(self, text: str, mode: str) -> Any:
        """
        Generiert Custom Embeddings für einen spezifischen Modus
        
        Args:
            text: Eingabetext
            mode: Embedding-Modus
            
        Returns:
            Embedding-Vektor oder Dictionary
        """
        try:
            # Validiere den Modus
            if mode not in self.get_embedding_modes():
                raise ValueError(f"Unbekannter Embedding-Modus: {mode}")
                
            # Cache-Check
            cache_key = f"custom_{mode}_{hash(text)}"
            cached_result = await self.cache_manager.get_embedding(cache_key)
            if cached_result:
                return cached_result
                
            # Generiere das Embedding
            embedding = await self._generate_custom_embedding(text, mode)
            
            # Cache das Ergebnis
            await self.cache_manager.set_embedding(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            return await self.error_handler.handle_embedding_error(e, text, mode)
    
    async def _generate_custom_embedding(self, text: str, mode: str) -> Any:
        """Interne Methode für Custom Embedding-Generierung"""
        # Implementieren Sie hier Ihre Logik
        pass
    
    def get_embedding_modes(self) -> List[str]:
        """Gibt verfügbare Embedding-Modi zurück"""
        return ["dense", "sparse", "multi_vector", "custom"]
```

### 2. Neue Suchstrategien implementieren

Erweitern Sie den `SearchService` für neue Suchstrategien:

```python
class SearchService:
    # ... bestehender Code
    
    async def perform_custom_search(self, query: str, strategy: str, **kwargs) -> List[SearchResult]:
        """
        Führt eine benutzerdefinierte Suche durch
        
        Args:
            query: Suchanfrage
            strategy: Suchstrategie
            **kwargs: Zusätzliche Parameter
            
        Returns:
            Liste von Suchergebnissen
        """
        try:
            # Validiere die Strategie
            if strategy not in self.get_search_strategies():
                raise ValueError(f"Unbekannte Suchstrategie: {strategy}")
                
            # Generiere Embeddings
            embeddings = await self.get_bge_m3_embeddings(query)
            
            # Führe die spezifische Suche durch
            results = await self._execute_custom_search(query, embeddings, strategy, **kwargs)
            
            # Formatieren und zurückgeben
            return await self._format_search_results(results)
            
        except Exception as e:
            logger.error(f"Fehler bei benutzerdefinierter Suche: {e}")
            raise
    
    async def _execute_custom_search(self, query: str, embeddings: Dict[str, Any], 
                                   strategy: str, **kwargs) -> List[Dict[str, Any]]:
        """Interne Methode für benutzerdefinierte Suche"""
        # Implementieren Sie hier Ihre Suchlogik
        pass
    
    def get_search_strategies(self) -> List[str]:
        """Gibt verfügbare Suchstrategien zurück"""
        return ["text_only", "image_only", "hybrid", "full_bge_m3", "custom"]
```

### 3. Custom Post-Processing hinzufügen

Fügen Sie benutzerdefiniertes Post-Processing für Suchergebnisse hinzu:

```python
class SearchService:
    # ... bestehender Code
    
    async def apply_custom_post_processing(self, results: List[SearchResult], 
                                         processing_config: Dict[str, Any]) -> List[SearchResult]:
        """
        Wendet benutzerdefiniertes Post-Processing auf Suchergebnisse an
        
        Args:
            results: Roh-Suchergebnisse
            processing_config: Konfiguration für das Processing
            
        Returns:
            Verarbeitete Suchergebnisse
        """
        try:
            if not processing_config:
                return results
                
            processed_results = []
            
            for result in results:
                processed_result = await self._process_single_result(result, processing_config)
                processed_results.append(processed_result)
                
            return processed_results
            
        except Exception as e:
            logger.error(f"Fehler bei Post-Processing: {e}")
            return results
    
    async def _process_single_result(self, result: SearchResult, 
                                   config: Dict[str, Any]) -> SearchResult:
        """Verarbeitet ein einzelnes Suchergebnis"""
        # Implementieren Sie hier Ihre Post-Processing Logik
        return result
```

### 4. Neue Metriken und Monitoring

Erweitern Sie das Monitoring für neue Metriken:

```python
class BGE_M3_Service:
    # ... bestehender Code
    
    def __init__(self, settings: Settings):
        # ... bestehender Code
        
        # Metriken initialisieren
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0.0,
            "last_request_time": None
        }
    
    async def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """Mit Metriken erweiterte Embedding-Generierung"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            result = await self._generate_embeddings_internal(text)
            self.metrics["successful_requests"] += 1
            self.metrics["cache_hits"] += result.get("cache_hits", 0)
            self.metrics["cache_misses"] += result.get("cache_misses", 0)
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            raise
            
        finally:
            processing_time = time.time() - start_time
            self.metrics["last_request_time"] = processing_time
            self._update_average_processing_time(processing_time)
            
        return result
    
    def _update_average_processing_time(self, new_time: float):
        """Aktualisiert die durchschnittliche Verarbeitungszeit"""
        current_avg = self.metrics["average_processing_time"]
        total_requests = self.metrics["total_requests"]
        
        if total_requests == 1:
            self.metrics["average_processing_time"] = new_time
        else:
            self.metrics["average_processing_time"] = (
                (current_avg * (total_requests - 1) + new_time) / total_requests
            )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Gibt aktuelle Metriken zurück"""
        return {
            "metrics": self.metrics.copy(),
            "timestamp": datetime.utcnow().isoformat(),
            "service_status": await self.health_check()
        }
```

##
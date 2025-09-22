# BGE-M3 API Endpoints

## Übersicht

Diese Dokumentation beschreibt die neuen API Endpoints, die für die BGE-M3 Unterstützung hinzugefügt wurden. BGE-M3 unterstützt drei verschiedene Embedding-Typen:

- **Dense Embeddings**: Kontextuelle Vektoren für semantische Suche
- **Sparse Embeddings**: Bag-of-Words Vektoren für lexikalische Suche  
- **Multi-Vector Embeddings**: ColBERT-ähnliche Vektoren für granulare Suche

## Query Endpoints

### `/api/v1/query` (Erweitert)

**Beschreibung**: Erweiterter Query-Endpoint mit BGE-M3 Unterstützung

**Method**: `POST`

**Parameter**:
```json
{
  "query": "Suchanfrage",
  "use_bge_m3": true,
  "search_mode": "hybrid",
  "alpha": 0.4,
  "beta": 0.3,
  "gamma": 0.3,
  "top_k": 10,
  "score_threshold": 0.5,
  "include_images": true,
  "session_id": "optional-session-id",
  "page": 1,
  "page_size": 10
}
```

**Parameter Details**:
- `use_bge_m3`: Verwendet BGE-M3 für Embedding-Generierung (default: `false`)
- `search_mode`: Suchstrategie ("dense", "sparse", "multivector", "hybrid")
- `alpha`: Gewicht für Dense-Suche (0.0-1.0)
- `beta`: Gewicht für Sparse-Suche (0.0-1.0)
- `gamma`: Gewicht für Multi-Vector Reranking (0.0-1.0)

**Antwort**:
```json
{
  "results": [
    {
      "id": "result-id",
      "score": 0.95,
      "document": "Dokumentinhalt",
      "page": 1,
      "image": null,
      "metadata": {
        "session_id": "session-id",
        "created_at": "2024-01-01T00:00:00Z",
        "search_type": "bge_m3_hybrid",
        "confidence": "high",
        "vector_types": ["dense", "sparse", "multivector"],
        "element_type": "text"
      }
    }
  ],
  "total": 10,
  "page": 1,
  "page_size": 10
}
```

### `/api/v1/query-bge-m3` (Neu)

**Beschreibung**: Spezieller Endpoint für BGE-M3 Suchfunktionen

**Method**: `POST`

**Parameter**:
```json
{
  "query": "Suchanfrage",
  "search_strategy": "full_bge_m3",
  "alpha": 0.4,
  "beta": 0.3,
  "gamma": 0.3,
  "top_k": 10,
  "score_threshold": 0.5,
  "include_images": true,
  "session_id": "optional-session-id",
  "page": 1,
  "page_size": 10
}
```

**Suchstrategien**:
- `dense_only`: Nur Dense Embeddings
- `sparse_only`: Nur Sparse Embeddings
- `full_bge_m3`: Alle drei Embedding-Typen mit Hybrid-Suche

**Antwort**: Erweitertes Antwortformat mit BGE-M3 spezifischen Metadaten

### `/api/v1/query-stream` (Erweitert)

**Beschreibung**: Streaming-Suche mit BGE-M3 Unterstützung

**Method**: `POST`

**Parameter**: Gleiche wie `/query` mit zusätzlichen Streaming-Parametern

**Antwort**: Streaming-Antwort mit asynchronen Suchergebnissen

## Ingest Endpoints

### `/api/v1/ingest` (Erweitert)

**Beschreibung**: Erweiterter Ingest-Endpoint mit BGE-M3 Unterstützung

**Method**: `POST`

**Parameter**:
```json
{
  "file": "pdf-file",
  "session_id": "optional-session-id",
  "use_bge_m3": true,
  "embedding_types": "all"
}
```

**Parameter Details**:
- `use_bge_m3`: Verwendet BGE-M3 für Embedding-Generierung (default: `false`)
- `embedding_types`: Embedding-Typen ("dense", "sparse", "multivector", "all")

**Antwort**:
```json
{
  "filename": "document.pdf",
  "num_pages": 10,
  "status": "success",
  "metadata": {
    "chunks_processed": 50,
    "images_processed": 5,
    "bge_m3_enabled": true,
    "embedding_types": ["dense", "sparse", "multivector"]
  }
}
```

### `/api/v1/ingest-bge-m3` (Neu)

**Beschreibung**: Spezieller Endpoint für BGE-M3 Ingest

**Method**: `POST`

**Parameter**:
```json
{
  "file": "pdf-file",
  "session_id": "optional-session-id",
  "include_dense": true,
  "include_sparse": true,
  "include_multivector": true,
  "batch_size": 32,
  "cache_embeddings": true
}
```

**Features**:
- Generiert alle drei Embedding-Typen
- Optimiert für Batch-Verarbeitung
- Fortschrittsmonitoring
- Caching-Unterstützung

**Antwort**: Detaillierte Antwort mit Embedding-Statistiken

### `/api/v1/ingest-batch` (Erweitert)

**Beschreibung**: Batch-Verarbeitung mit BGE-M3 Unterstützung

**Method**: `POST`

**Parameter**:
```json
{
  "files": ["file1.pdf", "file2.pdf"],
  "session_id": "optional-session-id",
  "use_bge_m3": true,
  "embedding_types": "all",
  "batch_size": 10,
  "progress_callback": "optional-callback-url"
}
```

**Features**:
- Effiziente Verarbeitung großer Dokumentmengen
- Fortschrittsberichterstattung
- Fehlerbehandlung für einzelne Dokumente
- Asynchrone Verarbeitung

## BGE-M3 Spezifische Schemas

### BGE_M3QueryRequest

```json
{
  "query": "Suchanfrage",
  "use_bge_m3": true,
  "search_mode": "hybrid",
  "alpha": 0.4,
  "beta": 0.3,
  "gamma": 0.3,
  "top_k": 10,
  "score_threshold": 0.5,
  "include_images": true,
  "session_id": "session-id",
  "page": 1,
  "page_size": 10
}
```

### BGE_M3IngestRequest

```json
{
  "file": "pdf-file",
  "session_id": "session-id",
  "use_bge_m3": true,
  "embedding_types": "all",
  "include_dense": true,
  "include_sparse": true,
  "include_multivector": true,
  "batch_size": 32,
  "cache_embeddings": true
}
```

### BGE_M3SearchResult

```json
{
  "id": "result-id",
  "score": 0.95,
  "document": "Dokumentinhalt",
  "page": 1,
  "image": null,
  "metadata": {
    "session_id": "session-id",
    "created_at": "2024-01-01T00:00:00Z",
    "search_type": "bge_m3_hybrid",
    "confidence": "high",
    "vector_types": ["dense", "sparse", "multivector"],
    "element_type": "text",
    "vector_completeness": 1.0,
    "metadata_quality": 1.0
  }
}
```

## Fehlerbehandlung

### BGE-M3 Service Fehler

Die BGE-M3 Endpoints implementieren robuste Fehlerbehandlung:

1. **Circuit Breaker Pattern**: Verhindert Überlastung bei Service-Ausfällen
2. **Retry Logic**: Automatische Wiederholung bei temporären Fehlern
3. **Fallback Mechanismen**: Verwendet Standard-Embeddings bei BGE-M3 Ausfällen
4. **Graceful Degradation**: System bleibt auch ohne BGE-M3 funktionsfähig

### Fehlercodes

- `422`: Ungültige Parameter
- `429`: Rate Limiting
- `500`: Interne Serverfehler
- `503`: BGE-M3 Service nicht verfügbar

## Performance Optimierungen

### Caching

- **Redis Cache**: Zwischenspeicherung von Embeddings
- **TTL**: 1 Stunde Standard
- **Hit Rate Monitoring**: Automatische Statistiken

### Batch Processing

- **Asynchrone Verarbeitung**: Parallele Embedding-Generierung
- **Gruppierung**: Effiziente Verarbeitung von Texten
- **Memory Management**: Verhindert Speicherüberlastung

### Rate Limiting

- **Anzahl Anfragen**: Begrenzung pro Minute
- **Concurrent Requests**: Maximale gleichzeitige Anfragen
- **Priorisierung**: Wichtige Anfragen werden bevorzugt

## Integration in bestehende Workflows

### Abwärtskompatibilität

- Alle bestehenden Endpoints bleiben funktionsfähig
- BGE-M3 ist optional und kann deaktiviert werden
- Fallback zu Standard-Embeddings bei Ausfällen

### Migration

1. **Testmodus**: Aktivieren Sie BGE-M3 mit `use_bge_m3: true`
2. **Vergleich**: Vergleichen Sie Ergebnisse mit Standard-Suche
3. **Deployment**: Schalten Sie schrittweise auf BGE-M3 um
4. **Monitoring**: Überwachen Sie Performance und Qualität

### Konfiguration

```python
# settings.py
BGE_M3_Settings(
    model_name="BAAI/bge-m3",
    model_device="auto",
    max_length=8192,
    cache_enabled=True,
    max_retries=3,
    batch_size=32
)
```

## Beispiele

### Beispiel 1: Hybride Suche mit BGE-M3

```python
import requests

response = requests.post("http://localhost:8000/api/v1/query", json={
    "query": "Künstliche Intelligenz Anwendungen",
    "use_bge_m3": True,
    "search_mode": "hybrid",
    "alpha": 0.4,
    "beta": 0.3,
    "gamma": 0.3,
    "top_k": 10
})

results = response.json()
```

### Beispiel 2: Batch-Ingest mit BGE-M3

```python
import requests

files = [("files", open("doc1.pdf", "rb")), ("files", open("doc2.pdf", "rb"))]

response = requests.post("http://localhost:8000/api/v1/ingest-batch", 
                        files=files,
                        data={
                            "use_bge_m3": True,
                            "embedding_types": "all",
                            "batch_size": 10
                        })

result = response.json()
```

## Troubleshooting

### Häufige Probleme

1. **BGE-M3 Service nicht verfügbar**
   - Überprüfen Sie die Konfiguration in `settings.py`
   - Stellen Sie sicher, dass das Modell heruntergeladen ist
   - Überprüfen Sie die Verfügbarkeit der Hardware

2. **Cache Fehler**
   - Überprüfen Sie Redis-Verbindung
   - Überprüfen Sie Speicherplatz
   - Testen Sie mit `cache_enabled=False`

3. **Performance Probleme**
   - Reduzieren Sie die Batch-Größe
   - Überprüfen Sie die Netzwerklatenz
   - Aktivieren Sie Caching

### Debugging

```python
# Health Check
response = requests.get("http://localhost:8000/health")
print(response.json())

# BGE-M3 Service Health
response = requests.post("http://localhost:8000/api/v1/query-bge-m3/health")
print(response.json())
```

## Weiterführende Links

- [BGE-M3 Paper](https://arxiv.org/abs/2402.03216)
- [Qdrant Hybrid Search](https://qdrant.tech/documentation/hybrid-search/)
- [Sentence Transformers](https://www.sbert.net/)
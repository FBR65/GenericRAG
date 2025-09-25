# BGE-M3 API Reference

## Übersicht

Dieses Dokument beschreibt alle neuen API Endpoints und Funktionen, die durch die BGE-M3-Integration im GenericRAG hinzugefügt wurden. Die API bietet umfassende Unterstützung für die Generierung verschiedener Embedding-Typen, hybride Suche und Batch-Operationen.

## Neue Endpoints

### 1. Embedding Endpoints

#### POST `/api/v1/embeddings/bge-m3/dense`

Generiert Dense Embeddings für einen Text mit BGE-M3.

**Request:**
```json
{
  "text": "Beispieltext für die Embedding-Generierung",
  "use_cache": true,
  "cache_ttl": 3600
}
```

**Parameters:**
| Parameter | Typ | Erforderlich | Beschreibung |
|-----------|-----|-------------|-------------|
| `text` | string | Ja | Der Text für den Embedding-Vektor |
| `use_cache` | boolean | Nein | Ob Caching verwendet werden soll (Standard: true) |
| `cache_ttl` | integer | Nein | Cache-Lebensdauer in Sekunden (Standard: 3600) |

**Response:**
```json
{
  "success": true,
  "dense": [0.1, 0.2, 0.3, ...],
  "dimension": 1024,
  "cache_hit": false,
  "processing_time": 0.45
}
```

**Status Codes:**
- `200`: Erfolgreiche Generierung
- `400`: Ungültige Eingabe
- `500`: Serverfehler

#### POST `/api/v1/embeddings/bge-m3/sparse`

Generiert Sparse Embeddings für einen Text mit BGE-M3.

**Request:**
```json
{
  "text": "Beispieltext für die Sparse-Embedding-Generierung",
  "max_features": 1000,
  "use_cache": true
}
```

**Parameters:**
| Parameter | Typ | Erforderlich | Beschreibung |
|-----------|-----|-------------|-------------|
| `text` | string | Ja | Der Text für den Sparse-Vektor |
| `max_features` | integer | Nein | Maximale Anzahl von Features (Standard: 1000) |
| `use_cache` | boolean | Nein | Ob Caching verwendet werden soll (Standard: true) |

**Response:**
```json
{
  "success": true,
  "sparse": {
    "0": 0.5,
    "15": 0.8,
    "42": 0.3,
    ...
  },
  "feature_count": 156,
  "cache_hit": false,
  "processing_time": 0.32
}
```

#### POST `/api/v1/embeddings/bge-m3/multi-vector`

Generiert Multi-Vector Embeddings (ColBERT) für einen Text mit BGE-M3.

**Request:**
```json
{
  "text": "Beispieltext für die Multi-Vector-Generierung",
  "vector_count": 16,
  "vector_dimension": 128,
  "use_cache": true
}
```

**Parameters:**
| Parameter | Typ | Erforderlich | Beschreibung |
|-----------|-----|-------------|-------------|
| `text` | string | Ja | Der Text für die Multi-Vektoren |
| `vector_count` | integer | Nein | Anzahl der Vektoren (Standard: 16) |
| `vector_dimension` | integer | Nein | Dimension jedes Vektors (Standard: 128) |
| `use_cache` | boolean | Nein | Ob Caching verwendet werden soll (Standard: true) |

**Response:**
```json
{
  "success": true,
  "multi_vector": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...],
    ...
  ],
  "vector_count": 16,
  "vector_dimension": 128,
  "cache_hit": false,
  "processing_time": 1.23
}
```

#### POST `/api/v1/embeddings/bge-m3/all`

Generiert alle drei Embedding-Typen (Dense, Sparse, Multi-Vector) für einen Text.

**Request:**
```json
{
  "text": "Beispieltext für alle Embedding-Typen",
  "use_cache": true,
  "cache_ttl": 3600
}
```

**Response:**
```json
{
  "success": true,
  "embeddings": {
    "dense": [0.1, 0.2, 0.3, ...],
    "sparse": {
      "0": 0.5,
      "15": 0.8,
      ...
    },
    "multi_vector": [
      [0.1, 0.2, 0.3, ...],
      [0.4, 0.5, 0.6, ...],
      ...
    ]
  },
  "dimensions": {
    "dense": 1024,
    "sparse": 1000,
    "multi_vector": {
      "count": 16,
      "dimension": 128
    }
  },
  "cache_hits": {
    "dense": false,
    "sparse": false,
    "multi_vector": false
  },
  "processing_time": 2.15
}
```

#### POST `/api/v1/embeddings/bge-m3/batch`

Batch-Generierung von Embeddings für mehrere Texte.

**Request:**
```json
{
  "texts": [
    "Erster Text",
    "Zweiter Text",
    "Dritter Text"
  ],
  "embedding_type": "all",
  "use_cache": true,
  "batch_size": 32
}
```

**Parameters:**
| Parameter | Typ | Erforderlich | Beschreibung |
|-----------|-----|-------------|-------------|
| `texts` | array | Ja | Liste von Texten für die Embedding-Generierung |
| `embedding_type` | string | Nein | Typ des Embeddings ("dense", "sparse", "multi_vector", "all") |
| `use_cache` | boolean | Nein | Ob Caching verwendet werden soll (Standard: true) |
| `batch_size` | integer | Nein | Größe der Batch-Verarbeitung (Standard: 32) |

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "text": "Erster Text",
      "embeddings": {
        "dense": [0.1, 0.2, 0.3, ...],
        "sparse": {"0": 0.5, ...},
        "multi_vector": [[0.1, 0.2, ...], ...]
      },
      "cache_hit": false,
      "processing_time": 0.45
    },
    {
      "text": "Zweiter Text",
      "embeddings": {
        "dense": [0.4, 0.5, 0.6, ...],
        "sparse": {"1": 0.7, ...},
        "multi_vector": [[0.4, 0.5, ...], ...]
      },
      "cache_hit": true,
      "processing_time": 0.02
    }
  ],
  "summary": {
    "total_texts": 3,
    "cache_hits": 1,
    "total_processing_time": 2.15,
    "texts_per_second": 1.39
  }
}
```

### 2. Search Endpoints

#### POST `/api/v1/search/bge-m3/hybrid`

Führt eine hybride Suche mit BGE-M3 durch, die Dense, Sparse und Multi-Vector Embeddings kombiniert.

**Request:**
```json
{
  "query": "Suchanfrage des Benutzers",
  "search_strategy": "full_bge_m3",
  "alpha": 0.4,
  "beta": 0.3,
  "gamma": 0.3,
  "top_k": 10,
  "score_threshold": 0.5,
  "metadata_filters": {
    "document_type": "pdf",
    "page_range": [1, 10]
  },
  "include_images": true,
  "session_id": "session_123",
  "page": 1,
  "page_size": 10
}
```

**Parameters:**
| Parameter | Typ | Erforderlich | Beschreibung |
|-----------|-----|-------------|-------------|
| `query` | string | Ja | Die Suchanfrage |
| `search_strategy` | string | Nein | Suchstrategie ("text_only", "image_only", "hybrid", "full_bge_m3") |
| `alpha` | float | Nein | Gewicht für Dense Suche (Standard: 0.4) |
| `beta` | float | Nein | Gewicht für Sparse Suche (Standard: 0.3) |
| `gamma` | float | Nein | Gewicht für Multi-Vector Reranking (Standard: 0.3) |
| `top_k` | integer | Nein | Anzahl der Ergebnisse (Standard: 10) |
| `score_threshold` | float | Nein | Mindest-Score für Ergebnisse (Standard: 0.5) |
| `metadata_filters` | object | Nein | Metadaten-Filter |
| `include_images` | boolean | Nein | Bilder in Ergebnissen einbeziehen (Standard: true) |
| `session_id` | string | Nein | Session-ID für Filterung |
| `page` | integer | Nein | Seitennummer (Standard: 1) |
| `page_size` | integer | Nein | Ergebnisse pro Seite (Standard: 10) |

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": "doc_123",
      "score": 0.92,
      "document": "Beispieltext aus dem Dokument...",
      "page": 5,
      "image": null,
      "metadata": {
        "session_id": "session_123",
        "created_at": "2024-01-15T10:30:00Z",
        "search_type": "text",
        "combined_score": 0.92,
        "related_text": ["Verwandter Text 1", "Verwandter Text 2"],
        "bbox": [100, 200, 300, 400],
        "element_type": "paragraph"
      }
    }
  ],
  "summary": {
    "total_results": 10,
    "text_results": 7,
    "image_results": 3,
    "avg_score": 0.78,
    "search_time": 0.85
  },
  "query_info": {
    "query": "Suchanfrage des Benutzers",
    "search_strategy": "full_bge_m3",
    "alpha": 0.4,
    "beta": 0.3,
    "gamma": 0.3
  }
}
```

#### POST `/api/v1/search/bge-m3/text`

Spezifische Textsuche mit BGE-M3 Dense Embeddings.

**Request:**
```json
{
  "query": "Textsuchanfrage",
  "alpha": 0.7,
  "top_k": 20,
  "score_threshold": 0.6
}
```

#### POST `/api/v1/search/bge-m3/image`

Spezifische Bildsuche mit BGE-M3.

**Request:**
```json
{
  "query": "Bildsuchanfrage",
  "text_context": "Kontexttext für die Bildsuche",
  "top_k": 15,
  "score_threshold": 0.5
}
```

### 3. Service Management Endpoints

#### GET `/api/v1/bge-m3/service/info`

Gibt Informationen über den BGE-M3 Service zurück.

**Response:**
```json
{
  "service_name": "BGE-M3 Service",
  "version": "1.0.0",
  "embedding_modes": ["dense", "sparse", "multi_vector", "all"],
  "cache_enabled": true,
  "model_ready": true,
  "health_status": "healthy",
  "configuration": {
    "dense_dimension": 1024,
    "sparse_dimension": 10000,
    "multi_vector_count": 16,
    "multi_vector_dimension": 128,
    "cache_ttl": 3600,
    "max_retries": 3
  }
}
```

#### GET `/api/v1/bge-m3/service/health`

Health Check für den BGE-M3 Service.

**Response:**
```json
{
  "status": "healthy",
  "cache_status": "healthy",
  "model_status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

#### GET `/api/v1/bge-m3/service/cache/stats`

Gibt Cache-Statistiken zurück.

**Response:**
```json
{
  "status": "healthy",
  "memory_usage": "1.5M",
  "hit_rate": 0.75,
  "total_requests": 1000,
  "cache_hits": 750,
  "cache_misses": 250,
  "evictions": 0
}
```

#### POST `/api/v1/bge-m3/service/cache/clear`

Löscht den Cache.

**Request:**
```json
{
  "pattern": "bge_m3_*",
  "all": false
}
```

**Response:**
```json
{
  "success": true,
  "cleared_keys": 150,
  "status": "success"
}
```

### 4. Configuration Endpoints

#### GET `/api/v1/bge-m3/config`

Gibt die aktuelle BGE-M3 Konfiguration zurück.

**Response:**
```json
{
  "model_name": "BGE-M3",
  "model_url": "http://localhost:8000",
  "dense_dimension": 1024,
  "sparse_dimension": 10000,
  "multi_vector_count": 16,
  "multi_vector_dimension": 128,
  "cache_enabled": true,
  "cache_ttl": 3600,
  "max_retries": 3,
  "retry_delay": 1.0,
  "batch_size": 32
}
```

#### PUT `/api/v1/bge-m3/config`

Aktualisiert die BGE-M3 Konfiguration.

**Request:**
```json
{
  "cache_ttl": 7200,
  "batch_size": 64,
  "max_retries": 5
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "updated_config": {
    "cache_ttl": 7200,
    "batch_size": 64,
    "max_retries": 5
  }
}
```

## Response Formate

### Erfolgreiche Antworten

Alle erfolgreichen Antworten folgen diesem allgemeinen Format:

```json
{
  "success": true,
  "data": {
    // spezifische Daten je nach Endpoint
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456",
    "processing_time": 0.45
  }
}
```

### Fehlerhafte Antworten

Fehlerantworten folgen diesem Format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Beschreibung des Fehlers",
    "details": {
      // zusätzliche Fehlerdetails
    }
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456"
  }
}
```

### Häufige Fehlercodes

| Code | Beschreibung |
|------|-------------|
| `INVALID_INPUT` | Ungültige Eingabedaten |
| `MODEL_ERROR` | Fehler bei der Modellverarbeitung |
| `CACHE_ERROR` | Fehler beim Cache-Zugriff |
| `SEARCH_ERROR` | Fehler bei der Suche |
| `CONFIG_ERROR` | Fehler bei der Konfiguration |
| `SERVICE_UNAVAILABLE` | Service nicht verfügbar |

## Code Beispiele

### Python Beispiele

#### Embedding-Generierung

```python
import requests
import json

# Dense Embedding generieren
response = requests.post(
    "http://localhost:8000/api/v1/embeddings/bge-m3/dense",
    json={
        "text": "Beispieltext für die Embedding-Generierung",
        "use_cache": True
    }
)

if response.status_code == 200:
    result = response.json()
    dense_embedding = result["data"]["dense"]
    print(f"Dense Embedding generiert: {len(dense_embedding)} Dimensionen")
else:
    print(f"Fehler: {response.json()}")

# Alle Embedding-Typen generieren
response = requests.post(
    "http://localhost:8000/api/v1/embeddings/bge-m3/all",
    json={
        "text": "Beispieltext für alle Embedding-Typen"
    }
)

if response.status_code == 200:
    result = response.json()
    embeddings = result["data"]["embeddings"]
    print(f"Alle Embedding-Typen generiert:")
    print(f"- Dense: {len(embeddings['dense'])} Dimensionen")
    print(f"- Sparse: {len(embeddings['sparse'])} Features")
    print(f"- Multi-Vector: {len(embeddings['multi_vector'])} Vektoren")
```

#### Hybride Suche

```python
import requests

# Hybride Suche durchführen
response = requests.post(
    "http://localhost:8000/api/v1/search/bge-m3/hybrid",
    json={
        "query": "Suchanfrage des Benutzers",
        "search_strategy": "full_bge_m3",
        "alpha": 0.4,
        "beta": 0.3,
        "gamma": 0.3,
        "top_k": 10,
        "score_threshold": 0.5,
        "metadata_filters": {
            "document_type": "pdf"
        }
    }
)

if response.status_code == 200:
    result = response.json()
    search_results = result["data"]["results"]
    print(f"Suchergebnisse: {len(search_results)} gefunden")
    
    for i, result in enumerate(search_results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Text: {result['document'][:100]}...")
        print(f"   Seite: {result['page']}")
        print(f"   Typ: {result['metadata']['search_type']}")
else:
    print(f"Fehler: {response.json()}")
```

#### Batch-Verarbeitung

```python
import requests
import asyncio
import aiohttp

async def batch_embedding_generation():
    texts = [
        "Erster Text für die Batch-Verarbeitung",
        "Zweiter Text mit verschiedenen Inhalten",
        "Dritter Text für Testzwecke"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            task = session.post(
                "http://localhost:8000/api/v1/embeddings/bge-m3/all",
                json={"text": text}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        for i, response in enumerate(responses):
            if response.status == 200:
                result = await response.json()
                embeddings = result["data"]["embeddings"]
                print(f"Text {i+1}: Generiert mit {len(embeddings['dense'])} Dense Features")
            else:
                print(f"Text {i+1}: Fehler - {response.status}")

# Ausführen
asyncio.run(batch_embedding_generation())
```

### cURL Beispiele

#### Dense Embedding generieren

```bash
curl -X POST "http://localhost:8000/api/v1/embeddings/bge-m3/dense" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Beispieltext für die Embedding-Generierung",
    "use_cache": true
  }'
```

#### Hybride Suche durchführen

```bash
curl -X POST "http://localhost:8000/api/v1/search/bge-m3/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Suchanfrage des Benutzers",
    "search_strategy": "full_bge_m3",
    "alpha": 0.4,
    "beta": 0.3,
    "gamma": 0.3,
    "top_k": 10,
    "score_threshold": 0.5
  }'
```

#### Batch-Embedding generieren

```bash
curl -X POST "http://localhost:8000/api/v1/embeddings/bge-m3/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Erster Text",
      "Zweiter Text",
      "Dritter Text"
    ],
    "embedding_type": "all",
    "use_cache": true
  }'
```

## Rate Limiting

### Limits

| Endpunkt | Anfragen pro Minute | Anfragen pro Stunde |
|----------|-------------------|-------------------|
| Embedding Endpoints | 60 | 1000 |
| Search Endpoints | 30 | 500 |
| Service Management | 120 | 2000 |
| Configuration | 10 | 100 |

### Best Practices

1. **Batch-Operationen nutzen**: Verwenden Sie Batch-Endpoints für mehrere Anfragen
2. **Caching aktivieren**: Nutzen Sie den Cache für wiederkehrende Anfragen
3. **Rate Limits beachten**: Implementieren Sie Exponential Backoff bei Fehlern
4. **Asynchrone Verarbeitung**: Verwenden Sie asynchrone Anwendungen für hohe Last

### Rate Limit Header

Die API gibt folgende Header für Rate Limiting zurück:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
X-RateLimit-Reset: 1642252800
X-RateLimit-Reset-Time: "2024-01-15T10:30:00Z"
```

### Rate Limit Fehler

Bei Überschreitung der Rate Limits erhalten Sie eine `429 Too Many Requests` Antwort:

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again later.",
    "retry_after": 60
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456"
  }
}
```

## Sicherheit

### API Keys

Für den Zugriff auf die BGE-M3 API benötigen Sie einen API Key:

```python
headers = {
    "Authorization": "Bearer your_api_key_here",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/api/v1/embeddings/bge-m3/dense",
    headers=headers,
    json={"text": "Beispieltext"}
)
```

### CORS

Die API unterstützt CORS für Webanwendungen:

```javascript
// Beispiel für CORS-Anfrage
fetch('http://localhost:8000/api/v1/embeddings/bge-m3/dense', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer your_api_key_here'
  },
  body: JSON.stringify({
    text: 'Beispieltext'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

### Input Validation

Die API validiert alle Eingaben:

- **Text**: Muss ein nicht-leerer String sein (max. 8192 Zeichen)
- **Numerische Werte**: Innerhalb definierter Grenzen
- **Arrays**: Nicht leer und korrekt formatiert
- **Objekte**: Korrekte Struktur und Datentypen

## Fehlerbehandlung

### Common Issues

1. **Invalid Input**: Überprüfen Sie die Eingabedaten
2. **Model Error**: Überprüfen Sie die Modellverfügbarkeit
3. **Cache Error**: Überprüfen Sie die Redis-Verbindung
4. **Search Error**: Überprüfen Sie die Qdrant-Verbindung
5. **Rate Limit**: Implementieren Sie Exponential Backoff

### Debugging

Für Debugging-Zwecke können Sie die `debug` Flag aktivieren:

```python
response = requests.post(
    "http://localhost:8000/api/v1/embeddings/bge-m3/dense",
    json={
        "text": "Beispieltext",
        "debug": True
    }
)
```

Dies gibt detaillierte Informationen über die Verarbeitung zurück.

---

*Weitere Dokumente in dieser Reihe:*
- [BGE-M3 Integration Guide](./bge_m3_integration_guide.md)
- [BGE-M3 Developer Guide](./bge_m3_developer_guide.md)
- [BGE-M3 Service Documentation](./services/bge_m3_service.md)
- [BGE-M3 Qdrant Integration](./services/bge_m3_qdrant_integration.md)
- [BGE-M3 Configuration Guide](./bge_m3_configuration.md)
- [BGE-M3 Usage Examples](./bge_m3_examples.md)
- [BGE-M3 Migration Guide](./bge_m3_migration_guide.md)
- [BGE-M3 Monitoring Guide](./bge_m3_monitoring.md)
- [BGE-M3 Troubleshooting Guide](./bge_m3_troubleshooting.md)
- [BGE-M3 Testing Documentation](./bge_m3_testing.md)
- [BGE-M3 Deployment Guide](./bge_m3_deployment.md)
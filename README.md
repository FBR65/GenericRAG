# Generic RAG-System

Ein fortschrittliches Retrieval-Augmented Generation (RAG)-System, das mit ColPali für visuelles Dokumentenverständnis, Qdrant für Vektorspeicherung und FastAPI/Gradio für die API- und Frontend-Schnittstelle aufgebaut ist.

## Funktionen

- **Visuelles Dokumentenverständnis**: Verwendet ColPali-Engine, um PDF-Seiten als Bilder zu verarbeiten
- **Vektordatenbank**: Qdrant mit Kosinus-Distanz für effiziente Ähnlichkeitssuche
- **KI-gestützte Antworten**: LLM-Integration für kontextbezogene Antwortgenerierung
- **Quellenangaben**: Automatische Quellenzuordnung mit Dokument- und Seitenreferenzen
- **Umfassende API**: Volle REST-API mit Upload-, Verarbeitungs-, Such- und Verwaltungsendpoints
- **Benutzerfreundliche Oberfläche**: Gradio-basiertes Frontend mit zwei Tabs für Dokumentenverwaltung und Abfragen
- **Komplett konfigurierbar**: Alle Einstellungen über config.py und Umgebungsvariablen

## Architektur

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │   FastAPI       │    │   Services      │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (Core Logic)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │   Qdrant DB     │
                                               │ (Vector Store)  │
                                               └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │   ColPali       │
                                               │ (Embeddings)    │
                                               └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │   LLM Service   │
                                               │ (OpenAI Compatible)│
                                               └─────────────────┘
```

## Schnellstart

### 1. Installation

```bash
# Repository klonen
git clone <repository-url>
cd GenericRAG

# Abhängigkeiten mit uv installieren
uv sync
```

### 2. Konfiguration

Kopieren Sie die Beispielkonfigurationsdatei und passen Sie sie an:

```bash
cp .env.example .env
```

Bearbeiten Sie `.env` mit Ihrer Konfiguration:

```env
# System Configuration
SYSTEM_NAME=Ihr RAG System
SYSTEM_VERSION=1.0.0
DESCRIPTION=Beschreibung Ihres RAG-Systems

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=ihre_qdrant_api_hier
QDRANT_COLLECTION_NAME=ihre_dokumente

# LLM Configuration
LLM_ENDPOINT=http://localhost:8000/v1/chat/completions
LLM_API_KEY=ihre_llm_api_hier
LLM_MODEL=ihr-modell
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# ColPali Configuration
COLPALI_MODEL_NAME=vidore/colqwen2-v1.0
COLPALI_DEVICE=cuda:0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_TITLE=Ihre RAG API
API_DESCRIPTION=REST API für Ihr RAG-System

# Frontend Configuration
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=7860
FRONTEND_TITLE=Ihre RAG Oberfläche
```

### 3. Dienste starten

#### Option A: Mit Startskripten

```bash
# API-Server starten
python start_api.py

# In einem neuen Terminal das Frontend starten
python start_frontend.py
```

#### Option B: Direkte Ausführung

```bash
# API-Server starten
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend starten
python src/frontend/app.py
```

### 4. Anwendung aufrufen

- **Frontend**: http://localhost:7860
- **API-Dokumentation**: http://localhost:8000/docs
- **API-Health-Check**: http://localhost:8000/health

## Verwendung

### Dokumenten-Upload & Verwaltung

1. Öffnen Sie die Gradio-Oberfläche
2. Gehen Sie zum Tab "Dokumenten-Upload & Verwaltung"
3. Laden Sie PDF-Dateien über die Datei-Upload-Schnittstelle hoch
4. Klicken Sie auf "Upload & Verarbeiten", um die Dokumente zu verarbeiten
5. Sehen Sie sich die verarbeiteten Dokumente in der Dokumentenliste an
6. Verwenden Sie "Ausgewähltes Dokument löschen", um Dokumente zu entfernen

### Abfrageschnittstelle

1. Gehen Sie zum Tab "Abfrageschnittstelle"
2. Geben Sie Ihre Frage in das Abfragefeld ein
3. Passen Sie bei Bedarf die Anzahl der Ergebnisse an
4. Klicken Sie auf "Suchen & Antwort generieren"
5. Sehen Sie sich die generierte Antwort und Quellenangaben an

## API-Endpunkte

### Kernendpunkte

- `GET /` - Root-Endpunkt
- `GET /health` - Health-Check
- `GET /system-status` - Systemstatus

### Dokumentenverwaltung

- `POST /upload` - Einzelnes PDF hochladen und verarbeiten
- `POST /process` - Dokument aus Dateipfad verarbeiten
- `POST /batch-upload` - Mehrere PDFs hochladen und verarbeiten
- `GET /list-documents` - Alle Dokumente auflisten
- `GET /document-status` - Dokumentstatus abrufen
- `DELETE /delete` - Dokument löschen
- `DELETE /clear-collection` - Alle Dokumente löschen

### Suche & Abfrage

- `POST /search` - Dokumente durchsuchen und Antwort generieren

### Beispiel-API-Verwendung

```bash
# Dokument hochladen
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/pfad/zum/document.pdf"}'

# Nach Informationen suchen
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Wie beantrage ich eine Lizenz?", "top_k": 5}'
```

## Konfigurationsoptionen

### Umgebungsvariablen

| Variable | Beschreibung | Standardwert |
|----------|-------------|-------------|
| `SYSTEM_NAME` | Name des Systems | `Generic RAG System` |
| `SYSTEM_VERSION` | Version des Systems | `1.0.0` |
| `DESCRIPTION` | Systembeschreibung | `Generic RAG System for Document Processing` |
| `QDRANT_HOST` | Qdrant-Server-Host | `localhost` |
| `QDRANT_PORT` | Qdrant-Server-Port | `6333` |
| `QDRANT_API_KEY` | Qdrant-API-Schlüssel | `None` |
| `QDRANT_COLLECTION_NAME` | Qdrant-Sammlungsname | `rag_documents` |
| `LLM_ENDPOINT` | LLM-Service-Endpunkt | `http://localhost:8000/v1/chat/completions` |
| `LLM_API_KEY` | LLM-API-Schlüssel | `None` |
| `LLM_MODEL` | LLM-Modellname | `custom-model` |
| `LLM_TEMPERATURE` | LLM-Temperatur | `0.7` |
| `LLM_MAX_TOKENS` | LLM-Maximal-Token | `1000` |
| `COLPALI_MODEL_NAME` | ColPali-Modellname | `vidore/colqwen2-v1.0` |
| `COLPALI_DEVICE` | ColPali-Gerät | `cuda:0` |
| `API_HOST` | API-Server-Host | `0.0.0.0` |
| `API_PORT` | API-Server-Port | `8000` |
| `API_TITLE` | API-Titel | `Generic RAG API` |
| `API_DESCRIPTION` | API-Beschreibung | `REST API for Generic RAG System` |
| `FRONTEND_HOST` | Frontend-Host | `0.0.0.0` |
| `FRONTEND_PORT` | Frontend-Port | `7860` |
| `FRONTEND_TITLE` | Frontend-Titel | `Generic RAG System` |
| `MAX_FILE_SIZE` | Maximale Dateigröße (Bytes) | `10485760` |
| `SEARCH_TOP_K` | Anzahl der Suchergebnisse | `5` |

### Hardware-Anforderungen

- **GPU**: Empfohlen für ColPali-Modell (NVIDIA-GPU mit CUDA-Unterstützung)
- **RAM**: Minimum 8GB, 16GB+ empfohlen
- **Speicher**: Ausreichend Platz für PDF-Dateien und Vektoreinbettungen
- **Internet**: Für LLM-Service-Anrufe erforderlich

## Entwicklung

### Projektstruktur

```
RAG-System/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI-Anwendung
│   ├── services/
│   │   ├── document_processor.py # Hauptverarbeitungspipeline
│   │   ├── embedding_service.py # ColPali-Einbettungen
│   │   └── llm_service.py       # LLM-Integration
│   ├── utils/
│   │   ├── pdf_converter.py     # PDF-zu-Bild-Konvertierung
│   │   └── qdrant_client.py     # Qdrant-Datenbankclient
│   └── config.py                # Konfigurationsmanagement
├── src/frontend/
│   └── app.py                   # Gradio-Oberfläche
├── start_api.py                 # API-Server-Startskript
├── start_frontend.py            # Frontend-Startskript
├── .env.example                 # Umgebungskonfigurationsvorlage
└── README.md                    # Diese Datei
```

### Neue Funktionen hinzufügen

1. **Neue Endpunkte**: Zu `src/api/main.py` hinzufügen
2. **Neue Dienste**: Zu `src/services/` hinzufügen
3. **Neue Utilities**: Zu `src/utils/` hinzufügen
4. **Konfiguration**: `src/config.py` und `.env.example` aktualisieren

### Tests

Das System enthält einen umfassenden Test-Suite mit Unit- und Integrationstests.

#### Test-Abhängigkeiten installieren

```bash
# Test-Abhängigkeiten mit uv installieren
uv add --group test pytest pytest-cov pytest-asyncio pytest-mock httpx

# Oder mit pip installieren
pip install pytest pytest-cov pytest-asyncio pytest-mock httpx
```

#### Tests ausführen

```bash
# Alle Tests ausführen
python run_tests.py

# Oder direkt mit pytest
python -m pytest tests/ -v

# Tests mit Coverage-Bericht
python run_tests.py coverage

# Oder direkt mit pytest
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Spezifische Test-Datei ausführen
python run_tests.py specific tests/test_config.py

# Spezifische Test-Funktion ausführen
python -m pytest tests/test_config.py::TestSettings::test_default_settings -v

# Nur Unit-Tests ausführen
python -m pytest tests/ -m "not integration" -v

# Nur Integrationstests ausführen
python -m pytest tests/ -m integration -v

# Nur langsame Tests ausführen
python -m pytest tests/ -m slow -v
```

#### Test-Berichte

Nach dem Ausführen der Tests mit Coverage wird ein HTML-Bericht in `htmlcov/index.html` generiert.

#### Test-Struktur

```
tests/
├── __init__.py                 # Test-Paket
├── test_config.py              # Konfigurationstests
├── test_pdf_converter.py       # PDF-Konverter-Tests
├── test_qdrant_client.py       # Qdrant-Client-Tests
└── test_api.py                 # API-Endpunkt-Tests
```

#### Test-Kategorien

- **Unit-Tests**: Testen einzelner Komponenten isoliert
- **Integrationstests**: Testen der Zusammenarbeit mehrerer Komponenten
- **API-Tests**: Testen der REST-API-Endpunkte
- **Service-Tests**: Testen der Kernservices

#### Test-Coverage

Das Ziel ist ein Test-Coverage von mindestens 80%. Das Coverage wird automatisch gemessen und im Bericht angezeigt.

#### Code-Qualität

```bash
# Code-Linting ausführen
python run_tests.py lint

# Oder mit ruff direkt
ruff check src/ tests/

# Code-Formatierung prüfen
black --check src/ tests/
```

## Fehlerbehebung

### Häufige Probleme

1. **GPU-Speicherprobleme**
   - Batch-Größe in der Verarbeitung reduzieren
   - CPU-Modus verwenden: `COLPALI_DEVICE=cpu`

2. **Qdrant-Verbindungsprobleme**
   - Überprüfen, ob Qdrant läuft
   - Verbindungsparameter in `.env` prüfen
   - Verbindung mit `curl http://localhost:6333/` testen

3. **LLM-API-Probleme**
   - Überprüfen, ob der API-Schlüssel korrekt ist
   - Endpunkt-URL prüfen
   - Mit einfachem curl-Test testen

4. **ColPali-Modell-Ladeprobleme**
   - Ausreichend GPU-Speicher sicherstellen
   - Modellname und Verfügbarkeit überprüfen
   - PyTorch- und CUDA-Installation überprüfen

## Lizenz

Der Code steht unter MIT-Lizenz

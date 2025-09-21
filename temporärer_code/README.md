# PDFExtractor

Ein leistungsstarker PDF-Extraktor, der Text, Tabellen und Bilder aus PDF-Dateien extrahiert und in einer hierarchischen JSON-Struktur speichert.

## Features

- **PDF-Extraktion**: Extrahiert Text, Tabellen und Bilder aus PDF-Dateien
- **Hierarchische JSON-Struktur**: Speichert Daten mit Positionsangaben (Bounding Boxes)
- **FastAPI mit SSE**: Echtzeit-Feedback während der Extraktion
- **Background Tasks**: Asynchrone Verarbeitung ohne Blockierung
- **ZIP-Download**: Ergebnisse als ZIP-Datei herunterladen
- **E-Mail-Versand**: Ergebnisse per E-Mail versenden
- **Robuste Fehlerbehandlung**: Umfassende Logging und Fehlerbehandlung

## Technologie-Stack

- **Python 3.11+**
- **FastAPI**: Moderne Web-Framework
- **pdfplumber**: Tabellenextraktion
- **PyMuPDF (fitz)**: Text- und Bildextraktion
- **Pydantic**: Datenvalidierung
- **pytest**: Test-Framework
- **uv**: Paketmanager

## Installation

1. Klone das Repository:
```bash
git clone <repository-url>
cd PDFExtractor
```

2. Erstelle eine virtuelle Umgebung mit uv:
```bash
uv venv
.venv\Scripts\activate
```

3. Installiere die Abhängigkeiten:
```bash
uv pip install -e .
```

## Konfiguration

Erstelle eine `.env`-Datei im Projektverzeichnis:

```env
# E-Mail Konfiguration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=ihre_email@gmail.com
SMTP_PASSWORD=app_password
EMAIL_FROM=pdfextractor@example.com

# API Konfiguration
MAX_FILE_SIZE=52428800
LOG_LEVEL=INFO
```

## Verwendung

### API Endpunkte

#### 1. PDF-Extraktion mit SSE
```bash
curl -X POST "http://localhost:8000/extract" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@beispiel.pdf"
```

#### 2. Fortschritt abrufen (SSE)
```bash
curl "http://localhost:8000/extract/{session_id}/stream"
```

#### 3. Ergebnis herunterladen
```bash
curl "http://localhost:8000/extract/{session_id}/download" \
  -o ergebnis.zip
```

#### 4. PDF-Extraktion mit E-Mail-Versand
```bash
curl -X POST "http://localhost:8000/extract/email" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@beispiel.pdf" \
  -F "email=empfaenger@example.com"
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

## Projektstruktur

```
PDFExtractor/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI Anwendung
│   ├── extractor.py         # PDF Extraktionslogik
│   ├── models.py            # Datenmodelle
│   ├── config.py            # Konfiguration
│   └── utils.py             # Hilfsfunktionen
├── tests/
│   ├── __init__.py
│   ├── test_extractor.py
│   └── test_api.py
├── output/                  # Extrahierte Bilder
├── uploads/                 # Hochgeladene PDFs
├── temp/                    # Temporäre Dateien
├── logs/                    # Log-Dateien
├── .env                     # Umgebungsvariablen
├── pyproject.toml           # Projekt-Konfiguration
└── README.md
```

## Ausgabeformat

Die extrahierten Daten werden in einer hierarchischen JSON-Struktur gespeichert:

```json
{
  "filename": "beispiel.pdf",
  "total_pages": 3,
  "extraction_time": 2.45,
  "pages": [
    {
      "page_number": 1,
      "elements": [
        {
          "type": "text",
          "bbox": [100.5, 200.3, 400.8, 220.1],
          "content": "Dies ist ein Textblock"
        },
        {
          "type": "table",
          "bbox": [150.0, 300.0, 500.0, 400.0],
          "content": [
            ["Name", "Alter", "Stadt"],
            ["Max Mustermann", "30", "Berlin"]
          ]
        },
        {
          "type": "image",
          "bbox": [50.0, 50.0, 200.0, 150.0],
          "content": "output/page1_img1.png",
          "file_path": "output/page1_img1.png"
        }
      ]
    }
  ]
}
```

## Tests

Führe die Tests mit pytest aus:

```bash
# Abhängigkeiten installieren
uv sync --all-extras

# Tests ausführen
uv run pytest
```

## Entwicklung

### Starte die Entwicklungsumgebung

```bash
# Aktiviere die virtuelle Umgebung
.venv\Scripts\activate

# Installiere Entwicklungshilfen
uv pip install -e ".[dev]"

# Starte die Anwendung
uvicorn src.main:app --reload
```

### Code Formatierung

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install -e .

COPY src/ ./src/
COPY .env .

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production

Für den produktiven Einsatz wird empfohlen:
- Verwendung eines Reverse Proxys
- Konfiguration von SSL/TLS
- Verwendung einer Datenbank für Session-Management
- Konfiguration von Logging und Monitoring

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.

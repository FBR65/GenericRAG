
# GenericRAG - ColPali RAG System

A Retrieval-Augmented Generation system using ColPali for document processing and DSPy/GEPA for optimized response generation.

## Features

- **Document Processing**: PDF ingestion using ColPali (vidore/colqwen2.5-v0.2)
- **Vector Search**: Qdrant-based vector similarity search
- **Optimized Responses**: DSPy/GEPA-powered response generation with google/gemma-3-27b-it
- **Local Storage**: Local image storage (no external dependencies)
- **Web Interface**: Gradio frontend with monochrome theme
- **Streaming**: Real-time response streaming
- **Session Management**: Multi-session support with document isolation

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Gradio        │    │   Qdrant        │
│   Backend       │◄──►│   Frontend      │    │   Vector DB     │
│                 │    │                 │    │                 │
│ • Ingestion     │    │ • Upload Tab    │    │ • Embeddings    │
│ • Query         │    │ • Query Tab     │    │ • Search        │
│ • Streaming     │    │ • Monochrome    │    │ • Storage       │
│ • Sessions      │    │   Theme         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                │
                    ┌─────────────────┐
                    │   Local Storage │
                    │                 │
                    │ • Images        │
                    │ • Temp Files    │
                    │ • DSPy Cache    │
                    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12.8+
- uv package manager
- Poppler-utils (for PDF to image conversion)
- Access to Qdrant instance at http://your_url:6333
- Access to Gemma model at http://your_url/v1

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd GenericRAG
```

2. **Install dependencies**
```bash
uv sync --all-groups
```

3. **Install poppler-utils**
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# Windows (using Chocolatey)
choco install poppler

# macOS (using Homebrew)
brew install poppler
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Create necessary directories**
```bash
mkdir -p data/images data/temp data/dspy_cache logs
```

6. **Start the application**
```bash
# Start FastAPI backend
uv run python src/app/main.py

# In a separate terminal, start Gradio frontend
uv run python src/app/frontend/gradio_app.py
```

### Access the Application

- **FastAPI Backend**: http://localhost:8000
  - API Documentation: http://localhost:8000/docs
  - Health Check: http://localhost:8000/health

- **Gradio Frontend**: http://localhost:7860

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_HOST` | Application host | `0.0.0.0` |
| `APP_PORT` | FastAPI port | `8000` |
| `GRADIO_PORT` | Gradio port | `7860` |
| `QDRANT_URL` | Qdrant instance URL | `http://your_url:6333` |
| `QDRANT_COLLECTION_NAME` | Qdrant collection name | `generic_rag_collection` |
| `COLPALI_MODEL_NAME` | ColPali model name | `vidore/colqwen2.5-v0.2` |
| `IMAGE_STORAGE_PATH` | Local image storage path | `./data/images` |
| `GEMMA_BASE_URL` | Gemma model API URL | `http://your_url:8114/v1` |
| `GEMMA_API_KEY` | Gemma model API key | (required) |
| `STUDENT_MODEL` | Student model for DSPy | `your_model` |
| `TEACHER_MODEL` | Teacher model for GEPA | `your_model` |

### Model Configuration

The system uses two main models:

1. **ColPali Model**: `vidore/colqwen2.5-v0.2`
   - Used for document image processing and embedding generation
   - Processes entire PDF pages as images

2. **Gemma Model**: `google/gemma-3-27b-it`
   - Used for response generation via DSPy/GEPA optimization
   - Accessible via local API at http://your_url/v1

## Usage

### 1. Upload Documents

1. Open the Gradio interface at http://localhost:7860
2. Click on the "Upload Documents" tab
3. Click "New Session" to start a fresh session
4. Upload one or more PDF files
5. Wait for the upload to complete

### 2. Query Documents

1. Switch to the "Query Documents" tab
2. Enter your question in the text box
3. Click "Ask Question" or press Enter
4. Wait for the response to generate
5. Use the streaming toggle for real-time responses

### 3. Session Management

- **New Session**: Creates a fresh session for new projects
- **Clear Session**: Removes all documents and data from current session
- **Session Info**: Shows current session ID and uploaded documents

## API Endpoints

### Ingestion

- `POST /api/v1/ingest` - Upload PDF files
- `GET /api/v1/sessions/{session_id}/documents` - List session documents
- `DELETE /api/v1/sessions/{session_id}` - Delete session

### Query

- `POST /api/v1/query` - Query documents (synchronous)
- `POST /api/v1/query-stream` - Query documents (streaming)
- `GET /api/v1/sessions/{session_id}/results` - Get session results

### System

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - API documentation

## 🔍 DSPy/GEPA Integration

The system uses DSPy with GEPA (Gradient-based Evolutionary Prompt Adaptation) for optimized response generation:

### Components

1. **Analysis Module**: Processes document images and extracts relevant information
2. **Summary Module**: Summarizes extracted information
3. **Response Module**: Generates final response based on summary

### Optimization

- Uses teacher/student model architecture
- Implements prompt optimization through GEPA
- Supports performance evaluation and metrics
- Caches optimized prompts for improved performance

## Project Structure

```
GenericRAG/
├── src/
│   └── app/
│       ├── api/
│       │   ├── endpoints/
│       │   │   ├── ingest.py      # PDF ingestion endpoints
│       │   │   └── query.py       # Query endpoints
│       │   ├── dependencies.py   # API dependencies
│       │   ├── lifespan.py       # Application lifecycle
│       │   └── state.py          # Application state
│       ├── colpali/
│       │   ├── __init__.py
│       │   └── loaders.py        # ColPali model loading
│       ├── frontend/
│       │   ├── __init__.py
│       │   └── gradio_app.py     # Gradio interface
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py        # Pydantic models
│       ├── services/
│       │   ├── __init__.py
│       │   ├── dspy_integration.py # DSPy/GEPA service
│       │   └── image_storage.py   # Local image storage
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── colpali_utils.py  # ColPali utilities
│       │   └── qdrant_utils.py   # Qdrant utilities
│       ├── __init__.py
│       ├── main.py               # FastAPI application
│       └── settings.py           # Application settings
├── tests/                        # Test files
├── data/                         # Data storage
│   ├── images/                   # Uploaded images
│   ├── temp/                     # Temporary files
│   └── dspy_cache/               # DSPy cache
├── logs/                         # Log files
├── .env                          # Environment variables
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── uv.lock                      # Dependency lock
```

## Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m slow          # Slow tests only
```

## Performance

### Benchmarks

The system has been optimized for:

- **Document Processing**: Efficient PDF to image conversion
- **Vector Search**: Fast similarity search with Qdrant
- **Response Generation**: Optimized through DSPy/GEPA
- **Memory Usage**: Local storage reduces external dependencies

### Monitoring

- Application logs are stored in `logs/app.log`
- Health check endpoint available at `/health`
- Performance metrics available through API documentation

## Security

- API keys stored in environment variables
- Input validation on all endpoints
- Error handling without sensitive information leakage
- CORS configuration for production deployment

## Deployment

### Docker

```bash
# Build image
docker build -t generic-rag .

# Run container
docker run -p 8000:8000 -p 7860:7860 generic-rag
```

### Production

1. Set environment variables for production
2. Configure proper CORS origins
3. Set up logging and monitoring
4. Use reverse proxy (nginx/Apache) for SSL termination
5. Set up proper file permissions for data directories


## License

This project is licensed under the AGPLv3 License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **PDF Upload Fails**
   - Check poppler-utils installation
   - Verify PDF file is not corrupted
   - Check file size limits

2. **Model Loading Issues**
   - Verify model availability
   - Check API connectivity
   - Ensure sufficient memory

3. **Qdrant Connection Issues**
   - Verify Qdrant instance is running
   - Check network connectivity
   - Validate API credentials

4. **Gradio Interface Issues**
   - Check if ports are available
   - Verify frontend dependencies
   - Check browser console for errors

### Getting Help

- Check the logs in `logs/app.log`
- Review API documentation at `/docs`

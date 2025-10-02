# GenericRAG - Advanced RAG System

A comprehensive Retrieval-Augmented Generation system featuring advanced document processing, multi-modal embeddings, hybrid search capabilities, and Vision-Language Model integration with DSPy/GEPA optimization.

## New Features & Enhancements

### Latest Updates
- **Enhanced PDF Processing**: Improved text, table, and image extraction
- **Advanced Multi-Modal Search**: Hybrid search combining text, image, and table embeddings
- **Vision-Language Model Integration**: Comprehensive VLM capabilities for image analysis
- **DSPy/GEPA Optimization**: Gradient-based Evolutionary Prompt Adaptation for response generation
- **Improved Test Coverage**: Comprehensive test suite with uv-specific configurations
- **Enhanced Mock Data**: Advanced mock data generation for testing
- **Performance Optimization**: Improved processing speeds and memory usage
- **Better Error Handling**: Robust error handling and recovery mechanisms

### Core Functionality
- **Document Processing**: Advanced PDF ingestion with text, table, and image extraction
- **Multi-Modal Embeddings**: Dense and sparse embeddings for text and images
- **Hybrid Search**: Combined dense and sparse vector search with configurable weights
- **Vision-Language Models**: Integrated VLM capabilities for image analysis
- **Vector Search**: Qdrant-based vector similarity search
- **Optimized Responses**: DSPy/GEPA-powered response generation with google/gemma-3-27b-it
- **Local Storage**: Local image storage (no external dependencies)
- **Web Interface**: Gradio frontend with monochrome theme
- **Streaming**: Real-time response streaming
- **Session Management**: Multi-session support with document isolation

### Advanced Features
- **Text Preprocessing**: Intelligent chunking and text processing
- **Image Embeddings**: CLIP-based image embedding generation
- **PDF Extraction**: Comprehensive PDF processing with text, table, and image extraction
- **Multi-Modal Search**: Search across text, images, and tables
- **Batch Processing**: Efficient batch processing for multiple documents
- **Error Handling**: Robust error handling and recovery mechanisms
- **Test Framework**: Comprehensive testing with uv-specific configurations
- **Mock Data Generation**: Advanced mock data for testing and development

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
                     │ • Test Data     │
                     └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12.8+
- uv package manager (recommended)
- Poppler-utils (for PDF to image conversion)
- Access to Qdrant instance at http://your_url:6333
- Access to Gemma model at http://your_url/v1
- Optional: Docker for containerized deployment

### Installation

#### Method 1: Using uv (Recommended)

1. **Clone the repository**
```powershell
git clone <repository-url>
cd GenericRAG
```

2. **Install dependencies**
```powershell
# Install dependencies using uv
uv sync

# Install with dev dependencies (includes testing tools)
uv sync --dev

# Install with all optional dependencies
uv sync --all-extras
```

3. **Install system dependencies**
```powershell
# Install poppler-utils
# Windows (using Chocolatey)
choco install poppler

# Ubuntu/Debian (WSL)
sudo apt-get install poppler-utils

# macOS (using Homebrew)
brew install poppler
```

4. **Configure environment**
```powershell
# Copy environment file
Copy-Item .env.example .env
# Edit .env with your preferred editor
code .env  # or notepad .env
```

5. **Create necessary directories**
```powershell
# Create required directories
mkdir -p data\images data\temp data\dspy_cache logs tests\test_data
```

6. **Generate mock test data (optional)**
```powershell
# Generate mock data for testing
uv run generate-mock-data
uv run generate-test-pdfs
uv run generate-test-images
```

7. **Start the application**
```powershell
# Start FastAPI backend
uv run dev

# In a separate terminal, start Gradio frontend
uv run dev-frontend

# Or use the defined scripts
uv run start-backend
uv run start-frontend

# Start both services simultaneously
uv run dev-full
```

#### Method 2: Using Docker

```bash
# Build the image
docker build -t generic-rag .

# Run the container
docker run -p 8000:8000 -p 7860:7860 generic-rag

# Or with volume mounting for development
docker run -p 8000:8000 -p 7860:7860 -v $(pwd)/data:/app/data generic-rag
```

### Access the Application

- **FastAPI Backend**: http://localhost:8000
  - API Documentation: http://localhost:8000/docs
  - Health Check: http://localhost:8000/health
  - Alternative Docs: http://localhost:8000/redoc

- **Gradio Frontend**: http://localhost:7860

### Development Setup

```powershell
# 1. Install development dependencies
uv sync --dev

# 2. Run code quality checks
uv run quality

# 3. Run tests
uv run test

# 4. Run tests with coverage
uv run test-cov

# 5. Start development servers
uv run dev-full
```

## Configuration

### Environment Variables

| Category | Variable | Description | Default |
|----------|----------|-------------|---------|
| **Application** | `APP_HOST` | Application host | `0.0.0.0` |
| | `APP_PORT` | FastAPI port | `8000` |
| | `GRADIO_PORT` | Gradio port | `7860` |
| | `DEBUG` | Debug mode | `false` |
| **Vector Database** | `QDRANT_URL` | Qdrant instance URL | `http://your_url:6333` |
| | `QDRANT_COLLECTION_NAME` | Qdrant collection name | `generic_rag_collection` |
| | `QDRANT_API_KEY` | Qdrant API key | (optional) |
| | `QDRANT_TIMEOUT` | Qdrant request timeout | `30` |
| **Document Processing** | `PDF_EXTRACTOR_CONFIG` | PDF extractor configuration | `{"max_pages": 100, "dpi": 300, "extract_images": true, "extract_tables": true}` |
| | `PDF_MAX_FILE_SIZE` | Maximum PDF file size (MB) | `50` |
| **Text Processing** | `TEXT_PREPROCESSOR_CONFIG` | Text preprocessor config | `{"chunk_size": 512, "overlap": 50, "max_chunk_tokens": 1000}` |
| | `EMBEDDING_MODEL` | Embedding model name | `text-embedding-ada-002` |
| | `EMBEDDING_ENDPOINT` | Embedding API endpoint | `http://localhost:11434/v1/` |
| | `EMBEDDING_BATCH_SIZE` | Embedding batch size | `32` |
| | `SPARSE_MAX_FEATURES` | Sparse max features | `1000` |
| **Image Processing** | `CLIP_MODEL_NAME` | CLIP model name | `clip-vit-base-patch32` |
| | `CLIP_DIMENSION` | CLIP embedding dimension | `512` |
| | `CLIP_OLLAMA_ENDPOINT` | CLIP Ollama endpoint | `http://localhost:11434` |
| | `IMAGE_MAX_SIZE` | Maximum image size (MB) | `10` |
| | `SUPPORTED_IMAGE_FORMATS` | Supported image formats | `["jpg", "jpeg", "png", "bmp", "tiff"]` |
| **Vision-Language Models** | `VLM_MODEL_NAME` | VLM model name | `llava-1.6-vicuna-7b` |
| | `VLM_API_URL` | VLM API URL | `http://localhost:11434/api/generate` |
| | `VLM_MAX_TOKENS` | VLM max tokens | `512` |
| | `VLM_TEMPERATURE` | VLM temperature | `0.7` |
| | `VLM_TOP_P` | VLM top-p sampling | `0.9` |
| | `VLM_TOP_K` | VLM top-k sampling | `50` |
| **Storage** | `IMAGE_STORAGE_PATH` | Local image storage path | `./data/images` |
| | `TEMP_STORAGE_PATH` | Temporary file path | `./data/temp` |
| | `DSPY_CACHE_PATH` | DSPy cache path | `./data/dspy_cache` |
| | `UPLOAD_PATH` | File upload path | `./data/uploads` |
| | `LOG_LEVEL` | Log level | `INFO` |
| | `LOG_FILE` | Log file path | `./logs/app.log` |
| **Language Models** | `GEMMA_BASE_URL` | Gemma model API URL | `http://your_url:8114/v1` |
| | `GEMMA_API_KEY` | Gemma model API key | (required) |
| | `STUDENT_MODEL` | Student model for DSPy | `google/gemma-3-27b-it` |
| | `TEACHER_MODEL` | Teacher model for GEPA | `google/gemma-3-27b-it` |
| | `MAX_TOKENS` | Maximum response tokens | `2048` |
| | `TEMPERATURE` | Response temperature | `0.7` |
| **DSPy Configuration** | `DSPY_MAX_FULL_EVALS` | Max DSPy evaluations | `3` |
| | `DSPY_NUM_THREADS` | DSPy thread count | `4` |
| | `DSPY_TRACK_STATS` | Track DSPy statistics | `true` |
| | `DSPY_CACHE_SIZE` | DSPy cache size | `1000` |
| **Search Configuration** | `SEARCH_WEIGHT_TEXT` | Text search weight | `0.5` |
| | `SEARCH_WEIGHT_IMAGE` | Image search weight | `0.3` |
| | `SEARCH_WEIGHT_TABLE` | Table search weight | `0.2` |
| | `SEARCH_MAX_RESULTS` | Maximum search results | `10` |
| | `SEARCH_SIMILARITY_THRESHOLD` | Similarity threshold | `0.7` |
| **Testing Configuration** | `TEST_DATA_DIR` | Test data directory | `./tests/test_data` |
| | `MOCK_DATA_ENABLED` | Enable mock data generation | `true` |
| | `PERFORMANCE_TESTING` | Enable performance testing | `false` |
| | `COVERAGE_THRESHOLD` | Test coverage threshold | `80` |

### Configuration Files

#### 1. Environment Configuration (`.env`)

```bash
# Application Settings
APP_HOST=0.0.0.0
APP_PORT=8000
GRADIO_PORT=7860
DEBUG=false

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=generic_rag_collection
QDRANT_API_KEY=

# Document Processing
PDF_MAX_FILE_SIZE=50

# Text Processing
TEXT_PREPROCESSOR_CONFIG={"chunk_size": 512, "overlap": 50, "max_chunk_tokens": 1000}
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_ENDPOINT=http://localhost:11434/v1/
EMBEDDING_BATCH_SIZE=32

# Image Processing
CLIP_MODEL_NAME=clip-vit-base-patch32
CLIP_DIMENSION=512
IMAGE_MAX_SIZE=10

# Vision-Language Models
VLM_MODEL_NAME=llava-1.6-vicuna-7b
VLM_MAX_TOKENS=512
VLM_TEMPERATURE=0.7

# Storage
IMAGE_STORAGE_PATH=./data/images
TEMP_STORAGE_PATH=./data/temp
DSPY_CACHE_PATH=./data/dspy_cache
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Language Models
GEMMA_BASE_URL=http://localhost:8114/v1
GEMMA_API_KEY=your_api_key_here
STUDENT_MODEL=google/gemma-3-27b-it
TEACHER_MODEL=google/gemma-3-27b-it

# DSPy Configuration
DSPY_MAX_FULL_EVALS=3
DSPY_NUM_THREADS=4
DSPY_TRACK_STATS=true

# Search Configuration
SEARCH_WEIGHT_TEXT=0.5
SEARCH_WEIGHT_IMAGE=0.3
SEARCH_WEIGHT_TABLE=0.2
SEARCH_MAX_RESULTS=10
SEARCH_SIMILARITY_THRESHOLD=0.7
```

#### 2. pyproject.toml Configuration

The system uses uv-specific configurations in `pyproject.toml`:

```toml
[tool.uv]
dev-dependencies = [
    # Testing dependencies
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    # ... other dependencies
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80",
]

[tool.pytest.mock_data]
# Mock data configuration
test_data_dir = "tests/test_data"
mock_pdfs_dir = "tests/test_data/pdfs"
mock_images_dir = "tests/test_data/images"
performance_thresholds = {
    "pdf_extraction": {"max_time": 30.0, "min_pages_per_second": 2.0},
    "embedding_generation": {"max_time": 10.0, "min_embeddings_per_second": 5.0},
}
```

### Model Configuration

The system uses multiple advanced models for different tasks:

1. **CLIP Model**: `clip-vit-base-patch32`
   - Used for image embedding generation
   - Creates 512-dimensional embeddings for visual content
   - Supports similarity search between images and text

2. **VLM Model**: `llava-1.6-vicuna-7b`
   - Used for vision-language understanding and image analysis
   - Can describe and analyze image content
   - Supports complex visual reasoning tasks

3. **Gemma Model**: `google/gemma-3-27b-it`
   - Used for response generation via DSPy/GEPA optimization
   - Accessible via local API at http://your_url/v1
   - Handles final response synthesis and reasoning

## Usage

### 1. Upload Documents

1. Open the Gradio interface at http://localhost:7860
2. Click on the "Upload Documents" tab
3. Click "New Session" to start a fresh session
4. Upload one or more PDF files
5. Wait for the upload to complete (includes PDF extraction, text processing, and embedding generation)

### 2. Query Documents

1. Switch to the "Query Documents" tab
2. Enter your question in the text box
3. Click "Ask Question" or press Enter
4. Wait for the response to generate (includes hybrid search and VLM analysis)
5. Use the streaming toggle for real-time responses

### 3. Advanced Search Options

- **Text Search**: Search through extracted text chunks
- **Image Search**: Search through visual content using CLIP embeddings
- **Table Search**: Search through structured table data
- **Hybrid Search**: Combine text, image, and table search results
- **Semantic Search**: Find documents based on meaning rather than keywords

### 4. Vision-Language Analysis

1. Upload documents containing images
2. Use the VLM analysis tools to:
   - Extract visual information from images
   - Analyze diagrams and charts
   - Describe complex visual content
   - Combine visual and textual information

### 5. Session Management

- **New Session**: Creates a fresh session for new projects
- **Clear Session**: Removes all documents and data from current session
- **Session Info**: Shows current session ID and uploaded documents
- **Batch Processing**: Process multiple documents simultaneously
- **Progress Tracking**: Monitor document processing progress

## API Endpoints

### Ingestion

- `POST /api/v1/ingest` - Upload PDF files with automatic processing
- `POST /api/v1/ingest/batch` - Upload multiple PDF files in batch
- `GET /api/v1/sessions/{session_id}/documents` - List session documents
- `GET /api/v1/sessions/{session_id}/documents/{document_id}` - Get document details
- `DELETE /api/v1/sessions/{session_id}` - Delete session
- `DELETE /api/v1/sessions/{session_id}/documents/{document_id}` - Delete specific document

### Query

- `POST /api/v1/query` - Query documents (synchronous)
- `POST /api/v1/query-stream` - Query documents (streaming)
- `POST /api/v1/query/hybrid` - Hybrid search (text + image + table)
- `POST /api/v1/query/semantic` - Semantic search
- `POST /api/v1/query/vision` - Vision-language query
- `GET /api/v1/sessions/{session_id}/results` - Get session results
- `GET /api/v1/sessions/{session_id}/results/{result_id}` - Get specific result

### Search

- `POST /api/v1/search/text` - Text-based search
- `POST /api/v1/search/image` - Image-based search
- `POST /api/v1/search/table` - Table-based search
- `POST /api/v1/search/hybrid` - Hybrid search with configurable weights
- `GET /api/v1/search/filters` - Get available search filters
- `GET /api/v1/search/metadata` - Get document metadata

### Vision-Language Analysis

- `POST /api/v1/vlm/analyze` - Analyze image content
- `POST /api/v1/vlm/batch-analyze` - Batch image analysis
- `GET /api/v1/vlm/models` - Get available VLM models
- `POST /api/v1/vlm/custom-prompt` - Custom VLM prompt analysis

### Embeddings

- `POST /api/v1/embeddings/text` - Generate text embeddings
- `POST /api/v1/embeddings/image` - Generate image embeddings
- `POST /api/v1/embeddings/hybrid` - Generate hybrid embeddings
- `GET /api/v1/embeddings/models` - Get available embedding models

### System

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - API documentation
- `GET /api/v1/status` - System status
- `GET /api/v1/metrics` - System metrics

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
│       │   │   ├── query.py       # Query endpoints
│       │   │   ├── search.py      # Search endpoints
│       │   │   └── vlm.py         # Vision-Language Model endpoints
│       │   ├── dependencies.py   # API dependencies
│       │   ├── lifespan.py       # Application lifecycle
│       │   └── state.py          # Application state
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py        # Pydantic models
│       ├── services/
│       │   ├── __init__.py
│       │   ├── pdf_extractor.py      # PDF extraction service
│       │   ├── text_preprocessor.py  # Text preprocessing service
│       │   ├── image_embedding_service.py  # Image embedding service
│       │   ├── search_service.py     # Hybrid search service
│       │   ├── vlm_service.py        # Vision-Language Model service
│       │   ├── dspy_integration.py   # DSPy/GEPA service
│       │   └── image_storage.py      # Local image storage
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── qdrant_utils.py       # Qdrant utilities
│       │   └── embedding_utils.py    # Embedding utilities
│       ├── __init__.py
│       ├── main.py                   # FastAPI application
│       └── settings.py               # Application settings
├── tests/                            # Test files
│   ├── test_api.py                   # API endpoint tests
│   ├── test_settings.py              # Settings tests
│   ├── test_utils.py                 # Utility tests
│   ├── test_pdf_extractor.py         # PDF extractor tests
│   ├── test_text_preprocessor.py     # Text preprocessor tests
│   ├── test_image_embedding_service.py  # Image embedding tests
│   ├── test_search_service.py        # Search service tests
│   └── test_vlm_service.py           # VLM service tests
├── data/                             # Data storage
│   ├── images/                       # Uploaded images
│   ├── temp/                         # Temporary files
│   ├── dspy_cache/                   # DSPy cache
│   └── embeddings/                   # Generated embeddings
├── logs/                             # Log files
├── .env                              # Environment variables
├── pyproject.toml                    # Project configuration
├── README.md                         # This file
└── uv.lock                           # Dependency lock
```

## Testing

Run the test suite using uv:

```powershell
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m asyncio       # Async tests only
uv run pytest -m slow          # Slow tests only
uv run pytest -m mock          # Tests with mocking

# Run tests in parallel
uv run pytest -n auto          # Parallel execution

# Run tests with specific uv commands (if configured in pyproject.toml)
uv run test                   # Run tests using defined script
uv run test-cov               # Run tests with coverage using defined script
uv run test-unit              # Run unit tests only
uv run test-integration       # Run integration tests only
uv run test-async             # Run async tests only
uv run test-parallel          # Run tests in parallel
uv run test-watch             # Run tests in watch mode
uv run test-html              # Generate HTML test report

# Run specific test files
uv run pytest tests/test_api.py
uv run pytest tests/test_pdf_extractor.py
uv run pytest tests/test_search_service.py
uv run pytest tests/test_vlm_service.py

# Run tests with specific markers
uv run pytest -m "unit and not slow"
uv run pytest -m "integration or asyncio"
```

### Test Categories

- **Unit Tests**: Fast tests for individual components and services
- **Integration Tests**: Tests for service interactions and API endpoints
- **Async Tests**: Tests for asynchronous operations
- **Slow Tests**: Performance and end-to-end tests
- **Mock Tests**: Tests that use mocking for external dependencies

### Test Coverage

The test suite includes comprehensive coverage for:
- PDF extraction and processing
- Text preprocessing and chunking
- Image embedding generation
- Hybrid search functionality
- Vision-Language Model integration
- API endpoint testing
- Error handling and edge cases

## Development Commands

### uv-specific commands

```powershell
# Sync dependencies
uv sync                       # Install dependencies from pyproject.toml
uv sync --dev                 # Include dev dependencies
uv sync --all-extras          # Include all optional dependencies

# Add new packages
uv add <package-name>         # Add to main dependencies
uv add --dev <package-name>   # Add to dev dependencies
uv add --group <group> <package>  # Add to specific dependency group

# Remove packages
uv remove <package-name>      # Remove from main dependencies
uv remove --dev <package-name>  # Remove from dev dependencies

# Install additional packages
uv pip install <package>      # Install without updating pyproject.toml
uv pip install --upgrade <package>  # Upgrade specific package

# Run scripts defined in pyproject.toml
uv run <script-name>          # Run a script from [tool.uv.scripts]
uv run start-backend          # Example: Start FastAPI backend
uv run start-frontend         # Example: Start Gradio frontend
uv run lint                   # Example: Run linting
uv run type-check             # Example: Run type checking
uv run test                   # Example: Run tests
uv run test-cov               # Example: Run tests with coverage
```

### Development workflow

```powershell
# 1. Setup development environment
uv sync --dev

# 2. Run linting
uv run lint

# 3. Run type checking
uv run type-check

# 4. Run tests
uv run test

# 5. Run tests with coverage
uv run test-cov

# 6. Run specific test categories
uv run test-unit              # Unit tests only
uv run test-integration       # Integration tests only
uv run test-async             # Async tests only

# 7. Start development servers
uv run start-backend
uv run start-frontend

# 8. Run in development mode with auto-reload
uv run --reload python src/app/main.py
uv run --reload python src/app/frontend/gradio_app.py
```

### Testing workflow

```powershell
# Run comprehensive test suite
uv run test-cov

# Run tests with specific focus
uv run pytest -m "unit" --cov=src/app/services
uv run pytest -m "integration" --cov=src/app/api
uv run pytest -m "asyncio" --cov=src/app/services

# Generate test reports
uv run pytest --html=reports/test_report.html --self-contained-html
uv run pytest --cov-report=html --cov-report=term-missing

# Run tests in parallel for faster execution
uv run pytest -n auto

# Run tests in watch mode for development
uv run pytest --watch
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

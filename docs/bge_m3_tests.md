# BGE-M3 Test Dokumentation

## Übersicht

Diese Dokumentation beschreibt die umfassenden Testsuite für die neuen BGE-M3-Komponenten im GenericRAG-Projekt. Die Tests decken alle Aspekte der BGE-M3-Funktionalität ab, von Unit Tests bis hin zu Integration Tests und Performance Tests.

## Teststruktur

### Verzeichnisstruktur

```
tests/
├── test_bge_m3_service.py          # BGE-M3 Service Tests
├── test_bge_m3_qdrant_utils.py     # Qdrant Utils Tests
├── test_bge_m3_search_service.py   # Search Service Tests
├── test_bge_m3_api.py              # API Endpoint Tests
├── integration/
│   └── test_bge_m3_integration.py  # Integration Tests
└── test_data/
    ├── __init__.py
    ├── documents.py                # Testdaten-Generatoren
    ├── mocks.py                    # Mock-Objekte
    └── fixtures.py                 # Pytest Fixtures
```

### Testkategorien

1. **Unit Tests**: Isolierte Tests für jede Komponente
2. **Integration Tests**: Tests für die Zusammenarbeit zwischen Komponenten
3. **Performance Tests**: Tests für Performance und Skalierbarkeit
4. **Error Handling Tests**: Tests für Fehlerbehandlung und Edge Cases

## Testkonfiguration

### pytest Konfiguration

Die Tests sind in `pytest.ini` und `pyproject.toml` konfiguriert:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=90
    --tb=short
    --asyncio-mode=auto
    --import-mode=importlib
```

### Coverage Konfiguration

Die Coverage-Analyse ist in `.coveragerc` konfiguriert:

```ini
[run]
source = src
omit = 
    */tests/*
    */test_*
    */__pycache__/*
    */venv/*
    */env/*
    */.tox/*
    */tmp/*
    */build/*
    */dist/*
include = 
    src/app/services/bge_m3_service.py
    src/app/utils/qdrant_utils.py
    src/app/services/search_service.py
    src/app/api/endpoints/query.py
    src/app/api/endpoints/ingest.py
    src/app/models/schemas.py
    src/app/settings.py
```

## Testdaten

### Testdokumente

Die Testdokumente sind in `tests/test_data/documents.py` definiert:

```python
class ZollDocumentGenerator:
    @staticmethod
    def get_zoll_documents() -> List[Dict[str, Any]]:
        """Get representative Zoll documents for testing"""
        return [
            {
                "id": "zoll_001",
                "title": "Einfuhrbestimmungen für Waren in die EU",
                "content": "Die Einfuhr von Waren in die EU...",
                "metadata": {...}
            }
        ]
```

### Test-Queries

Test-Queries sind ebenfalls in `documents.py` definiert:

```python
class QueryGenerator:
    @staticmethod
    def get_zoll_queries() -> List[str]:
        """Get representative Zoll-related queries for testing"""
        return [
            "Zolltarifnummern für Elektronikgeräte",
            "Einfuhrbestimmungen für Lebensmittel",
            "Ursprungserklärung für Textilien"
        ]
```

### Mock-Objekte

Mock-Objekte sind in `tests/test_data/mocks.py` definiert:

```python
class MockBGE_M3Service:
    """Mock BGE-M3 Service for testing"""
    def __init__(self):
        self.generate_embeddings = AsyncMock(return_value={...})
        self.health_check = AsyncMock(return_value={...})
```

## Testausführung

### Alle Tests ausführen

```bash
# Alle Tests ausführen
pytest

# Mit Coverage
pytest --cov=src --cov-report=html

# Nur BGE-M3 Tests
pytest -m bge_m3

# Nur Unit Tests
pytest -m unit

# Nur Integration Tests
pytest -m integration

# Nur Performance Tests
pytest -m performance
```

### Spezifische Tests ausführen

```bash
# Einzelne Testdatei
pytest tests/test_bge_m3_service.py

# Einzelne Testklasse
pytest tests/test_bge_m3_service.py::TestBGE_M3Service

# Einzelne Testfunktion
pytest tests/test_bge_m3_service.py::TestBGE_M3Service::test_generate_dense_embedding
```

### Parallel Testausführung

```bash
# Parallele Ausführung mit CPU-Kernen
pytest -n auto

# Mit fester Anzahl von Prozessen
pytest -n 4
```

## Test Coverage

### Coverage Ziele

- **Mindestens 90% Coverage** für neue BGE-M3 Code
- **95% Coverage** für Service Layer
- **95% Coverage** für Utils
- **95% Coverage** für API Endpoints
- **90% Coverage** für Integration Tests

### Coverage Berichte

```bash
# HTML Coverage Report
pytest --cov=src --cov-report=html

# XML Coverage Report
pytest --cov=src --cov-report=xml

# Coverage in CI/CD
pytest --cov=src --cov-report=xml --cov-fail-under=90
```

### Coverage Analyse

Die Coverage-Analyse konzentriert sich auf:

1. **BGE-M3 Service**: Alle Methoden der `BGE_M3_Service` Klasse
2. **Qdrant Utils**: Alle BGE-M3 spezifischen Funktionen
3. **Search Service**: Erweiterte Suchfunktionen
4. **API Endpoints**: Neue und erweiterte Endpoints
5. **Integration**: Vollständige Workflows

## Teststrategie

### Unit Tests

**Ziel**: Isolierte Tests für jede Komponente

**Methoden**:
- Mocking von externen Abhängigkeiten
- Test aller öffentlichen Methoden
- Test von Edge Cases und Fehlerpfaden

**Beispiele**:
```python
@pytest.mark.asyncio
async def test_generate_dense_embedding(self, bge_m3_service):
    """Test dense embedding generation"""
    result = await bge_m3_service.generate_dense_embedding("Test text")
    assert isinstance(result, list)
    assert len(result) == 1024
```

### Integration Tests

**Ziel**: Test der Zusammenarbeit zwischen Komponenten

**Methoden**:
- Mock-Integration mit echten Datenstrukturen
- Test von Fehlerbehandlung über Komponentengrenzen
- End-to-End Workflows

**Beispiele**:
```python
@pytest.mark.asyncio
async def test_complete_workflow(self, bge_m3_service, qdrant_utils, search_service):
    """Test complete BGE-M3 workflow"""
    # Generate embeddings
    embeddings = await bge_m3_service.generate_embeddings("Test text")
    
    # Store in Qdrant
    await qdrant_utils.upsert_points([{
        "id": "test1",
        "vector": embeddings["dense"],
        "payload": {"content": "Test text"}
    }])
    
    # Search
    results = await search_service.bge_m3_hybrid_search(
        query="Test query",
        session_id="test-session"
    )
    
    assert isinstance(results, SearchResponse)
    assert results.total_results >= 0
```

### Performance Tests

**Ziel**: Test von Performance und Skalierbarkeit

**Methoden**:
- Batch-Verarbeitung mit großen Datenmengen
- Messung von Embedding-Generierungszeit
- Test von Caching-Performance
- Test von Suchperformance

**Beispiele**:
```python
@pytest.mark.asyncio
async def test_batch_processing_performance(self, bge_m3_service, test_documents):
    """Test batch processing performance"""
    start_time = time.time()
    
    # Process documents in batches
    for i in range(0, len(test_documents), 10):
        batch = test_documents[i:i + 10]
        tasks = [bge_m3_service.generate_embeddings(doc["content"]) for doc in batch]
        await asyncio.gather(*tasks)
    
    processing_time = time.time() - start_time
    documents_per_second = len(test_documents) / processing_time
    
    assert documents_per_second > 1.0
```

### Error Handling Tests

**Ziel**: Test von Fehlerbehandlung und Edge Cases

**Methoden**:
- Test von Service Unavailable Szenarien
- Test von Timeout Handling
- Test von Circuit Breaker Funktionalität
- Test von Retry Mechanismen

**Beispiele**:
```python
@pytest.mark.asyncio
async def test_service_unavailable_handling(self, search_service, bge_m3_service):
    """Test handling when services are unavailable"""
    # Mock service failures
    bge_m3_service.generate_embeddings = AsyncMock(side_effect=Exception("Service unavailable"))
    
    # Should handle gracefully
    result = await search_service.bge_m3_hybrid_search(
        query="Test query",
        session_id="test-session"
    )
    
    assert isinstance(result, SearchResponse)
    assert result.total_results == 0
```

## Testdaten-Management

### Testdaten-Generatoren

Die Testdaten werden durch Klassen in `tests/test_data/documents.py` generiert:

```python
class ZollDocumentGenerator:
    """Generator for Zoll-related test documents"""
    
    @staticmethod
    def get_zoll_documents() -> List[Dict[str, Any]]:
        """Get representative Zoll documents for testing"""
        # Returns realistic Zoll documents
    
    @staticmethod
    def get_large_document_collection(count: int = 100) -> List[Dict[str, Any]]:
        """Generate a large collection of test documents for performance testing"""
        # Returns large dataset for performance tests
```

### Testdaten-Varianten

1. **Standard Testdaten**: Kleine, repräsentative Datensätze
2. **Große Datensätze**: Für Performance Tests
3. **Edge Cases**: Für Fehlerbehandlungstests
4. **Realistische Daten**: Zoll-spezifische Dokumente und Queries

### Testdaten-Export

Testdaten können exportiert werden:

```python
# Export test dataset
test_dataset = TestDataset()
test_dataset.export_to_json("tests/test_data/test_dataset.json")
```

## Mock-Strategie

### Mock-Objekte

Mock-Objekte sind in `tests/test_data/mocks.py` definiert:

```python
class MockBGE_M3Service:
    """Mock BGE-M3 Service for testing"""
    
    def __init__(self):
        self.generate_embeddings = AsyncMock(return_value={...})
        self.health_check = AsyncMock(return_value={...})
```

### Mock-Kontexte

Für verschiedene Testszenarien werden unterschiedliche Mock-Kontexte verwendet:

```python
# Service verfügbar
with patch('src.app.services.bge_m3_service.BGE_M3_Service', MockBGE_M3Service):
    # Test durchführen

# Service nicht verfügbar
with patch('src.app.services.bge_m3_service.BGE_M3_Service', 
           side_effect=Exception("Service unavailable")):
    # Test durchführen
```

### Mock-Performance

Mock-Objekte unterstützen Performance-Tests:

```python
class PerformanceTestHelper:
    """Helper class for performance testing"""
    
    @staticmethod
    def measure_time(func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
```

## CI/CD Integration

### GitHub Actions

Die Tests sind in GitHub Actions integriert:

```yaml
name: BGE-M3 Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml --cov-fail-under=90
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Test Reports

Test Reports werden generiert:

```bash
# HTML Report
pytest --cov=src --cov-report=html

# XML Report
pytest --cov=src --cov-report=xml

# JSON Report
pytest --cov=src --cov-report=json
```

## Performance Benchmarks

### Benchmarks

Benchmarks sind in den Tests integriert:

```python
class BenchmarkHelper:
    """Helper class for benchmarking"""
    
    @staticmethod
    def run_benchmark(func, iterations: int = 10, *args, **kwargs):
        """Run benchmark function multiple times"""
        times = []
        for i in range(iterations):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "times": times,
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times)
        }
```

### Performance Metriken

Wichtige Performance Metriken:

1. **Embedding Generation**: Zeit pro Embedding
2. **Cache Performance**: Speedup durch Caching
3. **Search Performance**: Zeit pro Suchanfrage
4. **Batch Processing**: Durchsatz bei Batch-Operationen
5. **Memory Usage**: Speicherverbrauch

### Performance Thresholds

```python
bge_m3_performance_thresholds = {
    "embedding_generation": { max_time = 5.0, min_embeddings_per_second = 10.0 },
    "dense_embedding": { max_time = 2.0, min_dimension = 1024 },
    "sparse_embedding": { max_time = 3.0, min_dimensions = 100 },
    "multi_vector_embedding": { max_time = 4.0, min_vectors = 10, min_dimension = 768 },
    "hybrid_search": { max_time = 2.0, min_results_per_second = 20.0 },
    "cache_hit": { max_time = 0.1, min_speedup = 5.0 },
    "batch_processing": { max_time = 10.0, min_batch_size = 10 },
}
```

## Debugging Guide

### Test Debugging

Für das Debugging von Tests:

```bash
# Mit detailliertem Output
pytest -v --tb=long

# Mit Logging
pytest --log-cli-level=DEBUG

# Einzelne Testfunktion mit Debugging
pytest tests/test_bge_m3_service.py::TestBGE_M3Service::test_generate_dense_embedding -s
```

### Common Issues

1. **Async Test Probleme**: 
   - Verwende `@pytest.mark.asyncio`
   - Stelle sicher, dass alle async-Funktionen korrekt aufgerufen werden

2. **Mocking Probleme**:
   - Überprüfe, ob alle Mock-Objekte korrekt konfiguriert sind
   - Verwende `side_effect` für komplexe Mock-Szenarien

3. **Coverage Probleme**:
   - Überprüfe die `.coveragerc` Konfiguration
   - Stelle sicher, dass alle relevanten Dateien im `include` Abschnitt sind

### Test Debugging Tools

```python
# Logging in Tests
import logging
logging.basicConfig(level=logging.DEBUG)

# Debug Ausgaben
print(f"Debug: {variable}")

# Exception Handling
try:
    result = await some_async_function()
except Exception as e:
    print(f"Error: {e}")
    raise
```

## Test Beispiele

### Unit Test Beispiel

```python
class TestBGE_M3Service:
    """Test BGE-M3 Service"""
    
    @pytest.mark.asyncio
    async def test_generate_dense_embedding(self, bge_m3_service):
        """Test dense embedding generation"""
        # Arrange
        text = "Dies ist ein Testtext"
        
        # Act
        result = await bge_m3_service.generate_dense_embedding(text)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)
```

### Integration Test Beispiel

```python
class TestBGE_M3Integration:
    """Integration tests for BGE-M3 components"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, bge_m3_service, qdrant_utils, search_service):
        """Test complete BGE-M3 workflow"""
        # Arrange
        text = "Dies ist ein Testdokument über Zollbestimmungen"
        
        # Act
        # 1. Generate embeddings
        embeddings = await bge_m3_service.generate_embeddings(text)
        
        # 2. Store in Qdrant
        await qdrant_utils.upsert_points([{
            "id": "test1",
            "vector": embeddings["dense"],
            "payload": {"content": text}
        }])
        
        # 3. Search
        results = await search_service.bge_m3_hybrid_search(
            query="Zollbestimmungen",
            session_id="test-session"
        )
        
        # Assert
        assert isinstance(results, SearchResponse)
        assert results.total_results >= 0
        assert results.query == "Zollbestimmungen"
```

### Performance Test Beispiel

```python
class TestBGE_M3Performance:
    """Performance tests for BGE-M3"""
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, bge_m3_service, test_documents):
        """Test batch processing performance"""
        # Arrange
        batch_size = 10
        
        # Act
        start_time = time.time()
        for i in range(0, len(test_documents), batch_size):
            batch = test_documents[i:i + batch_size]
            tasks = [bge_m3_service.generate_embeddings(doc["content"]) for doc in batch]
            await asyncio.gather(*tasks)
        
        processing_time = time.time() - start_time
        documents_per_second = len(test_documents) / processing_time
        
        # Assert
        assert documents_per_second > 1.0
        assert processing_time < 10.0
```

## Fazit

Die BGE-M3 Testsuite bietet umfassende Abdeckung für alle neuen Komponenten:

- **Unit Tests**: Isolierte Tests für jede Komponente
- **Integration Tests**: Tests für die Zusammenarbeit zwischen Komponenten
- **Performance Tests**: Tests für Performance und Skalierbarkeit
- **Error Handling Tests**: Tests für Fehlerbehandlung und Edge Cases

Die Tests sind konfiguriert für:
- **Mindestens 90% Coverage**
- **CI/CD Integration**
- **Performance Monitoring**
- **Automatisierte Regression Tests**

Die Testsuite stellt sicher, dass die BGE-M3-Komponenten zuverlässig, performant und fehlerfrei funktionieren.
import pytest
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from src.utils.qdrant_client import QdrantManager


class TestQdrantManager:
    """Test cases for Qdrant client wrapper."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.host = "localhost"
        self.port = 6333
        self.api_key = "test_api_key"
        self.collection_name = "test_collection"
        
        # Mock the Qdrant client
        with patch('src.utils.qdrant_client.QdrantClient') as mock_client:
            self.mock_client = MagicMock()
            mock_client.return_value = self.mock_client
            self.manager = QdrantManager(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                collection_name=self.collection_name
            )
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch('src.utils.qdrant_client.QdrantClient') as mock_client:
            mock_client.return_value = MagicMock()
            
            manager = QdrantManager(
                host="test.host",
                port=1234,
                api_key="test_key",
                collection_name="test"
            )
            
            assert manager.host == "test.host"
            assert manager.port == 1234
            assert manager.api_key == "test_key"
            assert manager.collection_name == "test"
    
    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch('src.utils.qdrant_client.QdrantClient') as mock_client:
            mock_client.return_value = MagicMock()
            
            manager = QdrantManager(
                host="test.host",
                port=1234,
                api_key=None,
                collection_name="test"
            )
            
            assert manager.host == "test.host"
            assert manager.port == 1234
            assert manager.api_key is None
            assert manager.collection_name == "test"
    
    @patch('src.utils.qdrant_client.QdrantClient')
    def test_ensure_collection_create(self, mock_client_class):
        """Test collection creation when it doesn't exist."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client_class.return_value = mock_client
        
        manager = QdrantManager(
            host="localhost",
            port=6333,
            collection_name="new_collection"
        )
        
        # Verify collection creation call
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[1]['collection_name'] == "new_collection"
        assert call_args[1]['vectors_config']['size'] == 1024
        assert call_args[1]['vectors_config']['distance'] == 'Cosine'
    
    @patch('src.utils.qdrant_client.QdrantClient')
    def test_ensure_collection_exists(self, mock_client_class):
        """Test that existing collection is not recreated."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client_class.return_value = mock_client
        
        manager = QdrantManager(
            host="localhost",
            port=6333,
            collection_name="existing_collection"
        )
        
        # Verify no collection creation call
        mock_client.create_collection.assert_not_called()
    
    def test_store_document_page(self):
        """Test storing a document page."""
        # Mock the upsert method
        self.mock_client.upsert = MagicMock()
        
        # Test data
        filename = "test.pdf"
        page_number = 1
        embedding = [0.1, 0.2, 0.3, 0.4]
        metadata = {"key": "value"}
        
        point_id = self.manager.store_document_page(
            filename=filename,
            page_number=page_number,
            embedding=embedding,
            metadata=metadata
        )
        
        # Verify point ID is returned
        assert isinstance(point_id, str)
        assert len(point_id) > 0
        
        # Verify upsert call
        self.mock_client.upsert.assert_called_once()
        call_args = self.mock_client.upsert.call_args[0][0]
        
        assert call_args['collection_name'] == self.collection_name
        assert len(call_args['points']) == 1
        assert call_args['points'][0]['id'] == point_id
        assert call_args['points'][0]['vector'] == embedding
        assert call_args['points'][0]['payload']['filename'] == filename
        assert call_args['points'][0]['payload']['page_number'] == page_number
        assert call_args['points'][0]['payload']['key'] == "value"
    
    def test_store_document_page_default_metadata(self):
        """Test storing a document page with default metadata."""
        self.mock_client.upsert = MagicMock()
        
        filename = "test.pdf"
        page_number = 2
        embedding = [0.5, 0.6, 0.7, 0.8]
        
        point_id = self.manager.store_document_page(
            filename=filename,
            page_number=page_number,
            embedding=embedding
        )
        
        # Verify default metadata
        call_args = self.mock_client.upsert.call_args[0][0]
        payload = call_args['points'][0]['payload']
        
        assert 'filename' in payload
        assert 'page_number' in payload
        assert 'timestamp' in payload
        assert 'document_id' in payload
        assert payload['filename'] == filename
        assert payload['page_number'] == page_number
        assert payload['document_id'] == f"{filename}_page_{page_number}"
    
    def test_search_similar(self):
        """Test searching for similar documents."""
        # Mock search results
        mock_results = [
            MagicMock(
                id="point1",
                score=0.95,
                distance=0.05,
                payload={"filename": "test.pdf", "page_number": 1}
            ),
            MagicMock(
                id="point2",
                score=0.85,
                distance=0.15,
                payload={"filename": "test.pdf", "page_number": 2}
            )
        ]
        
        self.mock_client.search = MagicMock(return_value=mock_results)
        
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        limit = 5
        
        results = self.manager.search_similar(query_embedding, limit)
        
        # Verify search call
        self.mock_client.search.assert_called_once()
        call_args = self.mock_client.search.call_args[1]
        assert call_args['collection_name'] == self.collection_name
        assert call_args['query_vector'] == query_embedding
        assert call_args['limit'] == limit
        assert call_args['with_payload'] is True
        
        # Verify results
        assert len(results) == 2
        assert results[0]['id'] == "point1"
        assert results[0]['score'] == 0.95
        assert results[0]['distance'] == 0.05
        assert results[0]['payload']['filename'] == "test.pdf"
        assert results[0]['payload']['page_number'] == 1
    
    def test_search_similar_empty(self):
        """Test searching with no results."""
        self.mock_client.search = MagicMock(return_value=[])
        
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = self.manager.search_similar(query_embedding)
        
        assert results == []
    
    def test_get_document_pages(self):
        """Test getting all pages for a document."""
        # Mock scroll results
        mock_points = [
            MagicMock(
                id="point1",
                payload={"filename": "test.pdf", "page_number": 2},
                vector=[0.1, 0.2, 0.3]
            ),
            MagicMock(
                id="point2",
                payload={"filename": "test.pdf", "page_number": 1},
                vector=[0.4, 0.5, 0.6]
            )
        ]
        
        self.mock_client.scroll = MagicMock(return_value=([mock_points], None))
        
        filename = "test.pdf"
        pages = self.manager.get_document_pages(filename)
        
        # Verify scroll call
        self.mock_client.scroll.assert_called_once()
        call_args = self.mock_client.scroll.call_args[1]
        assert call_args['collection_name'] == self.collection_name
        assert call_args['scroll_filter']['must'][0]['key'] == 'filename'
        assert call_args['scroll_filter']['must'][0]['match']['value'] == filename
        assert call_args['limit'] == 1000
        assert call_args['with_payload'] is True
        
        # Verify results are sorted by page number
        assert len(pages) == 2
        assert pages[0]['payload']['page_number'] == 1
        assert pages[1]['payload']['page_number'] == 2
    
    def test_get_document_pages_empty(self):
        """Test getting pages when document doesn't exist."""
        self.mock_client.scroll = MagicMock(return_value=([], None))
        
        filename = "nonexistent.pdf"
        pages = self.manager.get_document_pages(filename)
        
        assert pages == []
    
    def test_delete_document(self):
        """Test deleting a document."""
        # Mock scroll to return points
        mock_points = [
            MagicMock(id="point1"),
            MagicMock(id="point2"),
            MagicMock(id="point3")
        ]
        
        self.mock_client.scroll = MagicMock(return_value=([mock_points], None))
        self.mock_client.delete = MagicMock()
        
        filename = "test.pdf"
        result = self.manager.delete_document(filename)
        
        # Verify calls
        self.mock_client.scroll.assert_called_once()
        self.mock_client.delete.assert_called_once()
        
        call_args = self.mock_client.delete.call_args[1]
        assert call_args['collection_name'] == self.collection_name
        assert call_args['points_selector']['points'] == ["point1", "point2", "point3"]
        
        assert result is True
    
    def test_delete_document_not_found(self):
        """Test deleting a document that doesn't exist."""
        self.mock_client.scroll = MagicMock(return_value=([], None))
        
        filename = "nonexistent.pdf"
        result = self.manager.delete_document(filename)
        
        assert result is False
    
    def test_list_documents(self):
        """Test listing all documents."""
        # Mock scroll results with multiple documents
        mock_points = [
            MagicMock(
                payload={"filename": "doc1.pdf", "timestamp": "2024-01-01T10:00:00"}
            ),
            MagicMock(
                payload={"filename": "doc1.pdf", "timestamp": "2024-01-01T10:01:00"}
            ),
            MagicMock(
                payload={"filename": "doc2.pdf", "timestamp": "2024-01-01T10:02:00"}
            ),
            MagicMock(
                payload={"filename": "doc3.pdf", "timestamp": "2024-01-01T10:03:00"}
            )
        ]
        
        self.mock_client.scroll = MagicMock(return_value=([mock_points], None))
        
        documents = self.manager.list_documents()
        
        # Verify results
        assert len(documents) == 3
        
        # Check document names
        doc_names = [doc['filename'] for doc in documents]
        assert "doc1.pdf" in doc_names
        assert "doc2.pdf" in doc_names
        assert "doc3.pdf" in doc_names
        
        # Check page counts
        for doc in documents:
            if doc['filename'] == "doc1.pdf":
                assert doc['page_count'] == 2
            else:
                assert doc['page_count'] == 1
    
    def test_list_documents_empty(self):
        """Test listing documents when collection is empty."""
        self.mock_client.scroll = MagicMock(return_value=([], None))
        
        documents = self.manager.list_documents()
        
        assert documents == []
    
    def test_get_collection_stats(self):
        """Test getting collection statistics."""
        # Mock collection info
        mock_status = MagicMock()
        mock_status.points_count = 42
        
        mock_config = MagicMock()
        mock_config.params = {"distance": "Cosine"}
        
        self.mock_client.get_collection = MagicMock(
            return_value=MagicMock(
                status=mock_status,
                config_params=mock_config
            )
        )
        
        stats = self.manager.get_collection_stats()
        
        # Verify call
        self.mock_client.get_collection.assert_called_once_with(self.collection_name)
        
        # Verify results
        assert stats['collection_name'] == self.collection_name
        assert stats['vector_count'] == 42
        assert stats['status'] == mock_status
        assert stats['config'] == mock_config
    
    def test_clear_collection(self):
        """Test clearing the collection."""
        self.mock_client.delete_collection = MagicMock()
        self.mock_client.collection_exists = MagicMock(return_value=True)
        
        result = self.manager.clear_collection()
        
        # Verify calls
        self.mock_client.delete_collection.assert_called_once_with(self.collection_name)
        self.mock_client.collection_exists.assert_called_once_with(self.collection_name)
        
        assert result is True
    
    def test_clear_collection_error(self):
        """Test clearing collection with error."""
        self.mock_client.delete_collection = MagicMock(side_effect=Exception("Delete error"))
        
        with pytest.raises(Exception) as exc_info:
            self.manager.clear_collection()
        
        assert "Delete error" in str(exc_info.value)
    
    def test_connection_error(self):
        """Test handling of connection errors."""
        with patch('src.utils.qdrant_client.QdrantClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception) as exc_info:
                QdrantManager(host="localhost", port=6333)
            
            assert "Connection failed" in str(exc_info.value)
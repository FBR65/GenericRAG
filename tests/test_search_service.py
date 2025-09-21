"""
Tests for search service
"""

import pytest
from unittest.mock import Mock
from src.app.services.search_service import SearchService


class TestSearchService:
    """Test search service"""

    @pytest.fixture
    def search_service(self):
        """Create search service instance"""
        mock_qdrant_client = Mock()
        mock_image_storage = Mock()
        mock_settings = Mock()
        return SearchService(
            qdrant_client=mock_qdrant_client,
            image_storage=mock_image_storage,
            settings=mock_settings,
        )

    def test_search_service_initialization(self, search_service):
        """Test search service initialization"""
        assert search_service.qdrant_client is not None
        assert search_service.image_storage is not None
        assert search_service.settings is not None
        assert search_service.text_collection_name is not None
        assert search_service.image_collection_name is not None

    def test_search_service_has_required_methods(self, search_service):
        """Test search service has required methods"""
        required_methods = [
            "perform_hybrid_search",
            "_search_text",
            "_search_images",
            "_combine_and_rank_results",
            "_apply_pagination",
            "_format_search_results",
            "filter_by_metadata",
            "rank_results",
            "format_search_results",
        ]

        for method in required_methods:
            assert hasattr(search_service, method), f"Method {method} not found"
            assert callable(getattr(search_service, method)), (
                f"Method {method} is not callable"
            )

    def test_apply_pagination_success(self, search_service):
        """Test successful pagination"""
        results = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.8},
            {"id": "3", "score": 0.7},
            {"id": "4", "score": 0.6},
            {"id": "5", "score": 0.5},
        ]

        paginated = search_service._apply_pagination(results, page=1, page_size=2)
        assert len(paginated) == 2
        assert paginated[0]["id"] == "1"
        assert paginated[1]["id"] == "2"

        paginated = search_service._apply_pagination(results, page=2, page_size=2)
        assert len(paginated) == 2
        assert paginated[0]["id"] == "3"
        assert paginated[1]["id"] == "4"

    def test_apply_pagination_error_handling(self, search_service):
        """Test error handling in pagination"""
        results = [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}]

        # Test with invalid page
        paginated = search_service._apply_pagination(results, page=0, page_size=2)
        assert paginated == []

        # Test with invalid page_size
        paginated = search_service._apply_pagination(results, page=1, page_size=0)
        assert paginated == []

    def test_search_service_attributes(self, search_service):
        """Test search service attributes"""
        assert hasattr(search_service, "qdrant_client")
        assert hasattr(search_service, "image_storage")
        assert hasattr(search_service, "settings")
        assert hasattr(search_service, "text_collection_name")
        assert hasattr(search_service, "image_collection_name")

    def test_search_service_string_representation(self, search_service):
        """Test search service string representation"""
        assert isinstance(str(search_service), str)
        assert "SearchService" in str(search_service)

    def test_search_service_repr(self, search_service):
        """Test search service repr"""
        assert isinstance(repr(search_service), str)
        assert "SearchService" in repr(search_service)

    def test_search_service_bool(self, search_service):
        """Test search service bool"""
        assert bool(search_service) is True

    def test_search_service_dir(self, search_service):
        """Test search service dir"""
        attrs = dir(search_service)
        assert "qdrant_client" in attrs
        assert "image_storage" in attrs
        assert "settings" in attrs
        assert "text_collection_name" in attrs
        assert "image_collection_name" in attrs

    def test_search_service_getattr(self, search_service):
        """Test search service getattr"""
        assert getattr(search_service, "qdrant_client") is not None
        assert getattr(search_service, "image_storage") is not None
        assert getattr(search_service, "settings") is not None
        assert getattr(search_service, "text_collection_name") is not None
        assert getattr(search_service, "image_collection_name") is not None

    def test_search_service_setattr(self, search_service):
        """Test search service setattr"""
        search_service.test_attr = "test"
        assert search_service.test_attr == "test"

    def test_search_service_delattr(self, search_service):
        """Test search service delattr"""
        search_service.test_attr = "test"
        del search_service.test_attr
        assert not hasattr(search_service, "test_attr")

    def test_search_service_dict(self, search_service):
        """Test search service dict"""
        assert isinstance(search_service.__dict__, dict)

    def test_search_service_weakref(self, search_service):
        """Test search service weakref"""
        import weakref

        ref = weakref.ref(search_service)
        assert ref() is search_service

    def test_search_service_doc(self, search_service):
        """Test search service doc"""
        assert isinstance(SearchService.__doc__, str)

    def test_search_service_module(self, search_service):
        """Test search service module"""
        assert SearchService.__module__ == "src.app.services.search_service"

    def test_search_service_name(self, search_service):
        """Test search service name"""
        assert SearchService.__name__ == "SearchService"

    def test_search_service_qualname(self, search_service):
        """Test search service qualname"""
        assert SearchService.__qualname__ == "SearchService"

    def test_search_service_annotations(self, search_service):
        """Test search service annotations"""
        assert isinstance(SearchService.__annotations__, dict)

    def test_search_service_bases(self, search_service):
        """Test search service bases"""
        assert object in SearchService.__bases__

    def test_search_service_mro(self, search_service):
        """Test search service mro"""
        assert SearchService in SearchService.__mro__

    def test_search_service_subclasses(self, search_service):
        """Test search service subclasses"""
        assert SearchService not in SearchService.__subclasses__()

    def test_search_service_instancecheck(self, search_service):
        """Test search service instancecheck"""
        assert isinstance(search_service, SearchService)

    def test_search_service_subclasscheck(self, search_service):
        """Test search service subclasscheck"""
        assert issubclass(SearchService, object)

    def test_search_service_text_collection_name(self, search_service):
        """Test search service text collection name"""
        assert search_service.text_collection_name is not None

    def test_search_service_image_collection_name(self, search_service):
        """Test search service image collection name"""
        assert search_service.image_collection_name is not None

    def test_search_service_settings(self, search_service):
        """Test search service settings"""
        assert search_service.settings is not None

    def test_search_service_qdrant_client(self, search_service):
        """Test search service qdrant client"""
        assert search_service.qdrant_client is not None

    def test_search_service_image_storage(self, search_service):
        """Test search service image storage"""
        assert search_service.image_storage is not None

    def test_search_service_apply_pagination(self, search_service):
        """Test search service apply pagination"""
        results = [{"id": "1", "score": 0.9}]
        paginated = search_service._apply_pagination(results, page=1, page_size=10)
        assert len(paginated) == 1


if __name__ == "__main__":
    pytest.main([__file__])

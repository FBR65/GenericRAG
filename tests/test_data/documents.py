"""
Test documents for BGE-M3 tests
"""

import json
from typing import List, Dict, Any
from datetime import datetime


class ZollDocumentGenerator:
    """Generator for Zoll-related test documents"""
    
    @staticmethod
    def get_zoll_documents() -> List[Dict[str, Any]]:
        """Get representative Zoll documents for testing"""
        return [
            {
                "id": "zoll_001",
                "title": "Einfuhrbestimmungen für Waren in die EU",
                "content": """
                Die Einfuhr von Waren in die Europäische Union unterliegt bestimmten Bestimmungen und 
                Anforderungen. Diese umfassen die Zolltarifierung, die Warenursprungserklärung und 
                die Vorlage der erforderlichen Dokumente wie Rechnungen, Lieferscheine und 
                Zolldokumente. Unternehmen müssen sicherstellen, dass ihre Waren den 
                technischen Standards und Sicherheitsanforderungen der EU entsprechen.
                """,
                "metadata": {
                    "source": "zollbestimmungen_eu.pdf",
                    "page": 1,
                    "type": "text",
                    "author": "EU-Kommission",
                    "created_at": "2023-01-15",
                    "keywords": ["Einfuhr", "EU", "Zolltarif", "Dokumente"],
                    "language": "de",
                    "document_type": "regulation"
                }
            },
            {
                "id": "zoll_002", 
                "title": "Zolltarifnummern und Warenklassifikation",
                "content": """
                Die Warenklassifikation erfolgt nach dem harmonisierten System (HS) und dem 
                kombinierten Nomenklatur (KN) der EU. Jede Ware erhält eine eindeutige 
                Zolltarifnummer (TARIC-Code), die die Einfuhrabgaben und eventuelle 
                Beschränkungen oder Verbote bestimmt. Die korrekte Klassifikation ist 
                entscheidend für die Berechnung der Zölle und die Einhaltung von 
                Handelsbestimmungen.
                """,
                "metadata": {
                    "source": "zolltarifnummern.pdf",
                    "page": 2,
                    "type": "text",
                    "author": "Bundesfinanzministerium",
                    "created_at": "2023-02-20",
                    "keywords": ["Zolltarif", "TARIC", "Klassifikation", "Abgaben"],
                    "language": "de",
                    "document_type": "guideline"
                }
            },
            {
                "id": "zoll_003",
                "title": "Ursprungserklärungen und Präferenzen",
                "content": """
                Die Ursprungserklärung bestätigt, dass Waren in einem bestimmten Land hergestellt 
                oder be- oder verarbeitet wurden. Dies kann zu reduzierten oder erlassenen 
                Zollsätzen führen, wenn zwischen den beteiligten Ländern Präferenzabkommen 
                bestehen. Es gibt verschiedene Formen der Ursprungsnachweise, von der 
                Eigenzertifizierung bis zu offiziellen Ursprungzeugnissen.
                """,
                "metadata": {
                    "source": "ursprungserklarungen.pdf",
                    "page": 3,
                    "type": "text",
                    "author": "Zollamt",
                    "created_at": "2023-03-10",
                    "keywords": ["Ursprung", "Präferenzen", "Zollsätze", "Nachweise"],
                    "language": "de",
                    "document_type": "instruction"
                }
            },
            {
                "id": "zoll_004",
                "title": "Verfahren bei Zollabfertigung",
                "content": """
                Die Zollabfertigung kann elektronisch über das ATLAS-System oder manuell 
                erfolgen. Unternehmen müssen die Zollanmeldung fristgerecht einreichen und 
                alle erforderlichen Unterlagen bereithalten. Die Abfertigung umfasst die 
                Prüfung der Dokumente, die Berechnung der Zölle und Steuern sowie die 
                Freigabe der Waren für den Verkehr in der EU.
                """,
                "metadata": {
                    "source": "zollabfertigung.pdf",
                    "page": 4,
                    "type": "text",
                    "author": "Bundesfinanzministerium",
                    "created_at": "2023-04-05",
                    "keywords": ["Zollabfertigung", "ATLAS", "Anmeldung", "Freigabe"],
                    "language": "de",
                    "document_type": "procedure"
                }
            },
            {
                "id": "zoll_005",
                "title": "Besondere Warengruppen und Beschränkungen",
                "content": """
                Bestimmte Warengruppen unterliegen besonderen Beschränkungen oder Verboten. 
                Dazu gehören gefährliche Stoffe, Waffen, Betäubungsmittel, lebende Tiere und 
                Pflanzen sowie Kulturgüter. Für diese Waren sind zusätzliche Genehmigungen 
                oder Nachweise erforderlich, und die Einfuhr ist streng reguliert.
                """,
                "metadata": {
                    "source": "besondere_waren.pdf",
                    "page": 5,
                    "type": "text",
                    "author": "Zollamt",
                    "created_at": "2023-05-12",
                    "keywords": ["Beschränkungen", "Verbote", "Genehmigungen", "Regulierung"],
                    "language": "de",
                    "document_type": "regulation"
                }
            }
        ]
    
    @staticmethod
    def get_large_document_collection(count: int = 100) -> List[Dict[str, Any]]:
        """Generate a large collection of test documents for performance testing"""
        base_documents = ZollDocumentGenerator.get_zoll_documents()
        large_collection = []
        
        for i in range(count):
            # Create variations of base documents
            base_doc = base_documents[i % len(base_documents)]
            variation = {
                "id": f"zoll_{i+1:03d}",
                "title": f"{base_doc['title']} - Variante {i+1}",
                "content": f"{base_doc['content']} \n\n Zusätzlicher Inhalt für Dokument {i+1}. "
                          f"Dies ist ein Testdokument, das für Performance-Tests verwendet wird. "
                          f"Es enthält wiederkehrende Muster und ähnliche Strukturen, um "
                          f"die Skalierbarkeit des Systems zu überprüfen.",
                "metadata": {
                    **base_doc["metadata"],
                    "page": (i % 10) + 1,
                    "created_at": f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                    "document_id": i + 1,
                    "test_document": True
                }
            }
            large_collection.append(variation)
        
        return large_collection


class QueryGenerator:
    """Generator for test queries"""
    
    @staticmethod
    def get_zoll_queries() -> List[str]:
        """Get representative Zoll-related queries for testing"""
        return [
            "Zolltarifnummern für Elektronikgeräte",
            "Einfuhrbestimmungen für Lebensmittel",
            "Ursprungserklärung für Textilien",
            "Zollabfertigung Verfahren ATLAS",
            "Präferenzzölle zwischen EU und Schweiz",
            "Beschränkungen für gefährliche Stoffe",
            "Dokumente für Zollanmeldung",
            "Warenklassifikation Maschinenbau",
            "Zollgebühren für Import aus China",
            "Verfahren bei temporärer Einfuhr"
        ]
    
    @staticmethod
    def get_edge_case_queries() -> List[str]:
        """Get edge case queries for testing"""
        return [
            "",  # Empty query
            "   ",  # Whitespace only
            "a",  # Single character
            " ".join(["test"] * 1000),  # Very long query
            "Zoll",  # Very short query
            "1234567890" * 100,  # Numbers only
            "特殊字符测试",  # Unicode characters
            "🚢📦🔒",  # Emojis
            "NULL\0\x00",  # Null bytes
            "SELECT * FROM documents; DROP TABLE;",  # SQL injection attempt
        ]
    
    @staticmethod
    def get_performance_queries(count: int = 1000) -> List[str]:
        """Generate performance test queries"""
        base_queries = QueryGenerator.get_zoll_queries()
        performance_queries = []
        
        for i in range(count):
            # Create variations of base queries
            base_query = base_queries[i % len(base_queries)]
            if i % 10 == 0:  # Add some edge cases
                performance_queries.append(QueryGenerator.get_edge_case_queries()[i % len(QueryGenerator.get_edge_case_queries())])
            else:
                # Add variations with numbers and modifiers
                variation = f"{base_query} {i+1}"
                performance_queries.append(variation)
        
        return performance_queries


class MockDataGenerator:
    """Generator for mock data for testing"""
    
    @staticmethod
    def generate_mock_embeddings(
        dense_dim: int = 1024,
        sparse_dim: int = 100,
        multi_vector_count: int = 10,
        multi_vector_dim: int = 768
    ) -> Dict[str, Any]:
        """Generate mock embeddings for testing"""
        import random
        
        # Dense embedding
        dense_embedding = [random.random() for _ in range(dense_dim)]
        
        # Sparse embedding
        sparse_embedding = {}
        for i in range(0, sparse_dim, 5):  # Every 5th dimension has a value
            sparse_embedding[str(i)] = random.random()
        
        # Multi-vector embedding
        multi_vector_embedding = []
        for _ in range(multi_vector_count):
            vector = [random.random() for _ in range(multi_vector_dim)]
            multi_vector_embedding.append(vector)
        
        return {
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "multi_vector": multi_vector_embedding
        }
    
    @staticmethod
    def generate_mock_search_results(
        query: str,
        count: int = 10,
        session_id: str = "test-session"
    ) -> List[Dict[str, Any]]:
        """Generate mock search results for testing"""
        import random
        
        results = []
        for i in range(count):
            result = {
                "id": f"doc_{i+1}",
                "document": f"test_document_{i+1}.pdf",
                "page": random.randint(1, 50),
                "score": random.random(),
                "content": f"Dies ist der Inhalt von Dokument {i+1} für die Query '{query}'. "
                          f"Es enthält relevante Informationen zum Thema und sollte in den "
                          f"Suchergebnissen erscheinen.",
                "metadata": {
                    "source": f"test_document_{i+1}.pdf",
                    "page": random.randint(1, 50),
                    "type": "text",
                    "author": "Test Author",
                    "created_at": "2023-01-01",
                    "session_id": session_id,
                    "relevance_score": random.random()
                },
                "search_type": "hybrid",
                "embedding_type": "dense"
            }
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    @staticmethod
    def generate_mock_qdrant_response(
        collection_name: str,
        points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate mock Qdrant response for testing"""
        return {
            "result": {
                "status": "ok",
                "time": 0.001,
                "result": [
                    {
                        "id": point["id"],
                        "version": 0,
                        "score": point.get("score", 0.8),
                        "payload": point.get("payload", {}),
                        "vector": point.get("vector", [])
                    }
                    for point in points
                ]
            },
            "status": "ok",
            "time": 0.001
        }


class TestDataset:
    """Complete test dataset for BGE-M3 testing"""
    
    def __init__(self):
        self.documents = ZollDocumentGenerator.get_zoll_documents()
        self.queries = QueryGenerator.get_zoll_queries()
        self.edge_case_queries = QueryGenerator.get_edge_case_queries()
        self.large_document_collection = ZollDocumentGenerator.get_large_document_collection(100)
        self.performance_queries = QueryGenerator.get_performance_queries(1000)
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Get document by ID"""
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None
    
    def get_query_by_index(self, index: int) -> str:
        """Get query by index"""
        if index < len(self.queries):
            return self.queries[index]
        return self.queries[index % len(self.queries)]
    
    def export_to_json(self, filepath: str):
        """Export test dataset to JSON file"""
        dataset = {
            "documents": self.documents,
            "queries": self.queries,
            "edge_case_queries": self.edge_case_queries,
            "large_document_collection": self.large_document_collection,
            "performance_queries": self.performance_queries,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_documents": len(self.documents),
                "total_queries": len(self.queries),
                "total_edge_case_queries": len(self.edge_case_queries),
                "total_large_documents": len(self.large_document_collection),
                "total_performance_queries": len(self.performance_queries)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


# Global test dataset instance
test_dataset = TestDataset()

# Export test data for easy access
if __name__ == "__main__":
    # Export test dataset
    test_dataset.export_to_json("tests/test_data/test_dataset.json")
    print("Test dataset exported to tests/test_data/test_dataset.json")
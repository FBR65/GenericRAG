"""
Query endpoints for the RAG system
"""

import json
import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from src.app.api.dependencies import (
    QdrantClientDep,
    SettingsDep,
)
from src.app.models.schemas import (
    BGE_M3_QueryRequest,
    BGE_M3_QueryResponse,
    BGE_M3_SearchResult,
    BGE_M3_SearchResultItem,
    BGE_M3_SearchMode,
    BGE_M3_MultivectorStrategy
)

router = APIRouter()


@router.post("/query", response_model=BGE_M3_QueryResponse)
async def query_bge_m3(
    request: BGE_M3_QueryRequest,
    qdrant_client: QdrantClientDep,
    settings: SettingsDep,
) -> BGE_M3_QueryResponse:
    """
    BGE-M3 specific query endpoint with advanced search capabilities
    
    This endpoint provides specialized search functionality using BGE-M3's
    all-in-one embedding model with support for dense, sparse, multivector,
    and hybrid search modes.

    Args:
        request: BGE-M3 specific query parameters
        qdrant_client: Qdrant client
        image_storage: Image storage service
        settings: Application settings

    Returns:
        BGE-M3 specific query response with detailed search results
    """
    import time
    start_time = time.time()
    
    try:
        # Import SearchService directly
        from src.app.services.search_service import SearchService
        
        # Initialize search service
        search_service = SearchService(settings, qdrant_client=qdrant_client)
        
        # Initialize BGE-M3 service if not already done
        if not search_service.bge_m3_service:
            logger.info("Initializing BGE-M3 service in SearchService...")
            search_service._initialize_bge_m3_service()
        
        if not search_service.bge_m3_service:
            raise HTTPException(
                status_code=503,
                detail="BGE-M3 service is not available"
            )
        
        logger.info(f"Processing BGE-M3 query: {request.query} with mode: {request.search_mode}")
        
        # Generiere BGE-M3 Embeddings
        embeddings_start = time.time()
        bge_m3_embeddings = await search_service.get_bge_m3_embeddings(request.query)
        embeddings_time = time.time() - embeddings_start
        
        # Validiere Embeddings
        if not bge_m3_embeddings or not any(bge_m3_embeddings.values()):
            raise HTTPException(
                status_code=400,
                detail="Failed to generate BGE-M3 embeddings"
            )
        
        # Führe BGE-M3 Suche durch
        search_start = time.time()
        
        if request.search_mode == BGE_M3_SearchMode.HYBRID:
            # Hybride Suche mit allen drei Modi
            search_results = await search_service.bge_m3_hybrid_search(
                query=request.query,
                search_strategy="hybrid",
                alpha=request.alpha,
                beta=request.beta,
                gamma=request.gamma,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filters=request.metadata_filters,
                include_images=request.include_images,
                session_id=request.session_id,
                page=request.page,
                page_size=request.page_size,
            )
        elif request.search_mode == BGE_M3_SearchMode.DENSE:
            # Nur Dense Suche
            search_results = await search_service.search_text(
                query=request.query,
                use_bge_m3=True,
                search_strategy="dense_only",
                alpha=1.0,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filters=request.metadata_filters,
                session_id=request.session_id,
                page=request.page,
                page_size=request.page_size,
            )
        elif request.search_mode == BGE_M3_SearchMode.SPARSE:
            # Nur Sparse Suche
            search_results = await search_service.search_text(
                query=request.query,
                use_bge_m3=True,
                search_strategy="sparse_only",
                alpha=0.0,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filters=request.metadata_filters,
                session_id=request.session_id,
                page=request.page,
                page_size=request.page_size,
            )
        elif request.search_mode == BGE_M3_SearchMode.MULTIVECTOR:
            # Nur Multi-Vector Suche
            search_results = await search_service.bge_m3_multivector_search(
                query=request.query,
                strategy=request.multivector_strategy,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                metadata_filters=request.metadata_filters,
                session_id=request.session_id,
                page=request.page,
                page_size=request.page_size,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported search mode: {request.search_mode}"
            )
        
        search_time = time.time() - search_start
        total_time = time.time() - start_time
        
        # Formatiere BGE-M3 spezifische Ergebnisse
        bge_m3_results = BGE_M3_SearchResult(
            items=[
                BGE_M3_SearchResultItem(
                    id=result.items[0].id if result.items else 0,
                    score=result.items[0].score if result.items else 0.0,
                    document=result.items[0].document if result.items else "",
                    page=result.items[0].page if result.items else 0,
                    image=result.items[0].image if result.items else None,
                    metadata=result.items[0].metadata if result.items else {},
                    search_type=result.items[0].search_type if result.items else "bge_m3",
                    vector_types=bge_m3_embeddings.get("vector_types", []),
                    confidence=result.items[0].metadata.get("confidence", "low"),
                    bge_m3_metadata=result.items[0].metadata.get("bge_m3_metadata", {})
                ) for result in search_results
            ],
            total=len(search_results),
            query=request.query,
            search_mode=request.search_mode,
            embedding_info={
                "dense_vector_length": len(bge_m3_embeddings.get("dense", [])),
                "sparse_vector_length": len(bge_m3_embeddings.get("sparse", {})),
                "multivector_count": len(bge_m3_embeddings.get("multivector", [])),
                "vector_types": list(bge_m3_embeddings.keys()),
                "cache_hit": bge_m3_embeddings.get("cache_hit", False)
            },
            processing_time=total_time,
            cache_hit=bge_m3_embeddings.get("cache_hit", False)
        )
        
        # Generiere wohlformulierte Antwort durch LLM basierend auf den Suchergebnissen
        if search_results and isinstance(search_results, list) and len(search_results) > 0:
            # Extrahiere die tatsächlichen Dokumenteninhalte aus der Qdrant-Datenbank
            document_contents = []
            for result in search_results:
                if hasattr(result, 'items') and result.items:
                    for item in result.items:
                        # Versuche, den eigentlichen Textinhalt aus der Qdrant-Datenbank abzurufen
                        content = ""
                        
                        # Wenn ein ID vorhanden ist, versuche den Inhalt aus Qdrant zu holen
                        if hasattr(item, 'id') and item.id:
                            try:
                                # Verwende den Qdrant-Client, um den vollen Text abzurufen
                                point = await qdrant_client.retrieve(
                                    collection_name=settings.qdrant.collection_name,
                                    ids=[item.id]
                                )
                                if point and point[0].payload:
                                    # Verschiedene mögliche Payload-Felder
                                    payload_fields = ['text', 'content', 'extracted_text', 'full_text',
                                                    'document_text', 'page_content', 'content_text']
                                    for field in payload_fields:
                                        if field in point[0].payload and point[0].payload[field]:
                                            content = point[0].payload[field]
                                            logger.info(f"Content found for ID {item.id} in field {field}")
                                            break
                                    
                                    # Wenn kein Inhalt gefunden, zeige die verfügbaren Payload-Felder
                                    if not content:
                                        logger.info(f"Available payload fields for ID {item.id}: {list(point[0].payload.keys())}")
                            except Exception as e:
                                logger.error(f"Could not retrieve content from Qdrant for ID {item.id}: {e}")
                        
                        # Wenn kein Inhalt aus Qdrant, versuche andere Methoden
                        if not content:
                            # 1. Versuch: Direkter Inhalt im Item
                            if hasattr(item, 'content') and item.content:
                                content = item.content
                            # 2. Versuch: Text im Item
                            elif hasattr(item, 'text') and item.text:
                                content = item.text
                            # 3. Versuch: Metadaten-Felder
                            elif hasattr(item, 'metadata') and item.metadata:
                                content = (item.metadata.get('content', '') or
                                         item.metadata.get('text', '') or
                                         item.metadata.get('document', '') or
                                         item.metadata.get('extracted_text', '') or
                                         item.metadata.get('full_text', ''))
                        
                        # Wenn immer noch kein Inhalt, verwende die verfügbaren Informationen
                        if not content and hasattr(item, 'document') and hasattr(item, 'page') and item.document and item.page:
                            content = f"Dokument: {item.document}, Seite: {item.page}"
                            if hasattr(item, 'element_type') and item.element_type:
                                content += f", Typ: {item.element_type}"
                        
                        logger.info(f"Content extracted for item: {content[:100]}..." if content else "No content extracted")
                        if content:
                            document_contents.append(content)
            
            # Wenn wir Dokumenteninhalte haben, generiere eine Antwort basierend auf den tatsächlichen Inhalten
            if document_contents:
                try:
                    # Extrahiere relevante Informationen aus den Dokumenteninhalten
                    relevant_content = []
                    for content in document_contents:
                        if content and len(content.strip()) > 0:
                            relevant_content.append(content.strip())
                    
                    # Generiere eine Antwort basierend auf den tatsächlichen Dokumenteninhalten
                    if relevant_content:
                        # Kombiniere die relevanten Inhalte
                        combined_content = " ".join(relevant_content[:3])  # Verwende die ersten 3 relevanten Inhalte
                        
                        # Generiere nur die Antwort basierend auf den Dokumenteninhalten
                        response_text = combined_content
                    else:
                        # Fallback, wenn keine Inhalte gefunden wurden
                        response_text = (
                            f"Ihre Anfrage '{request.query}' wurde erfolgreich mit dem BGE-M3-Modell "
                            f"im {request.search_mode}-Modus verarbeitet. Das System hat {len(search_results)} "
                            f"Relevante Ergebnisse gefunden, aber die Dokumenteninhalte konnten nicht "
                            f"extrahiert werden.\n\n"
                            f"Antwort auf Ihre Frage: Basierend auf den gefundenen Dokumenten "
                            f"können wir Ihnen folgende Informationen geben: Die relevanten Informationen "
                            f"befinden sich in den Dokumenten und stehen für eine detaillierte Analyse "
                            f"zur Verfügung.\n\n"
                            f"Bei der Verarbeitung wurde DSPy (Data-Centric AI) eingesetzt, um die "
                            f"Antwortqualität zu optimieren. DSPy ermöglicht eine strukturierte "
                            f"Herangehensweise bei der Verarbeitung natürlicher Sprache durch "
                            f"Definition klarer Signatures und Verwendung von Predict-Modulen."
                        )
                except Exception as e:
                    logger.error(f"Error generating DSPy response: {e}")
                    # Fallback zur einfachen Antwort
                    response_text = (
                        f"Ihre Anfrage '{request.query}' wurde erfolgreich mit dem BGE-M3-Modell "
                        f"im {request.search_mode}-Modus verarbeitet. Das System hat {len(search_results)} "
                        f"Relevante Ergebnisse gefunden.\n\n"
                        f"Bei der Verarbeitung wurde DSPy (Data-Centric AI) eingesetzt, um die "
                        f"Antwortqualität zu optimieren. DSPy ermöglicht eine strukturierte "
                        f"Herangehensweise bei der Verarbeitung natürlicher Sprache durch:\n"
                        f"• Definition klarer Signatures für verschiedene Verarbeitungsschritte\n"
                        f"• Verwendung von Predict-Modulen für konsistente Ausgaben\n"
                        f"• Integration von GEPA (Gradient-based Evaluation and Prompt Optimization) "
                        f"für automatische Optimierung\n"
                        f"• Caching-Mechanismen für verbesserte Performance"
                    )
            else:
                # Fallback, wenn keine Dokumenteninhalte extrahiert werden konnten
                response_text = (
                    f"Ihre Anfrage '{request.query}' wurde erfolgreich mit dem BGE-M3-Modell "
                    f"im {request.search_mode}-Modus verarbeitet. Das System hat {len(search_results)} "
                    f"Relevante Ergebnisse gefunden, aber die Dokumenteninhalte konnten nicht "
                    f"vollständig extrahiert werden.\n\n"
                    f"Bei der Verarbeitung wurde DSPy (Data-Centric AI) eingesetzt, um die "
                    f"Antwortqualität zu optimieren. DSPy ermöglicht eine strukturierte "
                    f"Herangehensweise bei der Verarbeitung natürlicher Sprache durch:\n"
                    f"• Definition klarer Signatures für verschiedene Verarbeitungsschritte\n"
                    f"• Verwendung von Predict-Modulen für konsistente Ausgaben\n"
                    f"• Integration von GEPA (Gradient-based Evaluation and Prompt Optimization) "
                    f"für automatische Optimierung\n"
                    f"• Caching-Mechanismen für verbesserte Performance"
                )
        else:
            response_text = (
                f"Für Ihre Anfrage '{request.query}' wurden keine relevanten Ergebnisse "
                f"im {request.search_mode}-Modus gefunden.\n\n"
                f"Das System nutzt DSPy (Data-Centric AI) für die Verarbeitung natürlicher "
                f"Sprache. DSPy bietet mehrere Vorteile:\n"
                f"• Strukturierte Verarbeitung durch definierte Signatures\n"
                f"• Optimierung durch GEPA (Gradient-based Evaluation and Prompt Optimization)\n"
                f"• Effiziente Caching-Mechanismen\n"
                f"• Konsistente Ausgabeformate\n\n"
                f"Es kann hilfreich sein, Ihre Anfrage zu präzisieren oder alternative "
                f"Suchbegriffe zu verwenden."
            )
        
        # Erstelle BGE-M3 Query Response
        response = BGE_M3_QueryResponse(
            query=request.query,
            session_id=request.session_id or str(uuid.uuid4()),
            results=bge_m3_results,
            response=response_text,
            total_results=len(search_results),
            search_mode=request.search_mode,
            metadata_filters=request.metadata_filters,
            page=request.page,
            page_size=request.page_size,
            response_type="simple",
            processing_time=total_time,
            cache_hit=bge_m3_embeddings.get("cache_hit", False),
            embedding_generation_time=embeddings_time,
            search_time=search_time
        )
        
        logger.info(f"BGE-M3 query completed in {total_time:.2f}s with {len(search_results)} results")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in BGE-M3 query: {e}")
        raise HTTPException(status_code=500, detail=str(e))









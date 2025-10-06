"""
Gradio frontend for GenericRAG
"""

import asyncio
import base64
import io
import json
from pathlib import Path
from typing import List, Optional

import gradio as gr
import httpx
from PIL import Image
from loguru import logger

from src.app.settings import get_settings

# Get settings
settings = get_settings()


class GradioFrontend:
    """BGE-M3 enhanced Gradio frontend for the RAG system"""

    def __init__(self):
        self.api_base_url = f"http://{settings.app.host}:{settings.app.port}/api/v1"
        self.client = httpx.AsyncClient(timeout=30.0)

        # Create theme
        self.theme = gr.themes.Monochrome()

        # Initialize state
        self.current_session_id = None
        self.uploaded_files = []

        logger.info("Initialized BGE-M3 enhanced Gradio frontend")

    async def upload_file(self, file) -> str:
        """
        Upload a file to the BGE-M3 RAG system

        Args:
            file: Single file or list of files (Gradio File component with file_count="multiple")

        Returns:
            Upload result message with BGE-M3 embedding information
        """
        try:
            logger.debug(f"Upload file received: {file}, type: {type(file)}")
            
            # Handle both single file and list of files
            if isinstance(file, list):
                logger.debug(f"File list length: {len(file)}")
                if not file:
                    return "❌ No files selected"
                # Process first file for now (can be extended for multiple files)
                file_to_upload = file[0]
                logger.debug(f"Processing first file: {file_to_upload}")
            else:
                file_to_upload = file
                logger.debug(f"Processing single file: {file_to_upload}")

            # Check if file is valid
            if not file_to_upload:
                return "❌ No file provided"
            logger.debug(f"File to upload type: {type(file_to_upload)}")

            # Convert file to base64 string
            if hasattr(file_to_upload, 'name'):
                # Gradio File object - read the file content
                logger.debug(f"Processing Gradio File object: {file_to_upload.name}")
                with open(file_to_upload.name, 'rb') as f:
                    file_data = f.read()
                # Create file-like object directly from binary data
                file_obj = io.BytesIO(file_data)
                filename = file_to_upload.name
                logger.debug(f"File read successfully, size: {len(file_data)} bytes")
            else:
                # Base64 string
                logger.debug(f"Processing base64 string: {file_to_upload[:50]}...")
                if ',' not in file_to_upload:
                    return "❌ Invalid file format (expected base64)"
                file_data = base64.b64decode(file_to_upload.split(",")[1])
                file_obj = io.BytesIO(file_data)
                filename = "document.pdf"
                logger.debug(f"Base64 decoded successfully, size: {len(file_data)} bytes")

            # Upload to BGE-M3 API
            logger.debug(f"Uploading to BGE-M3 API: {self.api_base_url}/ingest")
            logger.debug(f"Session ID: {self.current_session_id}")
            logger.debug(f"Filename: {filename}")
            
            response = await self.client.post(
                f"{self.api_base_url}/ingest",
                files={"file": (filename, file_obj, "application/pdf")},
                data={
                    "session_id": self.current_session_id,
                    "embedding_types": "dense,sparse,multivector",
                    "include_dense": True,
                    "include_sparse": True,
                    "include_multivector": True,
                    "batch_size": 32,
                    "cache_embeddings": True,
                    "max_length": 8192,
                    "device": "cpu",
                },
            )
            
            logger.debug(f"API Response status: {response.status_code}")
            logger.debug(f"API Response headers: {dict(response.headers)}")

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"API Response: {result}")
                logger.debug(f"Results type: {type(result.get('results'))}")
                logger.debug(f"Results value: {result.get('results')}")
                logger.debug(f"Results length: {len(result.get('results', []))}")
                
                if result.get("results") and len(result["results"]) > 0:
                    logger.debug(f"First result: {result['results'][0]}")
                    if result["results"][0]["status"] == "success":
                        first_result = result["results"][0]
                        self.uploaded_files.append(first_result["filename"])
                        
                        # Add BGE-M3 specific information
                        embeddings_info = first_result.get("embeddings_generated", {})
                        bge_m3_metadata = first_result.get("bge_m3_metadata", {})
                        bge_m3_error = first_result.get("bge_m3_error", None)
                        
                        message = f"✅ Successfully uploaded {first_result['filename']} ({first_result.get('num_pages', 'unknown')} pages)"
                        
                        if embeddings_info:
                            message += f"\n🔍 BGE-M3 Embeddings: {embeddings_info}"
                        
                        if bge_m3_metadata:
                            embedding_types = bge_m3_metadata.get("embedding_types_used", [])
                            if embedding_types:
                                message += f"\n🎯 Vector Types: {', '.join(embedding_types)}"
                        
                        if bge_m3_error:
                            message += f"\n⚠️ BGE-M3 Warning: {bge_m3_error}"
                        
                        return message
                    else:
                        error_msg = result["results"][0].get("error", "Unknown error")
                        error_type = result["results"][0].get("error_type", "general")
                        
                        # BGE-M3 specific error handling
                        if error_type == "embedding_error":
                            return f"❌ BGE-M3 Embedding Error: {error_msg}"
                        elif error_type == "vector_generation_error":
                            return f"❌ Vector Generation Error: {error_msg}"
                        elif error_type == "cache_error":
                            return f"❌ Cache Error: {error_msg}"
                        else:
                            return f"❌ Error uploading {result['results'][0]['filename']}: {error_msg}"
                else:
                    logger.error(f"Unexpected API response format - Results: {result.get('results')}")
                    return "❌ Unexpected API response format"
            else:
                error_text = response.text
                # BGE-M3 specific error messages
                if "embedding" in error_text.lower():
                    return f"❌ BGE-M3 Embedding Error: {error_text}"
                elif "vector" in error_text.lower():
                    return f"❌ Vector Processing Error: {error_text}"
                elif "cache" in error_text.lower():
                    return f"❌ Cache Error: {error_text}"
                else:
                    return f"❌ Upload failed: {response.status_code} - {error_text}"

        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return f"❌ Error uploading file: {str(e)}"

    async def query_rag(
        self,
        query: str,
        search_strategy: str = "hybrid",
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        include_images: bool = True,
        page_number: Optional[int] = None,
        element_type: Optional[str] = None,
    ) -> str:
        """
        Query the BGE-M3 enhanced RAG system

        Args:
            query: User query
            search_strategy: Search strategy ("hybrid", "dense", "sparse", "multivector")
            alpha: Weight for dense vs sparse search (0.0-1.0)
            beta: Weight for multivector reranking (0.0-1.0)
            gamma: Weight for multivector component (0.0-1.0)
            include_images: Whether to include image results
            page_number: Filter by page number
            element_type: Filter by element type

        Returns:
            BGE-M3 enhanced query response
        """
        if not query.strip():
            return "❌ Please enter a query"

        try:
            # Prepare metadata filters
            metadata_filters = {}
            if page_number is not None:
                metadata_filters["page_number"] = page_number
            if element_type is not None:
                metadata_filters["type"] = element_type

            # Use BGE-M3 regular endpoint
            response = await self.client.post(
                f"{self.api_base_url}/query",
                json={
                    "query": query,
                    "session_id": self.current_session_id,
                    "search_mode": search_strategy,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "include_images": include_images,
                    "metadata_filters": metadata_filters
                    if metadata_filters
                    else None,
                },
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "No response generated")
                total_results = result.get("total_results", 0)
                search_mode = result.get("search_mode", "hybrid")
                response_type = result.get("response_type", "vlm")

                # Add BGE-M3 information if available
                embedding_info = result.get("embedding_info", {})
                bge_m3_metadata = result.get("bge_m3_metadata", {})
                
                if embedding_info or bge_m3_metadata:
                    response_text = f"🤖 **BGE-M3 Enhanced Response**\n\n{response_text}"
                    
                    # Add processing information
                    if embedding_info:
                        vector_types = embedding_info.get("vector_types", [])
                        cache_hit = embedding_info.get("cache_hit", False)
                        processing_time = embedding_info.get("processing_time", 0)
                        cache_stats = embedding_info.get("cache_stats", {})
                        
                        response_text += f"\n\n⏱️ **Processing Time:** {processing_time:.3f}s"
                        response_text += f"\n🔍 **Vector Types:** {', '.join(vector_types)}"
                        response_text += f"\n🎯 **Cache Hit:** {'Yes' if cache_hit else 'No'}"
                        
                        if cache_stats:
                            cache_size = cache_stats.get("cache_size", 0)
                            cache_hits = cache_stats.get("cache_hits", 0)
                            cache_misses = cache_stats.get("cache_misses", 0)
                            response_text += f"\n📊 **Cache Stats:** {cache_size} cached, {cache_hits} hits, {cache_misses} misses"
                    
                    # Add BGE-M3 specific metadata
                    if bge_m3_metadata:
                        multivector_strategy = bge_m3_metadata.get("multivector_strategy", "average")
                        dense_similarity = bge_m3_metadata.get("dense_similarity", 0)
                        sparse_similarity = bge_m3_metadata.get("sparse_similarity", 0)
                        multivector_similarity = bge_m3_metadata.get("multivector_similarity", 0)
                        
                        response_text += f"\n\n🎯 **BGE-M3 Metadata:**"
                        response_text += f"\n   • Multivector Strategy: {multivector_strategy}"
                        response_text += f"\n   • Dense Similarity: {dense_similarity:.3f}"
                        response_text += f"\n   • Sparse Similarity: {sparse_similarity:.3f}"
                        response_text += f"\n   • Multivector Similarity: {multivector_similarity:.3f}"

                # Add results summary
                summary = f"\n\n📊 **Search Summary**"
                summary += f"\n   • Found {total_results} relevant documents"
                summary += f"\n   • Search Mode: {search_mode}"
                summary += f"\n   • Response Type: {response_type}"

                # Add top 3 results with enhanced BGE-M3 information
                if result.get("results", {}).get("items"):
                    summary += "\n\n🔍 **Top Results:**"

                    text_count = 0
                    image_count = 0

                    for i, result_item in enumerate(result["results"]["items"][:3], 1):
                        doc_name = result_item.get("document", "Unknown")
                        page = result_item.get("page", 0)
                        score = result_item.get("score", 0)
                        search_type = result_item.get("search_type", "text")
                        element_type = result_item.get("element_type", "text")
                        vector_types = result_item.get("vector_types", [])
                        bge_m3_scores = result_item.get("bge_m3_scores", {})
                        
                        if search_type == "text":
                            text_count += 1
                            emoji = "📄"
                            if element_type == "table":
                                emoji = "📊"
                            elif element_type == "chart":
                                emoji = "📈"
                            
                            summary += f"\n{i}. {emoji} **{doc_name}** (Page {page}, Score: {score:.3f})"
                            
                            # Add vector type information
                            if vector_types:
                                summary += f"\n   🔍 Vector Types: {', '.join(vector_types)}"
                            
                            # Add BGE-M3 specific scores
                            if bge_m3_scores:
                                dense_score = bge_m3_scores.get("dense_score", 0)
                                sparse_score = bge_m3_scores.get("sparse_score", 0)
                                multivector_score = bge_m3_scores.get("multivector_score", 0)
                                summary += f"\n   🎯 BGE-M3 Scores: Dense={dense_score:.3f}, Sparse={sparse_score:.3f}, Multi={multivector_score:.3f}"
                        else:
                            image_count += 1
                            summary += f"\n{i}. 🖼️ **{doc_name}** (Page {page}, Score: {score:.3f})"

                    # Add result type breakdown
                    if text_count > 0 or image_count > 0:
                        summary += f"\n\n📊 **Result Breakdown:** {text_count} text, {image_count} image"

                return response_text + summary
            else:
                error_text = response.text
                # BGE-M3 specific error messages
                if "embedding" in error_text.lower():
                    return f"❌ BGE-M3 Embedding Error: {error_text}"
                elif "vector" in error_text.lower():
                    return f"❌ Vector Processing Error: {error_text}"
                elif "cache" in error_text.lower():
                    return f"❌ Cache Error: {error_text}"
                elif "search" in error_text.lower():
                    return f"❌ Search Error: {error_text}"
                else:
                    return f"❌ Query failed: {response.status_code} - {error_text}"

        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            # BGE-M3 specific error handling
            error_str = str(e).lower()
            if "embedding" in error_str:
                return f"❌ BGE-M3 Embedding Error: {str(e)}"
            elif "vector" in error_str:
                return f"❌ Vector Processing Error: {str(e)}"
            elif "timeout" in error_str:
                return f"❌ Processing Timeout: {str(e)}"
            else:
                return f"❌ Error querying RAG: {str(e)}"

    async def create_new_session(self) -> str:
        """
        Create a new BGE-M3 enhanced session

        Returns:
            Session creation message
        """
        try:
            # Generate new session ID
            import uuid

            self.current_session_id = str(uuid.uuid4())
            self.uploaded_files = []

            return f"🆕 New session created: {self.current_session_id[:8]}..."

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return f"❌ Error creating session: {str(e)}"

    async def get_session_info(self) -> str:
        """
        Get current BGE-M3 enhanced session information

        Returns:
            Session information with BGE-M3 details
        """
        if not self.current_session_id:
            return "❌ No active session"

        try:
            # Get session information
            response = await self.client.get(
                f"{self.api_base_url}/sessions/{self.current_session_id}"
            )

            if response.status_code == 200:
                session_info = response.json()
                documents = session_info.get("documents", [])
                bge_m3_session_info = session_info.get("bge_m3_session_info", {})
                
                # Build session info display
                session_display = f"📁 **Session:** {self.current_session_id[:8]}...\n\n"
                
                if documents:
                    session_display += "📄 **Documents:**\n"
                    for doc in documents:
                        session_display += f"• {doc}\n"
                else:
                    session_display += "📄 **Documents:**\nNo documents uploaded yet\n"
                
                # Add BGE-M3 specific session information
                if bge_m3_session_info:
                    session_display += "\n🎯 **BGE-M3 Session Info:**\n"
                    
                    # Embedding statistics
                    embedding_stats = bge_m3_session_info.get("embedding_statistics", {})
                    if embedding_stats:
                        session_display += "   **Embedding Statistics:**\n"
                        total_embeddings = embedding_stats.get("total_embeddings", 0)
                        dense_embeddings = embedding_stats.get("dense_embeddings", 0)
                        sparse_embeddings = embedding_stats.get("sparse_embeddings", 0)
                        multivector_embeddings = embedding_stats.get("multivector_embeddings", 0)
                        
                        session_display += f"   • Total Embeddings: {total_embeddings}\n"
                        session_display += f"   • Dense Embeddings: {dense_embeddings}\n"
                        session_display += f"   • Sparse Embeddings: {sparse_embeddings}\n"
                        session_display += f"   • Multivector Embeddings: {multivector_embeddings}\n"
                    
                    # Cache information
                    cache_info = bge_m3_session_info.get("cache_info", {})
                    if cache_info:
                        session_display += "   **Cache Information:**\n"
                        cache_size = cache_info.get("cache_size", 0)
                        cache_hits = cache_info.get("cache_hits", 0)
                        cache_hit_rate = cache_info.get("cache_hit_rate", 0)
                        
                        session_display += f"   • Cache Size: {cache_size} embeddings\n"
                        session_display += f"   • Cache Hits: {cache_hits}\n"
                        session_display += f"   • Cache Hit Rate: {cache_hit_rate:.2%}\n"
                    
                    # Performance metrics
                    performance_metrics = bge_m3_session_info.get("performance_metrics", {})
                    if performance_metrics:
                        session_display += "   **Performance Metrics:**\n"
                        avg_embedding_time = performance_metrics.get("avg_embedding_time", 0)
                        avg_query_time = performance_metrics.get("avg_query_time", 0)
                        total_processing_time = performance_metrics.get("total_processing_time", 0)
                        
                        session_display += f"   • Avg Embedding Time: {avg_embedding_time:.3f}s\n"
                        session_display += f"   • Avg Query Time: {avg_query_time:.3f}s\n"
                        session_display += f"   • Total Processing Time: {total_processing_time:.2f}s\n"
                    
                    # Vector type distribution
                    vector_distribution = bge_m3_session_info.get("vector_distribution", {})
                    if vector_distribution:
                        session_display += "   **Vector Type Distribution:**\n"
                        for vector_type, count in vector_distribution.items():
                            session_display += f"   • {vector_type}: {count}\n"
                
                return session_display
            else:
                return f"❌ Error getting session info: {response.status_code}"

        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return f"❌ Error getting session info: {str(e)}"

    async def clear_session(self) -> str:
        """
        Clear current BGE-M3 enhanced session

        Returns:
            Session clear message
        """
        if not self.current_session_id:
            return "❌ No active session to clear"

        try:
            # Delete session
            response = await self.client.delete(
                f"{self.api_base_url}/sessions/{self.current_session_id}"
            )

            if response.status_code == 200:
                result = response.json()
                self.current_session_id = None
                self.uploaded_files = []
                deleted_count = result.get('deleted_documents', 0)
                return f"🗑️ Session cleared. Deleted {deleted_count} documents."
            else:
                return f"❌ Error clearing session: {response.status_code}"

        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return f"❌ Error clearing session: {str(e)}"

    def create_interface(self) -> gr.Interface:
        """
        Create the BGE-M3 enhanced Gradio interface

        Returns:
            Gradio interface with BGE-M3 specific features
        """
        with gr.Blocks(
            theme=self.theme,
            title="GenericRAG - RAG System",
        ) as demo:
            # Header
            gr.Markdown("# 📚 GenericRAG - BGE-M3 RAG System")
            gr.Markdown("Upload PDF documents and query them using advanced BGE-M3 AI models")

            # Session management
            with gr.Row():
                session_info = gr.Textbox(
                    label="Session Info",
                    placeholder="Click 'New Session' to start",
                    interactive=False,
                    lines=2,
                )
                new_session_btn = gr.Button("🆕 New Session", variant="primary")
                clear_session_btn = gr.Button("🗑️ Clear Session", variant="secondary")

            # Upload tab
            with gr.Tab("📤 Upload Documents"):
                gr.Markdown("Upload PDF documents to add them to your BGE-M3 enhanced knowledge base")

                with gr.Row():
                    file_upload = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        file_count="multiple",
                    )
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        placeholder="Upload status will appear here",
                        interactive=False,
                        lines=2,
                    )

                upload_btn = gr.Button("📤 Upload Files", variant="primary")

                # Uploaded files list
                gr.Markdown("### Uploaded Files")
                uploaded_files_list = gr.Textbox(
                    label="Uploaded Files",
                    placeholder="No files uploaded yet",
                    interactive=False,
                    lines=4,
                )

            # Query tab
            with gr.Tab("🔍 Query Documents"):
                gr.Markdown("Ask questions about your BGE-M3 enhanced documents")

                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask something about your documents...",
                        lines=3,
                    )
                    query_btn = gr.Button("🔍 Ask Question", variant="primary")

                with gr.Row():
                    clear_query_btn = gr.Button("🗑️ Clear Query")

                # Search settings
                with gr.Accordion("⚙️ Search Settings", open=False):
                    with gr.Row():
                        search_strategy = gr.Radio(
                            choices=["hybrid", "dense", "sparse", "multivector"],
                            value="hybrid",
                            label="Search Strategy",
                        )
                        alpha_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Dense vs Sparse Weight",
                        )

                    with gr.Row():
                        include_images = gr.Checkbox(
                            label="Include Images",
                            value=True,
                        )
                        page_filter = gr.Number(
                            label="Filter by Page (optional)",
                            precision=0,
                            minimum=1,
                        )

                # Element type filter
                with gr.Row():
                    element_type = gr.Dropdown(
                        choices=["all", "text", "image", "table", "chart"],
                        value="all",
                        label="Element Type Filter",
                    )

                # BGE-M3 specific settings
                with gr.Accordion("🎯 BGE-M3 Settings", open=False):
                    gr.Markdown("### Advanced BGE-M3 Configuration")
                    
                    with gr.Row():
                        beta_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Multivector Weight (β)",
                        )
                        gamma_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.2,
                            step=0.1,
                            label="Multivector Reranking (γ)",
                        )
                    
                    with gr.Row():
                        multivector_strategy = gr.Dropdown(
                            choices=["average", "max", "sum", "weighted"],
                            value="average",
                            label="Multivector Strategy",
                        )
                        alpha_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Dense vs Sparse Weight (α)",
                        )
                    
                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=512,
                            maximum=8192,
                            value=8192,
                            step=256,
                            label="Max Token Length",
                        )
                        batch_size = gr.Slider(
                            minimum=8,
                            maximum=64,
                            value=32,
                            step=8,
                            label="Batch Size",
                        )
                    
                    with gr.Row():
                        device = gr.Dropdown(
                            choices=["cpu", "cuda", "auto"],
                            value="cpu",
                            label="Processing Device",
                        )
                        cache_embeddings = gr.Checkbox(
                            label="Cache Embeddings",
                            value=True,
                        )

                query_output = gr.Textbox(
                    label="Response",
                    placeholder="Response will appear here...",
                    lines=8,
                    interactive=False,
                )

            # Settings
            with gr.Accordion("⚙️ Settings", open=False):
                gr.Markdown("### API Configuration")
                api_url_display = gr.Textbox(
                    label="API URL",
                    value=self.api_base_url,
                    interactive=False,
                )

                gr.Markdown("### Model Information")
                model_info = gr.Textbox(
                    label="Models",
                    value=f"LLM: {settings.llm.student_model} | Embedding: BGE-M3",
                    interactive=False,
                    lines=1,
                )

            # Event handlers
            new_session_btn.click(fn=self.create_new_session, outputs=[session_info])

            clear_session_btn.click(
                fn=self.clear_session, outputs=[session_info, uploaded_files_list]
            )

            upload_btn.click(
                fn=self.upload_file, inputs=[file_upload], outputs=[upload_status]
            ).then(fn=self.get_session_info, outputs=[session_info]).then(
                fn=lambda: "\n".join(self.uploaded_files)
                if self.uploaded_files
                else "No files uploaded yet",
                outputs=[uploaded_files_list],
            )

            query_btn.click(
                fn=self.query_rag,
                inputs=[
                    query_input,
                    search_strategy,
                    alpha_slider,
                    beta_slider,
                    gamma_slider,
                    include_images,
                    page_filter,
                    element_type,
                ],
                outputs=[query_output],
            )

            clear_query_btn.click(fn=lambda: "", outputs=[query_input])

            # Footer
            gr.Markdown("---")
            gr.Markdown(
                "💡 **Tips:**\n"
                "• Start with a new session for each project\n"
                "• Upload multiple PDFs to build a comprehensive knowledge base\n"
                "• Be specific in your questions for better results\n"
                "• Try different search strategies for optimal results\n"
                "• Use metadata filters to narrow down your search\n"
                "• Adjust the Dense vs Sparse weight (α) for different query types\n"
                "• Use BGE-M3 settings to fine-tune vector search (β, γ)\n"
                "• Choose appropriate multivector strategy for your use case\n"
                "• BGE-M3 generates dense, sparse, and multivector embeddings simultaneously\n"
                "• Cache hits significantly improve response time for repeated queries\n"
                "• Monitor session info for embedding statistics and performance metrics\n"
                "• Use different device settings (CPU/GPU) based on your hardware\n"
                "• Adjust batch size for optimal memory usage and performance"
            )

        return demo


def create_gradio_app() -> gr.Interface:
    """Create and return the BGE-M3 enhanced Gradio application"""
    frontend = GradioFrontend()
    return frontend.create_interface()


if __name__ == "__main__":
    # Create and launch the BGE-M3 enhanced app
    demo = create_gradio_app()

    # Launch with custom settings
    demo.launch(
        server_name=settings.app.host,
        server_port=settings.gradio.port,
        share=False,
        debug=True,
        show_error=True,
    )

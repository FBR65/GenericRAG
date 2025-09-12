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

from app.settings import get_settings

# Get settings
settings = get_settings()


class GradioFrontend:
    """Gradio frontend for the RAG system"""
    
    def __init__(self):
        self.api_base_url = f"http://{settings.app.host}:{settings.app.port}/api/v1"
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Create theme
        self.theme = gr.themes.Monochrome()
        
        # Initialize state
        self.current_session_id = None
        self.uploaded_files = []
        
        logger.info("Initialized Gradio frontend")
    
    async def upload_file(self, file: str) -> str:
        """
        Upload a file to the RAG system
        
        Args:
            file: Base64 encoded file
            
        Returns:
            Upload result message
        """
        try:
            # Decode base64 file
            file_data = base64.b64decode(file.split(',')[1])
            
            # Create file-like object
            file_obj = io.BytesIO(file_data)
            
            # Upload to API
            response = await self.client.post(
                f"{self.api_base_url}/ingest",
                files={"file": ("document.pdf", file_obj, "application/pdf")},
                data={"session_id": self.current_session_id},
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["results"][0]["status"] == "success":
                    self.uploaded_files.append(result["results"][0]["filename"])
                    return f"✅ Successfully uploaded {result['results'][0]['filename']} ({result['results'][0]['num_pages']} pages)"
                else:
                    return f"❌ Error uploading {result['results'][0]['filename']}: {result['results'][0]['error']}"
            else:
                return f"❌ Upload failed: {response.status_code} - {response.text}"
                
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return f"❌ Error uploading file: {str(e)}"
    
    async def query_rag(self, query: str, use_streaming: bool = True) -> str:
        """
        Query the RAG system
        
        Args:
            query: User query
            use_streaming: Whether to use streaming
            
        Returns:
            Query response
        """
        if not query.strip():
            return "❌ Please enter a query"
        
        try:
            if use_streaming:
                # Use streaming endpoint
                response = await self.client.post(
                    f"{self.api_base_url}/query-stream",
                    json={"query": query, "session_id": self.current_session_id},
                    headers={"Accept": "text/event-stream"},
                )
                
                if response.status_code == 200:
                    # Process streaming response
                    full_response = ""
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("status") == "completed":
                                    full_response = f"✅ Query completed. Found {data.get('total_results', 0)} results."
                                    break
                                elif data.get("type") == "text":
                                    full_response += data["content"]
                                elif data.get("error"):
                                    full_response = f"❌ Error: {data['error']}"
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    return full_response or "❌ No response received"
                else:
                    return f"❌ Query failed: {response.status_code} - {response.text}"
            else:
                # Use regular endpoint
                response = await self.client.post(
                    f"{self.api_base_url}/query",
                    json={"query": query, "session_id": self.current_session_id},
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "No response generated")
                    total_results = result.get("total_results", 0)
                    
                    # Add results summary
                    summary = f"\n\n📊 Found {total_results} relevant documents"
                    
                    # Add top 3 results with images if available
                    if result.get("results"):
                        summary += "\n\n🔍 Top Results:"
                        for i, result_item in enumerate(result["results"][:3], 1):
                            doc_name = result_item.get("document", "Unknown")
                            page = result_item.get("page", 0)
                            score = result_item.get("score", 0)
                            summary += f"\n{i}. {doc_name} (Page {page}, Score: {score:.3f})"
                    
                    return response_text + summary
                else:
                    return f"❌ Query failed: {response.status_code} - {response.text}"
                    
        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return f"❌ Error querying RAG: {str(e)}"
    
    async def create_new_session(self) -> str:
        """
        Create a new session
        
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
        Get current session information
        
        Returns:
            Session information
        """
        if not self.current_session_id:
            return "❌ No active session"
        
        try:
            # Get session documents
            response = await self.client.get(
                f"{self.api_base_url}/sessions/{self.current_session_id}/documents"
            )
            
            if response.status_code == 200:
                documents = response.json()
                if documents:
                    doc_list = "\n".join([f"• {doc}" for doc in documents])
                    return f"📁 Session: {self.current_session_id[:8]}...\n\nDocuments:\n{doc_list}"
                else:
                    return f"📁 Session: {self.current_session_id[:8]}...\n\nNo documents uploaded yet"
            else:
                return f"❌ Error getting session info: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return f"❌ Error getting session info: {str(e)}"
    
    async def clear_session(self) -> str:
        """
        Clear current session
        
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
                return f"🗑️ Session cleared. Deleted {result.get('deleted_images', 0)} images."
            else:
                return f"❌ Error clearing session: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return f"❌ Error clearing session: {str(e)}"
    
    def create_interface(self) -> gr.Interface:
        """
        Create the Gradio interface
        
        Returns:
            Gradio interface
        """
        with gr.Blocks(
            theme=self.theme,
            title="GenericRAG - ColPali RAG System",
            description="Upload PDFs and query them using ColPali and DSPy/GEPA optimization"
        ) as demo:
            
            # Header
            gr.Markdown("# 📚 GenericRAG - ColPali RAG System")
            gr.Markdown("Upload PDF documents and query them using advanced AI models")
            
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
                gr.Markdown("Upload PDF documents to add them to your knowledge base")
                
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
                gr.Markdown("Ask questions about your uploaded documents")
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask something about your documents...",
                        lines=3,
                    )
                    query_btn = gr.Button("🔍 Ask Question", variant="primary")
                
                with gr.Row():
                    streaming_toggle = gr.Checkbox(
                        label="Use Streaming Response",
                        value=True,
                    )
                    clear_query_btn = gr.Button("🗑️ Clear Query")
                
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
                    value=f"ColPali: {settings.colpali.colpali_model_name}\nLLM: {settings.llm.student_model}",
                    interactive=False,
                    lines=2,
                )
            
            # Event handlers
            new_session_btn.click(
                fn=self.create_new_session,
                outputs=[session_info]
            )
            
            clear_session_btn.click(
                fn=self.clear_session,
                outputs=[session_info, uploaded_files_list]
            )
            
            upload_btn.click(
                fn=self.upload_file,
                inputs=[file_upload],
                outputs=[upload_status]
            ).then(
                fn=self.get_session_info,
                outputs=[session_info]
            ).then(
                fn=lambda: "\n".join(self.uploaded_files) if self.uploaded_files else "No files uploaded yet",
                outputs=[uploaded_files_list]
            )
            
            query_btn.click(
                fn=self.query_rag,
                inputs=[query_input, streaming_toggle],
                outputs=[query_output]
            )
            
            clear_query_btn.click(
                fn=lambda: "",
                outputs=[query_input]
            )
            
            # Footer
            gr.Markdown("---")
            gr.Markdown(
                "💡 **Tips:**\n"
                "• Start with a new session for each project\n"
                "• Upload multiple PDFs to build a comprehensive knowledge base\n"
                "• Be specific in your questions for better results\n"
                "• Use streaming for real-time response generation"
            )
        
        return demo


def create_gradio_app() -> gr.Interface:
    """Create and return the Gradio application"""
    frontend = GradioFrontend()
    return frontend.create_interface()


if __name__ == "__main__":
    # Create and launch the app
    demo = create_gradio_app()
    
    # Launch with custom settings
    demo.launch(
        server_name=settings.app.host,
        server_port=settings.app.gradio_port,
        share=False,
        debug=True,
        show_error=True,
    )
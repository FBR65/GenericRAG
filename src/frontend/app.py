import gradio as gr
import requests
import json
import os
from typing import List, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenericRAGInterface:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        
        # Set up themes
        self.theme = gr.themes.Monochrome()
        
        # Initialize interface components
        self.setup_interface()
    
    def setup_interface(self):
        """Set up the Gradio interface with two tabs."""
        with gr.Blocks(theme=self.theme, title="Generic RAG System") as self.interface:
            gr.Markdown("# Generic RAG System")
            gr.Markdown("Upload and process documents, then query them using AI-powered search.")
            
            with gr.Tabs():
                # Tab 1: Document Upload & Management
                with gr.TabItem("Document Upload & Management"):
                    self.setup_upload_tab()
                
                # Tab 2: Query Interface
                with gr.TabItem("Query Interface"):
                    self.setup_query_tab()
    
    def setup_upload_tab(self):
        """Set up the document upload and management tab."""
        gr.Markdown("## Upload Documents")
        
        with gr.Row():
            with gr.Column():
                file_upload = gr.File(
                    label="Upload PDF Files",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath"
                )
                
                upload_button = gr.Button("Upload & Process", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                gr.Markdown("### Batch Upload")
                batch_upload_button = gr.Button("Batch Upload All Files", variant="secondary")
                
            with gr.Column():
                # Document list
                gr.Markdown("### Uploaded Documents")
                document_list = gr.Dataframe(
                    headers=["Filename", "Pages", "Status", "First Seen"],
                    datatype=["str", "int", "str", "str"],
                    interactive=False
                )
                
                refresh_button = gr.Button("Refresh Document List")
                delete_button = gr.Button("Delete Selected Document", variant="stop")
        
        # System status
        gr.Markdown("### System Status")
        system_status = gr.JSON(label="System Status")
        
        # Event handlers
        upload_button.click(
            fn=self.upload_documents,
            inputs=[file_upload],
            outputs=[upload_status, document_list]
        )
        
        batch_upload_button.click(
            fn=self.batch_upload_documents,
            inputs=[file_upload],
            outputs=[upload_status, document_list]
        )
        
        refresh_button.click(
            fn=self.refresh_document_list,
            outputs=[document_list]
        )
        
        delete_button.click(
            fn=self.delete_document,
            inputs=[document_list],
            outputs=[upload_status, document_list]
        )
        
        # Initial load
        self.refresh_document_list()
        self.get_system_status()
    
    def setup_query_tab(self):
        """Set up the query interface tab."""
        gr.Markdown("## Dokumente befragen")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Geben Sie die Frage ein",
                    placeholder="Befragen Sie das RAG ...",
                    lines=3
                )
                
                search_button = gr.Button("Suche & Generiere Antwort", variant="primary")
                
                with gr.Accordion("Erweiterte Optionen", open=False):
                    top_k_slider = gr.Slider(
                        label="Anzahl der Ergebnisse",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )
                    
                clear_button = gr.Button("Löschen")
                
            with gr.Column():
                # Results display
                answer_output = gr.Textbox(
                    label="Generiere Antwort",
                    lines=10,
                    interactive=False
                )
                
                sources_output = gr.JSON(label="Quellen")
                
                search_results_output = gr.Dataframe(
                    headers=["Dokumentname", "Seite", "Relevanz Score"],
                    datatype=["str", "int", "float"],
                    interactive=False
                )
        
        # Event handlers
        search_button.click(
            fn=self.search_and_answer,
            inputs=[query_input, top_k_slider],
            outputs=[answer_output, sources_output, search_results_output]
        )
        
        clear_button.click(
            fn=lambda: (None, None, None),
            outputs=[answer_output, sources_output, search_results_output]
        )
    
    def upload_documents(self, files):
        """Upload and process documents."""
        if not files:
            return "Keine Dateien ausgewählt", None
        
        results = []
        for file in files:
            try:
                # For filepath type, file is already a path string
                if isinstance(file, str):
                    file_path = file
                else:
                    # Save uploaded file
                    upload_dir = "uploads"
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    file_path = os.path.join(upload_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                
                # Process the document
                response = self.session.post(
                    f"{self.api_base_url}/process",
                    data={"file_path": file_path}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append(f"✓ {file.name}: {result.get('processed_pages', 0)} pages processed")
                else:
                    results.append(f"✗ {file.name}: {response.text}")
                
                # Clean up
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            except Exception as e:
                results.append(f"✗ {file.name}: {str(e)}")
        
        status_message = "\n".join(results)
        return status_message, self.refresh_document_list()
    
    def batch_upload_documents(self, files):
        """Batch upload multiple documents."""
        if not files:
            return "Keine Dateien ausgewählt", None
        
        try:
            # Prepare files for upload
            files_data = []
            for file in files:
                if isinstance(file, str):
                    # For filepath type, read the file and create tuple
                    with open(file, "rb") as f:
                        file_content = f.read()
                    files_data.append(("files", (os.path.basename(file), file_content, "application/pdf")))
                else:
                    # For binary type, use the file object directly
                    files_data.append(("files", (file.name, file, "application/pdf")))
            
            # Upload files
            response = self.session.post(
                f"{self.api_base_url}/batch-upload",
                files=files_data
            )
            
            if response.status_code == 200:
                result = response.json()
                status_message = f"Batch Upload Fertig: {result.get('total_files', 0)} Dateien verarbeitet"
                return status_message, self.refresh_document_list()
            else:
                return f"Batch upload failed: {response.text}", None
                
        except Exception as e:
            return f"Batch upload error: {str(e)}", None
    
    def refresh_document_list(self):
        """Refresh the document list."""
        try:
            response = self.session.get(f"{self.api_base_url}/list-documents")
            if response.status_code == 200:
                documents = response.json().get("documents", [])
                
                # Format data for display
                data = []
                for doc in documents:
                    data.append([
                        doc.get("filename", "Unknown"),
                        doc.get("page_count", 0),
                        "Processed",
                        doc.get("first_seen", "Unknown")
                    ])
                
                return data
            else:
                return [["Error", 0, "Failed to load", ""]]
                
        except Exception as e:
            logger.error(f"Error refreshing document list: {str(e)}")
            return [["Error", 0, str(e), ""]]
    
    def delete_document(self, document_list):
        """Delete a selected document."""
        if not document_list or len(document_list) == 0:
            return "No document selected", None
        
        # Get the selected filename (first row, first column)
        selected_filename = document_list[0][0] if document_list else None
        
        if not selected_filename:
            return "No document selected", None
        
        try:
            response = self.session.delete(
                f"{self.api_base_url}/delete",
                data={"filename": selected_filename}
            )
            
            if response.status_code == 200:
                status_message = f"Document '{selected_filename}' deleted successfully"
                return status_message, self.refresh_document_list()
            else:
                return f"Failed to delete document: {response.text}", None
                
        except Exception as e:
            return f"Error deleting document: {str(e)}", None
    
    def search_and_answer(self, query, top_k):
        """Search for documents and generate an answer."""
        if not query.strip():
            return "Please enter a query", None, None
        
        try:
            response = self.session.post(
                f"{self.api_base_url}/search",
                data={"query": query, "top_k": top_k}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                answer = result.get("answer", "No answer generated")
                sources = result.get("sources", [])
                
                # Format search results
                search_results = []
                for source in sources:
                    search_results.append([
                        source.get("filename", "Unknown"),
                        source.get("page_number", 0),
                        source.get("relevance_score", 0.0)
                    ])
                
                return answer, sources, search_results
            else:
                return f"Search failed: {response.text}", None, None
                
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return f"Search error: {str(e)}", None, None
    
    def get_system_status(self):
        """Get system status."""
        try:
            response = self.session.get(f"{self.api_base_url}/system-status")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Failed to get system status"}
                
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {"error": str(e)}
    
    def launch(self, server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False, **kwargs):
        """Launch the Gradio interface."""
        
        self.interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            **kwargs
        )

def main():
    """Main function to launch the Gradio interface."""
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("Warning: API server may not be running or is unhealthy")
    except requests.exceptions.ConnectionError:
        print("Warning: Cannot connect to API server at http://localhost:8000")
        print("Please start the API server first: python -m src.api.main")
        return
    
    # Create and launch the interface
    interface = GenericRAGInterface()
    interface.launch()

if __name__ == "__main__":
    main()
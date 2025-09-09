#!/usr/bin/env python3
"""
Startup script for the Generic RAG Gradio frontend.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.frontend.app import GenericRAGInterface
from src.config import settings

def main():
    """Start the Gradio frontend."""
    print(f"Starting {settings.system_name} Frontend...")
    print("Make sure the API server is running first:")
    print("  python start_api.py")
    print("-" * 50)
    
    # Create and launch the interface
    interface = GenericRAGInterface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )

if __name__ == "__main__":
    main()
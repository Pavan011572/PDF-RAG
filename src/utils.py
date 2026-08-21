import os
import shutil
from typing import List, Dict, Any

# Path definitions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")
IMAGES_DIR = os.path.join(DATA_DIR, "extracted_images")


def ensure_directories_exist() -> None:
    """Ensure all required runtime data directories exist."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def clear_directory(directory_path: str) -> None:
    """Remove all files and subdirectories inside a given directory while keeping the directory itself."""
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            if item == ".gitkeep":
                continue
            item_path = os.path.join(directory_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")


def clear_knowledge_base() -> None:
    """Clean uploaded files, extracted images, and FAISS vector index files."""
    clear_directory(UPLOADS_DIR)
    clear_directory(VECTORSTORE_DIR)
    clear_directory(IMAGES_DIR)


def format_citations(sources: List[Dict[str, Any]]) -> List[str]:
    """Format source information for presentation in the UI."""
    formatted_citations = []
    seen = set()
    for src in sources:
        source_name = src.get("source", "Unknown Document")
        page_num = src.get("page_number", "Unknown")
        citation_key = f"{source_name}_p{page_num}"
        if citation_key not in seen:
            seen.add(citation_key)
            formatted_citations.append(f"• **{source_name}** — Page {page_num}")
    return formatted_citations

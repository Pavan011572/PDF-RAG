import os
from typing import Optional, List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.utils import VECTORSTORE_DIR, ensure_directories_exist


class VectorStoreManager:
    """Manages creation, loading, saving, and persistence of the FAISS vector database."""

    INDEX_NAME = "index"

    @classmethod
    def create_vector_store(
        cls,
        documents: List[Document],
        embeddings
    ) -> FAISS:
        """Create a new FAISS vector store from document chunks and embeddings."""
        ensure_directories_exist()
        if not documents:
            raise ValueError("Cannot create vector store with empty document list.")

        print(f"Creating FAISS vector index with {len(documents)} document chunks...")
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        return vector_store

    @classmethod
    def save_vector_store(
        cls,
        vector_store: FAISS,
        folder_path: str = VECTORSTORE_DIR
    ) -> None:
        """Save FAISS index and metadata to local disk."""
        ensure_directories_exist()
        vector_store.save_local(folder_path=folder_path, index_name=cls.INDEX_NAME)
        print(f"FAISS index successfully saved to: {folder_path}")

    @classmethod
    def load_vector_store(
        cls,
        embeddings,
        folder_path: str = VECTORSTORE_DIR
    ) -> Optional[FAISS]:
        """Load FAISS index and metadata from local disk if it exists."""
        if not cls.vector_store_exists(folder_path):
            return None

        try:
            print(f"Loading FAISS index from: {folder_path}")
            vector_store = FAISS.load_local(
                folder_path=folder_path,
                embeddings=embeddings,
                index_name=cls.INDEX_NAME,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
            return None

    @classmethod
    def vector_store_exists(cls, folder_path: str = VECTORSTORE_DIR) -> bool:
        """Check if a saved FAISS index exists at the target directory."""
        faiss_file = os.path.join(folder_path, f"{cls.INDEX_NAME}.faiss")
        pkl_file = os.path.join(folder_path, f"{cls.INDEX_NAME}.pkl")
        return os.path.exists(faiss_file) and os.path.exists(pkl_file)

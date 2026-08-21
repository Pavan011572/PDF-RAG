import os
from typing import List, Dict, Any, Optional

from src.pdf_loader import PDFLoader
from src.text_splitter import TextSplitterManager
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.llm import LLMManager
from src.utils import VECTORSTORE_DIR


class RAGPipeline:
    """
    End-to-End Retrieval-Augmented Generation Pipeline.
    Orchestrates PDF ingestion, chunking, embedding, FAISS indexing, similarity retrieval, and LLM answer generation.
    Supports Google Gemini (Free) and OpenAI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "gemini",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.embedding_manager = EmbeddingManager(model_name=embedding_model_name)
        self.llm_manager = LLMManager(api_key=api_key, provider=provider)
        self.vector_store = None
        self._load_existing_vector_store()

    def set_api_key(self, api_key: str, provider: str = "gemini") -> None:
        """Update the LLM API Key and provider."""
        self.llm_manager.api_key = api_key
        self.llm_manager.provider = provider.lower()

    def _load_existing_vector_store(self) -> None:
        """Attempt to load an existing persisted FAISS vector index."""
        embeddings = self.embedding_manager.get_embedding_model()
        self.vector_store = VectorStoreManager.load_vector_store(embeddings=embeddings)

    def build_knowledge_base(
        self,
        file_paths: List[str],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Process PDF files, extract text, split into chunks, generate embeddings, and build FAISS index.
        """
        # Step 1: Extract text page-by-page from PDFs (with Vision OCR fallback for image/handwritten PDFs)
        raw_documents = PDFLoader.load_multiple_pdfs(file_paths, llm_manager=self.llm_manager)
        if not raw_documents:
            raise ValueError("No text could be extracted from the uploaded PDF documents.")

        # Step 2: Split text into chunks with metadata
        splitter = TextSplitterManager(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_documents = splitter.split_documents(raw_documents)
        if not chunked_documents:
            raise ValueError("No valid text chunks created from documents.")

        # Step 3 & 4: Embed chunks and build FAISS index
        embeddings = self.embedding_manager.get_embedding_model()
        self.vector_store = VectorStoreManager.create_vector_store(
            documents=chunked_documents,
            embeddings=embeddings
        )

        # Step 5: Save FAISS index locally
        VectorStoreManager.save_vector_store(self.vector_store)

        return {
            "num_documents": len(file_paths),
            "num_pages": len(raw_documents),
            "num_chunks": len(chunked_documents)
        }

    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.

        Returns a dictionary containing:
        - answer: LLM answer string
        - sources: List of source documents with page numbers & scores & image paths
        - context_chunks: Detailed retrieved chunks
        """
        if not self.vector_store:
            self._load_existing_vector_store()

        if not self.vector_store:
            return {
                "answer": "No knowledge base available. Please upload a PDF and click 'Build Knowledge Base' first.",
                "sources": [],
                "context_chunks": []
            }

        # Step 1 & 2: Similarity search using FAISS
        retriever = RAGRetriever(self.vector_store)
        context_chunks = retriever.retrieve(query=question, top_k=top_k)

        if not context_chunks:
            return {
                "answer": "I could not find the answer in the uploaded document.",
                "sources": [],
                "context_chunks": []
            }

        # Step 3: LLM generation
        answer = self.llm_manager.generate_answer(question=question, context_chunks=context_chunks)

        # Build clean source summary list with image paths
        sources = []
        for chunk in context_chunks:
            sources.append({
                "source": chunk["source"],
                "page": chunk["page_number"],
                "score": chunk["score"],
                "image_path": chunk.get("image_path", "")
            })


        return {
            "answer": answer,
            "sources": sources,
            "context_chunks": context_chunks
        }

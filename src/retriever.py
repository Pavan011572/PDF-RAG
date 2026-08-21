from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS


class RAGRetriever:
    """Handles semantic similarity search against the FAISS vector index."""

    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search for a given question.

        Returns a list of dictionaries with content, metadata, and relevance scores.
        """
        if not query or not query.strip():
            return []

        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")

        # Similarity search with distance score
        results_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )

        retrieved_chunks = []
        for doc, score in results_with_scores:
            retrieved_chunks.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown Document"),
                "page_number": doc.metadata.get("page_number", 0),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "image_path": doc.metadata.get("image_path", ""),
                "score": round(float(score), 4)
            })


        return retrieved_chunks

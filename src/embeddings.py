from typing import Optional, List
from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingManager:
    """
    Manages loading and caching of Sentence Transformer embedding models.
    Default model: 'all-MiniLM-L6-v2'
    """
    _cached_embeddings: Optional[HuggingFaceEmbeddings] = None
    _cached_model_name: Optional[str] = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        Returns a cached instance of HuggingFaceEmbeddings to avoid reloading model weights repeatedly.
        """
        if (
            EmbeddingManager._cached_embeddings is None
            or EmbeddingManager._cached_model_name != self.model_name
        ):
            print(f"Loading embedding model: {self.model_name}...")
            EmbeddingManager._cached_embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            EmbeddingManager._cached_model_name = self.model_name

        return EmbeddingManager._cached_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        model = self.get_embedding_model()
        return model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document chunk texts."""
        model = self.get_embedding_model()
        return model.embed_documents(texts)

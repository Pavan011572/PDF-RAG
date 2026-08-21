import pytest
from langchain_core.documents import Document
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.llm import LLMManager


def test_vector_store_and_retrieval():
    """Verify vector index creation, search, and top-K relevance retrieval."""
    docs = [
        Document(
            page_content="The model achieved an accuracy of 94.6% on the benchmark dataset.",
            metadata={"source": "report.pdf", "page_number": 3, "chunk_id": "report.pdf_p3_c01"}
        ),
        Document(
            page_content="Gradient descent optimization was performed using Adam optimizer.",
            metadata={"source": "report.pdf", "page_number": 4, "chunk_id": "report.pdf_p4_c01"}
        ),
        Document(
            page_content="The weather forecast predicts sunny skies and warm temperatures.",
            metadata={"source": "weather.pdf", "page_number": 1, "chunk_id": "weather.pdf_p1_c01"}
        )
    ]

    embedding_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    embeddings = embedding_mgr.get_embedding_model()

    vector_store = VectorStoreManager.create_vector_store(docs, embeddings)
    retriever = RAGRetriever(vector_store)

    # Search query
    query = "What accuracy did the model achieve?"
    results = retriever.retrieve(query=query, top_k=2)

    assert len(results) == 2
    top_result = results[0]
    assert "94.6%" in top_result["content"]
    assert top_result["source"] == "report.pdf"
    assert top_result["page_number"] == 3
    assert "score" in top_result


def test_llm_prompt_rendering():
    """Verify QA prompt construction includes rules and context chunks."""
    llm_mgr = LLMManager(api_key="mock_key")
    context_chunks = [
        {
            "content": "The proposed architecture uses a 12-layer Transformer.",
            "source": "paper.pdf",
            "page_number": 7,
            "score": 0.15
        }
    ]

    prompt = llm_mgr.render_prompt(
        question="What architecture was proposed?",
        context_chunks=context_chunks
    )

    assert "You are a document question-answering assistant." in prompt
    assert "[Chunk 1] (Source: paper.pdf, Page: 7)" in prompt
    assert "The proposed architecture uses a 12-layer Transformer." in prompt
    assert "Question:\nWhat architecture was proposed?" in prompt

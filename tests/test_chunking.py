import pytest
from src.text_splitter import TextSplitterManager


def test_chunking_metadata_preservation():
    """Verify that chunking retains source, page number, and assigns unique chunk_ids."""
    raw_documents = [
        {
            "content": "This is a long sentence meant to test text chunking functionality. " * 10,
            "source": "research_paper.pdf",
            "page_number": 5
        }
    ]

    splitter = TextSplitterManager(chunk_size=100, chunk_overlap=20)
    chunked_docs = splitter.split_documents(raw_documents)

    assert len(chunked_docs) > 1

    for idx, doc in enumerate(chunked_docs):
        assert doc.metadata["source"] == "research_paper.pdf"
        assert doc.metadata["page_number"] == 5
        assert doc.metadata["chunk_id"] == f"research_paper.pdf_p5_c{idx + 1:02d}"
        assert len(doc.page_content) <= 120  # Max length around chunk_size + margin


def test_chunking_multiple_pages():
    """Verify chunking across multiple pages preserves independent page metadata."""
    raw_documents = [
        {
            "content": "Page 1 content " * 5,
            "source": "doc.pdf",
            "page_number": 1
        },
        {
            "content": "Page 2 content " * 5,
            "source": "doc.pdf",
            "page_number": 2
        }
    ]

    splitter = TextSplitterManager(chunk_size=50, chunk_overlap=10)
    chunked_docs = splitter.split_documents(raw_documents)

    page1_chunks = [d for d in chunked_docs if d.metadata["page_number"] == 1]
    page2_chunks = [d for d in chunked_docs if d.metadata["page_number"] == 2]

    assert len(page1_chunks) > 0
    assert len(page2_chunks) > 0
    assert page1_chunks[0].metadata["chunk_id"] == "doc.pdf_p1_c01"
    assert page2_chunks[0].metadata["chunk_id"] == "doc.pdf_p2_c01"

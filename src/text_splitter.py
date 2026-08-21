from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextSplitterManager:
    """Manages document chunking using RecursiveCharacterTextSplitter while maintaining metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, raw_documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Split raw document pages into smaller, meaningful text chunks.

        Each input doc has:
        - content: str
        - source: str
        - page_number: int

        Returns a list of LangChain Document objects containing chunk metadata and a unique chunk_id.
        """
        chunked_documents: List[Document] = []

        for doc in raw_documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            page_number = doc.get("page_number", 0)
            image_path = doc.get("image_path", "")

            # Split text using LangChain text splitter
            chunks = self.splitter.split_text(content)

            for chunk_idx, chunk_text in enumerate(chunks):
                # Create unique chunk identifier e.g. research_paper.pdf_p5_c03
                chunk_id = f"{source}_p{page_number}_c{chunk_idx + 1:02d}"

                metadata = {
                    "source": source,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                    "image_path": image_path,
                }


                chunked_documents.append(
                    Document(page_content=chunk_text, metadata=metadata)
                )

        return chunked_documents

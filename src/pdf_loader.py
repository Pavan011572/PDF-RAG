import os
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from src.utils import IMAGES_DIR, ensure_directories_exist


class PDFProcessingError(Exception):
    """Custom exception raised for errors during PDF processing."""
    pass


class PDFLoader:
    """Handles PDF validation, page image rendering, and OCR/Text extraction."""

    @staticmethod
    def validate_pdf(file_path: str) -> bool:
        """Validate if a file is a non-empty, readable PDF file."""
        if not os.path.exists(file_path):
            raise PDFProcessingError(f"File not found: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise PDFProcessingError(f"Invalid file extension. Expected .pdf: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise PDFProcessingError(f"Uploaded file is empty (0 bytes): {file_path}")

        return True

    @classmethod
    def load_pdf(cls, file_path: str, llm_manager: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Extract text page-by-page from a single PDF document.
        For handwritten or image-based pages, renders the page to PNG and uses Vision AI for OCR.

        Returns a list of dictionaries, each containing:
        - content: Extracted or OCR-transcribed text from page
        - source: PDF filename
        - page_number: 1-indexed page number
        - image_path: Path to rendered page PNG image
        """
        cls.validate_pdf(file_path)
        ensure_directories_exist()
        filename = os.path.basename(file_path)
        documents = []

        try:
            doc = fitz.open(file_path)
            if doc.is_encrypted:
                if not doc.authenticate(""):
                    raise PDFProcessingError(
                        f"The PDF '{filename}' is password-protected. Please remove the password before uploading."
                    )

            if doc.page_count == 0:
                raise PDFProcessingError(f"PDF contains no pages: {filename}")

            total_extracted_text = ""
            for i, page in enumerate(doc):
                page_number = i + 1

                # 1. Save page as high-res PNG image
                pix = page.get_pixmap(dpi=200)
                clean_base = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in os.path.splitext(filename)[0]])
                image_filename = f"{clean_base}_p{page_number}.png"
                image_path = os.path.join(IMAGES_DIR, image_filename)
                pix.save(image_path)


                # 2. Try standard embedded text extraction
                text = page.get_text("text") or ""
                text = text.strip()

                # 3. If page contains little/no text (handwritten or scanned image PDF), use Vision OCR
                if len(text) < 30 and llm_manager and hasattr(llm_manager, "transcribe_image"):
                    print(f"Page {page_number} of '{filename}' appears image-based/handwritten. Running Vision OCR...")
                    ocr_text = llm_manager.transcribe_image(image_path)
                    if ocr_text:
                        text = ocr_text.strip()

                if text:
                    total_extracted_text += text
                    documents.append({
                        "content": text,
                        "source": filename,
                        "page_number": page_number,
                        "image_path": image_path
                    })

            doc.close()

            if not total_extracted_text:
                raise PDFProcessingError(
                    f"The PDF '{filename}' does not contain extractable text or readable images."
                )

        except PDFProcessingError:
            raise
        except Exception as e:
            raise PDFProcessingError(f"Error extracting text from '{filename}': {str(e)}")

        return documents

    @classmethod
    def load_multiple_pdfs(cls, file_paths: List[str], llm_manager: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Extract text from multiple PDF files and combine document pages."""
        all_documents = []
        errors = []

        for path in file_paths:
            try:
                docs = cls.load_pdf(path, llm_manager=llm_manager)
                all_documents.extend(docs)
            except PDFProcessingError as e:
                errors.append(str(e))

        if errors and not all_documents:
            raise PDFProcessingError(" | ".join(errors))

        return all_documents

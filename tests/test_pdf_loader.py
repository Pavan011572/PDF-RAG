import os
import tempfile
import pytest
from src.pdf_loader import PDFLoader, PDFProcessingError


def create_sample_pdf(file_path: str, pages_text: list[str]) -> str:
    """Helper to generate a test PDF file with text content on specific pages."""
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(file_path)
        for text in pages_text:
            c.drawString(100, 700, text)
            c.showPage()
        c.save()
    except ImportError:
        # Minimal valid PDF string fallback if reportlab is not available
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
            b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
            b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
            b"4 0 obj <</Length 44>> stream\nBT /F1 12 Tf 100 700 Td (Sample test document text) Tj ET\nendstream\nendobj\n"
            b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000338 00000 n \n"
            b"trailer <</Size 6 /Root 1 0 R>>\nstartxref\n415\n%%EOF\n"
        )
        with open(file_path, "wb") as f:
            f.write(pdf_content)

    return file_path


def test_validate_nonexistent_file():
    """Test validation fails for non-existent files."""
    with pytest.raises(PDFProcessingError, match="File not found"):
        PDFLoader.validate_pdf("non_existent_file.pdf")


def test_validate_invalid_extension():
    """Test validation fails for non-PDF extensions."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"Hello world")
        tmp_path = tmp.name

    try:
        with pytest.raises(PDFProcessingError, match="Invalid file extension"):
            PDFLoader.validate_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_validate_empty_file():
    """Test validation fails for empty 0-byte PDF files."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with pytest.raises(PDFProcessingError, match="empty"):
            PDFLoader.validate_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_load_pdf_success():
    """Test successful text extraction and page metadata preservation."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pages = [
            "Machine learning is a field of study in artificial intelligence.",
            "Retrieval Augmented Generation combines search with LLMs."
        ]
        create_sample_pdf(tmp_path, pages)

        docs = PDFLoader.load_pdf(tmp_path)

        assert len(docs) == 2
        assert "Machine learning" in docs[0]["content"]
        assert docs[0]["page_number"] == 1
        assert docs[0]["source"] == os.path.basename(tmp_path)

        assert "Retrieval Augmented" in docs[1]["content"]
        assert docs[1]["page_number"] == 2

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

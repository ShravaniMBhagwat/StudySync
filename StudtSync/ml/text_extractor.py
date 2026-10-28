"""
ml/text_extractor.py — Text extraction utilities for StudySync.

Supports:
  - PDF files via PyMuPDF (fitz)
  - Plain-text (.txt) files

Usage:
    from ml.text_extractor import extract_text

    text = extract_text("path/to/note.pdf")
    text = extract_text("path/to/note.txt")
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class MalformedPDFError(ValueError):
    """Raised when a PDF cannot be opened or parsed."""


class ScannedPDFError(ValueError):
    """Raised when a PDF has no extractable text (likely an image scan)."""


def extract_text(filepath: str) -> str:
    """
    Extract and return the full text content from a PDF or TXT file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the file.

    Returns
    -------
    str
        Extracted plain text. An empty TXT remains an empty string.

    Raises
    ------
    ValueError
        If the file extension is unsupported or the PDF is malformed/scanned.
    FileNotFoundError
        If the file does not exist at the given path.
    """
    ext = os.path.splitext(filepath)[1].lower()

    # Validate extension first — unsupported types are always an error
    # regardless of whether the file exists.
    if ext not in (".pdf", ".txt"):
        raise ValueError(
            f"Unsupported file type '{ext}'. Only .pdf and .txt are supported."
        )

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if ext == ".pdf":
        return _extract_pdf(filepath)
    return _extract_txt(filepath)


def _extract_pdf(filepath: str) -> str:
    """
    Use PyMuPDF (fitz) to extract text page-by-page from a PDF.

    Each page's text is separated by a newline. PDFs without extractable
    text raise ScannedPDFError so callers can explain that OCR is unavailable.
    """
    try:
        import pymupdf as fitz  # PyMuPDF (pymupdf is the modern import name)

        pages = []
        with open(filepath, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise MalformedPDFError("Encrypted PDFs are not supported.")
            for page in doc:
                text = page.get_text("text")  # plain text mode
                if text.strip():
                    pages.append(text)
        if not pages:
            raise ScannedPDFError("PDF contains no extractable text.")
        return "\n".join(pages)
    except (MalformedPDFError, ScannedPDFError):
        raise
    except Exception as exc:
        logger.warning("PDF parsing failed (%s)", type(exc).__name__)
        raise MalformedPDFError("PDF could not be opened or parsed.") from exc


def _extract_txt(filepath: str) -> str:
    """
    Read a plain-text file with UTF-8 encoding (fallback to latin-1).
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(filepath, "r", encoding=encoding) as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
    return ""


def get_page_count(filepath: str) -> Optional[int]:
    """
    Return the number of pages in a PDF, or None for non-PDF files.
    Used by the QuestionBank mapper to record page references.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".pdf":
        return None
    try:
        import pymupdf as fitz
        with open(filepath, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            return len(doc)
    except Exception as exc:
        logger.warning("PDF page count failed (%s)", type(exc).__name__)
        return None


def extract_pages(filepath: str) -> list[dict]:
    """
    Extract text per-page from a PDF, preserving page numbers.

    Returns
    -------
    list of dict, each with keys:
        'page_number' (int, 1-indexed)
        'text'        (str)
    """
    import pymupdf as fitz
    results = []
    if not filepath.lower().endswith(".pdf"):
        return [{"page_number": 1, "text": _extract_txt(filepath)}]

    try:
        with open(filepath, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    results.append({"page_number": i, "text": text})
    except Exception as exc:
        logger.warning("PDF page extraction failed (%s)", type(exc).__name__)
        raise MalformedPDFError("PDF could not be opened or parsed.") from exc
    if not results:
        raise ScannedPDFError("PDF contains no extractable text.")
    return results

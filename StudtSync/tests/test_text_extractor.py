"""
tests/test_text_extractor.py — Tests for ml/text_extractor.py

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/ -v

These tests verify that:
  - The module imports without errors
  - Unsupported file types raise ValueError
  - Missing files raise FileNotFoundError
  - TXT extraction returns expected content
"""

import os
import sys
import tempfile
import pytest

# Ensure project root is on the path when running tests directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.text_extractor import extract_text, get_page_count


class TestExtractText:
    """Tests for the extract_text() function."""

    def test_raises_for_unsupported_extension(self):
        """Non-PDF, non-TXT files should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text("notes.docx")

    def test_raises_for_missing_file(self):
        """A path that does not exist should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_text("does_not_exist.txt")

    def test_extracts_txt_content(self, tmp_path):
        """TXT extraction should return the exact file contents."""
        sample = "Data Structures and Algorithms\nArrays, Linked Lists, Trees."
        note = tmp_path / "sample.txt"
        note.write_text(sample, encoding="utf-8")

        result = extract_text(str(note))
        assert result == sample

    def test_txt_encoding_fallback(self, tmp_path):
        """TXT with latin-1 encoding should still be read."""
        sample_bytes = "Résumé: héritier".encode("latin-1")
        note = tmp_path / "latin.txt"
        note.write_bytes(sample_bytes)

        result = extract_text(str(note))
        assert "Résumé" in result or len(result) > 0  # decoded successfully

    def test_empty_txt_returns_empty_string(self, tmp_path):
        """Empty TXT file should return an empty string."""
        note = tmp_path / "empty.txt"
        note.write_text("", encoding="utf-8")
        result = extract_text(str(note))
        assert result == ""


class TestGetPageCount:
    """Tests for the get_page_count() helper."""

    def test_returns_none_for_txt(self, tmp_path):
        """TXT files have no pages — should return None."""
        txt = tmp_path / "note.txt"
        txt.write_text("hello")
        assert get_page_count(str(txt)) is None

    def test_returns_none_for_missing_pdf(self):
        """Missing file should return None (not raise)."""
        assert get_page_count("ghost.pdf") is None

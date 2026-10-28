"""Route and upload tests for the StudySync Flask application."""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app


@pytest.fixture
def client(tmp_path):
    application = flask_app.app
    original_config = {
        key: application.config.get(key)
        for key in ("UPLOAD_FOLDER", "MAX_CONTENT_LENGTH", "TESTING", "SECRET_KEY")
    }
    application.config.update(
        TESTING=True,
        SECRET_KEY="test-secret",
        UPLOAD_FOLDER=str(tmp_path),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    )
    with application.test_client() as test_client:
        yield test_client
    application.config.update(original_config)


def text_pdf(text: str = "StudySync test document") -> bytes:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


class TestRoutes:
    @pytest.mark.parametrize(
        "path",
        ["/", "/dashboard", "/notemap", "/questionbank", "/analytics", "/about"],
    )
    def test_page_routes(self, client, path):
        response = client.get(path)
        assert response.status_code == 200

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json == {"status": "ok"}

    def test_dsbd_legacy_route(self, client):
        response = client.get("/dsbd", follow_redirects=True)
        assert response.status_code == 200

    def test_source_code_page_lists_and_renders_project_files(self, client):
        home = client.get("/")
        assert b"Source code" in home.data
        assert b"Mini-Project" not in home.data
        assert b"Getting Started" not in home.data

        listing = client.get("/source-code")
        assert listing.status_code == 200
        assert b"app.py" in listing.data
        assert b"ml/classifier.py" in listing.data

        source = client.get("/source-code?file=ml%2Fclassifier.py")
        assert source.status_code == 200
        assert b"TF-IDF subject classification" in source.data

    @pytest.mark.parametrize(
        "file_path",
        ["../config.py", "uploads/example.py", ".venv/pyvenv.cfg"],
    )
    def test_source_code_page_rejects_files_outside_source_catalog(self, client, file_path):
        assert client.get("/source-code", query_string={"file": file_path}).status_code == 404

    def test_404_for_unknown_route(self, client):
        assert client.get("/this-page-does-not-exist").status_code == 404


class TestNoteMapUploads:
    def test_note_upload_shows_prediction_scores_and_review_option(self, client):
        response = client.post(
            "/notemap/analyze",
            data={
                "note_file": (
                    io.BytesIO(
                        b"A queue processes items in first in first out order. "
                        b"Enqueue inserts and dequeue removes the front item."
                    ),
                    "queue_notes.txt",
                )
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        assert b"Data Structures" in response.data
        assert b"decision margin" in response.data or b"log score" in response.data
        assert b"Review or correct this prediction" in response.data
        assert b"real student notes" in response.data

    def test_prediction_correction_is_reviewed_but_not_persisted(self, client):
        response = client.post(
            "/notemap/correct",
            data={
                "predicted_subject": "Data Structures",
                "corrected_subject": "Operating Systems",
            },
        )
        assert response.status_code == 200
        assert b"Operating Systems" in response.data
        assert b"not stored" in response.data

    def test_unknown_prediction_can_be_corrected(self, client):
        response = client.post(
            "/notemap/correct",
            data={
                "predicted_subject": "Unknown / outside the demonstration vocabulary",
                "corrected_subject": "Artificial Intelligence",
            },
        )
        assert response.status_code == 200
        assert b"Artificial Intelligence" in response.data

    def test_prediction_correction_rejects_unknown_subject(self, client):
        response = client.post(
            "/notemap/correct",
            data={
                "predicted_subject": "not a subject",
                "corrected_subject": "Operating Systems",
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Choose a subject from the available list" in response.data

    def test_extracts_txt_and_removes_temporary_upload(self, client, tmp_path):
        response = client.post(
            "/notemap/analyze",
            data={"note_file": (io.BytesIO(b"Arrays and linked lists"), "../../lecture.txt")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        assert b"lecture.txt" in response.data
        assert b"23 characters" in response.data
        assert list(tmp_path.iterdir()) == []

    def test_extracts_text_pdf_and_removes_temporary_upload(self, client, tmp_path):
        response = client.post(
            "/notemap/analyze",
            data={"note_file": (io.BytesIO(text_pdf()), "notes.pdf")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        assert b"notes.pdf" in response.data
        assert b"1 pages" in response.data
        assert list(tmp_path.iterdir()) == []

    def test_missing_file_has_clear_message(self, client):
        response = client.post("/notemap/analyze", data={}, follow_redirects=True)
        assert response.status_code == 200
        assert b"Please choose a note file" in response.data

    def test_rejects_unsupported_extension(self, client):
        response = client.post(
            "/notemap/analyze",
            data={"note_file": (io.BytesIO(b"not a note"), "notes.docx")},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"must be a .pdf, .txt file" in response.data

    def test_rejects_empty_file(self, client):
        response = client.post(
            "/notemap/analyze",
            data={"note_file": (io.BytesIO(b""), "empty.txt")},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"file is empty" in response.data

    def test_rejects_blank_text_file(self, client):
        response = client.post(
            "/notemap/analyze",
            data={"note_file": (io.BytesIO(b"  \n "), "blank.txt")},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"contains no text" in response.data

    def test_scanned_pdf_explains_ocr_not_supported(self, client):
        response = client.post(
            "/notemap/analyze",
            data={"note_file": (io.BytesIO(text_pdf("")), "scan.pdf")},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Scanned PDFs are not supported yet; OCR is not available" in response.data

    def test_rejects_malformed_pdf_and_cleans_temporary_upload(self, client, tmp_path):
        response = client.post(
            "/notemap/analyze",
            data={"note_file": (io.BytesIO(b"not a PDF"), "broken.pdf")},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"valid, unencrypted PDF" in response.data
        assert list(tmp_path.iterdir()) == []

    def test_rejects_oversized_request(self, client):
        flask_app.app.config["MAX_CONTENT_LENGTH"] = 1024
        response = client.post(
            "/notemap/analyze",
            data={"note_file": (io.BytesIO(b"x" * 2048), "large.txt")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 413
        assert b"upload is too large" in response.data


class TestQuestionBankUploads:
    def test_extracts_syllabus_and_multiple_papers(self, client, tmp_path):
        response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (
                    io.BytesIO(text_pdf("1. Data structures\n2. Operating systems")),
                    "syllabus.pdf",
                ),
                "pyq_files": [
                    (
                        io.BytesIO(text_pdf("1. Explain linked list traversal in detail.")),
                        "2023.pdf",
                    ),
                    (
                        io.BytesIO(text_pdf("1. Explain process scheduling algorithms.")),
                        "2024.pdf",
                    ),
                ],
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        assert b"Review syllabus topics" in response.data
        assert b"syllabus.pdf" in response.data
        assert b"2023.pdf" in response.data
        assert b"2024.pdf" in response.data
        assert list(tmp_path.iterdir()) == []

    def test_missing_question_papers_is_rejected(self, client):
        response = client.post(
            "/questionbank/analyze",
            data={"syllabus_file": (io.BytesIO(text_pdf()), "syllabus.pdf")},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"at least one previous-year question paper" in response.data

    def test_missing_all_questionbank_inputs_reports_both(self, client):
        response = client.post(
            "/questionbank/analyze",
            data={},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Please choose a syllabus file" in response.data
        assert b"at least one previous-year question paper" in response.data

    def test_missing_syllabus_is_rejected(self, client):
        response = client.post(
            "/questionbank/analyze",
            data={"pyq_files": [(io.BytesIO(text_pdf()), "paper.pdf")]},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Please choose a syllabus file" in response.data

    def test_syllabus_must_be_pdf(self, client):
        response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (io.BytesIO(b"syllabus"), "syllabus.txt"),
                "pyq_files": [(io.BytesIO(text_pdf()), "paper.pdf")],
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Syllabus must be a .pdf file" in response.data

    def test_question_papers_must_be_pdf(self, client):
        response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (io.BytesIO(text_pdf()), "syllabus.pdf"),
                "pyq_files": [(io.BytesIO(b"paper"), "paper.txt")],
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Question paper must be a .pdf file" in response.data

    def test_rejects_empty_question_paper(self, client):
        response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (io.BytesIO(text_pdf()), "syllabus.pdf"),
                "pyq_files": [(io.BytesIO(b""), "paper.pdf")],
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"question paper file is empty" in response.data

    def test_rejects_scanned_question_paper(self, client):
        response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (io.BytesIO(text_pdf()), "syllabus.pdf"),
                "pyq_files": [(io.BytesIO(text_pdf("")), "paper.pdf")],
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"OCR is not available" in response.data

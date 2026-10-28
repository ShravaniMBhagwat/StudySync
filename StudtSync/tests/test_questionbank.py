"""QuestionBank extraction, similarity, trend, and workflow tests."""

import io
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app
from ml.similarity_mapper import (
    compute_coverage_gaps,
    extract_syllabus_topics,
    map_questions_to_topics,
    parse_topic_heading_and_subheadings,
    split_questions,
)
from ml.trend_analyzer import compute_topic_frequency, compute_year_trend


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
    with flask_app._QUESTIONBANK_STATE_LOCK:
        flask_app._QUESTIONBANK_STATE.clear()


def make_pdf(text: str) -> bytes:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def workflow_id_from(response) -> str:
    page = response.get_data(as_text=True)
    match = re.search(r'name="workflow_id" value="([a-f0-9]+)"', page)
    assert match, "workflow token should be in the review form"
    return match.group(1)


class TestQuestionExtraction:
    def test_syllabus_topic_extraction_removes_numbering_and_deduplicates(self):
        topics = extract_syllabus_topics(
            "Unit 1\n1. Relational database design\n2. SQL query processing\n"
            "2. SQL query processing\n"
        )
        assert topics == ["Relational database design", "SQL query processing"]

    def test_question_splitting_groups_continuation_lines(self):
        questions = split_questions(
            [
                {
                    "page_number": 3,
                    "text": (
                        "1. Explain relational keys in a database.\n"
                        "A key identifies a tuple and supports integrity.\n"
                        "2. How does SQL join two relations?\n"
                        "Give an example using a primary key."
                    ),
                }
            ],
            "paper_2023.pdf",
            "2023",
        )
        assert len(questions) == 2
        assert questions[0]["page_number"] == 3
        assert questions[0]["source_file"] == "paper_2023.pdf"
        assert "supports integrity" in questions[0]["text"]
        assert "Give an example" in questions[1]["text"]

    def test_similarity_marks_unknown_vocabulary_for_review(self):
        frame = map_questions_to_topics(
            ["Relational database design"],
            [
                {
                    "text": "quasar zephyr xenolith",
                    "source_file": "notes.pdf",
                    "page_number": 1,
                    "year": "Unknown",
                }
            ],
        )
        assert isinstance(frame, pd.DataFrame)
        assert frame.iloc[0]["matched_topic"] == "Unmatched"
        assert frame.iloc[0]["needs_review"]
        assert frame.iloc[0]["similarity_score"] == 0

    def test_close_topic_scores_are_marked_for_manual_review(self):
        frame = map_questions_to_topics(
            ["database systems", "database systems"],
            [
                {
                    "text": "database systems",
                    "source_file": "paper.pdf",
                    "page_number": 1,
                    "year": "Unknown",
                }
            ],
        )
        assert frame.iloc[0]["matched_topic"] == "database systems"
        assert frame.iloc[0]["needs_review"]
        assert "two best topic scores are close" in frame.iloc[0]["review_reason"]

    def test_pandas_frequency_gaps_and_year_trend(self):
        frame = map_questions_to_topics(
            ["SQL queries", "Network routing"],
            [
                {
                    "text": "SQL queries select data from tables",
                    "source_file": "paper_2023.pdf",
                    "page_number": 1,
                    "year": "2023",
                }
            ],
        )
        frequency = compute_topic_frequency(
            frame,
            ["SQL queries", "Network routing"],
        )
        assert frequency.to_dict("records") == [
            {"topic": "SQL queries", "question_count": 1},
            {"topic": "Network routing", "question_count": 0},
        ]
        assert compute_coverage_gaps(
            ["SQL queries", "Network routing"], frame
        ) == ["Network routing"]
        trend = compute_year_trend(frame)
        assert trend.loc["SQL queries", "2023"] == 1
        assert compute_year_trend(
            frame.assign(year="Unknown")
        ).empty


    def test_conceptual_mapping_identifies_concepts_and_subheadings(self):
        heading, subs = parse_topic_heading_and_subheadings(
            "Unit 1: Stacks and Queues: Push, Pop, Enqueue, Dequeue, Circular Queue"
        )
        assert heading == "Stacks and Queues"
        assert len(subs) >= 3

        frame = map_questions_to_topics(
            ["Stacks and Queues", "Graph Theory"],
            [{"text": "Explain FIFO buffer with insertion and deletion operations in a circular queue."}],
            topic_subheadings={"Stacks and Queues": subs}
        )
        assert frame.iloc[0]["matched_topic"] == "Stacks and Queues"
        assert frame.iloc[0]["similarity_score"] > 0.15


class TestQuestionBankWorkflow:
    def test_analytics_renders_empty_questionbank_state_without_fake_charts(
        self, client, tmp_path
    ):
        with flask_app._QUESTIONBANK_STATE_LOCK:
            flask_app._QUESTIONBANK_STATE.clear()
        previous_dataset_dir = flask_app.DATASET_DIR
        flask_app.DATASET_DIR = str(tmp_path)
        try:
            response = client.get("/analytics")
        finally:
            flask_app.DATASET_DIR = previous_dataset_dir

        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "No QuestionBank analysis is available yet" in body
        assert "NoteMap classification comparison" in body
        assert "Upload papers with at least three extractable candidate questions" in body
        assert 'alt="Bar chart of actual mapped question frequency' not in body
        assert 'alt="KMeans clusters of actual uploaded question text' not in body
        assert "student-mat.csv file is not installed" in body

    def test_full_upload_review_mapping_correction_and_recalculation(
        self, client, tmp_path
    ):
        syllabus_pdf = make_pdf(
            "1. Relational database design\n2. SQL query processing"
        )
        paper_pdf = make_pdf(
            "1. Explain relational keys in a database.\n"
            "A key identifies a tuple and supports integrity.\n"
            "2. How does SQL join two relations?\n"
            "Give an example using a primary key."
        )
        upload_response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (io.BytesIO(syllabus_pdf), "syllabus.pdf"),
                "pyq_files": [(io.BytesIO(paper_pdf), "exam_2023.pdf")],
            },
            content_type="multipart/form-data",
        )
        assert upload_response.status_code == 200
        assert b"Review syllabus topics" in upload_response.data
        assert list(tmp_path.iterdir()) == []

        token = workflow_id_from(upload_response)
        mapped_response = client.post(
            "/questionbank/map",
            data={
                "workflow_id": token,
                "topics": "Relational database design\nSQL query processing",
            },
        )
        assert mapped_response.status_code == 200
        body = mapped_response.get_data(as_text=True)
        assert "Mapping Results" in body
        assert "exam_2023.pdf" in body
        assert "p. 1" in body
        assert "Similarity" in body
        assert "data:image/png;base64," in body
        assert "Year-wise topic trends" in body
        assert "not a probability" in body
        assert body.count('name="mapping"') == 2

        corrected_response = client.post(
            "/questionbank/map",
            data={
                "workflow_id": token,
                "topics": ["Relational database design", "SQL query processing"],
                "mapping": ["SQL query processing", "SQL query processing"],
            },
        )
        assert corrected_response.status_code == 200
        corrected_body = corrected_response.get_data(as_text=True)
        assert "mappings were applied" in corrected_body
        assert "<td>SQL query processing</td><td>2</td>" in corrected_body
        assert "<td>Relational database design</td><td>0</td>" in corrected_body
        mapping_export = client.get(
            f"/analytics/export/question-mappings.csv?workflow_id={token}"
        )
        assert mapping_export.status_code == 200
        assert b"SQL query processing" in mapping_export.data
        frequency_export = client.get(
            f"/analytics/export/topic-frequency.csv?workflow_id={token}"
        )
        assert b"SQL query processing,2" in frequency_export.data

    def test_duplicate_papers_are_ignored_and_missing_year_is_reported(self, client):
        syllabus_pdf = make_pdf(
            "1. Relational database design\n2. SQL query processing"
        )
        paper_pdf = make_pdf("1. Explain relational database design in detail.")
        response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (io.BytesIO(syllabus_pdf), "syllabus.pdf"),
                "pyq_files": [
                    (io.BytesIO(paper_pdf), "exam.pdf"),
                    (io.BytesIO(paper_pdf), "copy_2023.pdf"),
                ],
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        assert b"copy_2023.pdf" in response.data
        token = workflow_id_from(response)
        result = client.post(
            "/questionbank/map",
            data={"workflow_id": token, "topics": "Relational database design\nSQL query processing"},
        )
        assert result.status_code == 200
        assert b"Year trends are unavailable" in result.data

    @pytest.mark.parametrize(
        ("paper_content", "paper_name", "expected"),
        [
            (b"", "empty.pdf", b"file is empty"),
            (b"not a pdf", "broken.pdf", b"valid, unencrypted PDF"),
        ],
    )
    def test_rejects_empty_or_malformed_pdf(
        self, client, paper_content, paper_name, expected
    ):
        response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (
                    io.BytesIO(make_pdf("1. Database systems")),
                    "syllabus.pdf",
                ),
                "pyq_files": [(io.BytesIO(paper_content), paper_name)],
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert expected in response.data

    def test_rejects_scanned_pdf_with_ocr_message(self, client):
        response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (
                    io.BytesIO(make_pdf("1. Database systems")),
                    "syllabus.pdf",
                ),
                "pyq_files": [(io.BytesIO(make_pdf("")), "scan.pdf")],
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"OCR is not available" in response.data

    def test_missing_workflow_state_explains_restart_or_expiry(self, client):
        response = client.post(
            "/questionbank/map",
            data={"workflow_id": "expired-token", "topics": "Database systems"},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"review has expired or the server restarted" in response.data

    def test_analytics_charts_follow_latest_question_upload_and_export_csv(self, client):
        syllabus_pdf = make_pdf(
            "1. Relational database design\n2. Network packet routing"
        )

        def upload_and_map(filename: str, question_text: str) -> str:
            response = client.post(
                "/questionbank/analyze",
                data={
                    "syllabus_file": (
                        io.BytesIO(syllabus_pdf),
                        "course_syllabus.pdf",
                    ),
                    "pyq_files": [
                        (io.BytesIO(make_pdf(question_text)), filename)
                    ],
                },
                content_type="multipart/form-data",
            )
            assert response.status_code == 200
            token = workflow_id_from(response)
            mapped = client.post(
                "/questionbank/map",
                data={
                    "workflow_id": token,
                    "topics": [
                        "Relational database design",
                        "Network packet routing",
                    ],
                },
            )
            assert mapped.status_code == 200
            return token

        first_token = upload_and_map(
            "exam_2022.pdf",
            "1. Explain relational database design and keys in detail.\n"
            "2. Describe normalization and relational schema design.\n"
            "3. Compare primary keys and foreign keys in database tables.",
        )
        first_dashboard = client.get("/analytics")
        first_body = first_dashboard.get_data(as_text=True)
        first_chart = re.search(
            r'<img src="(data:image/png;base64,[^"]+)" alt="Bar chart of actual mapped question frequency',
            first_body,
        )
        assert first_chart
        assert "exam_2022.pdf" in first_body
        assert "held-out test-set" in first_body
        assert "selected model confusion matrix" in first_body.lower()
        first_cluster_chart = re.search(
            r'<img src="(data:image/png;base64,[^"]+)" alt="KMeans clusters of actual uploaded question text',
            first_body,
        )
        assert first_cluster_chart

        mapping_csv = client.get(
            f"/analytics/export/question-mappings.csv?workflow_id={first_token}"
        )
        assert mapping_csv.status_code == 200
        assert mapping_csv.mimetype == "text/csv"
        assert b"question_text,matched_topic,similarity_score" in mapping_csv.data
        assert b"exam_2022.pdf" in mapping_csv.data

        frequency_csv = client.get(
            f"/analytics/export/topic-frequency.csv?workflow_id={first_token}"
        )
        assert frequency_csv.status_code == 200
        assert b"topic,question_count" in frequency_csv.data

        upload_and_map(
            "exam_2023.pdf",
            "1. Explain network packet routing between different networks in detail.\n"
            "2. Describe how routers forward IP packets between networks.\n"
            "3. Compare TCP and UDP transport protocols.",
        )
        second_body = client.get("/analytics").get_data(as_text=True)
        second_chart = re.search(
            r'<img src="(data:image/png;base64,[^"]+)" alt="Bar chart of actual mapped question frequency',
            second_body,
        )
        assert second_chart
        assert second_chart.group(1) != first_chart.group(1)
        assert "exam_2023.pdf" in second_body
        assert "exam_2022.pdf" not in second_body
        second_cluster_chart = re.search(
            r'<img src="(data:image/png;base64,[^"]+)" alt="KMeans clusters of actual uploaded question text',
            second_body,
        )
        assert second_cluster_chart
        assert second_cluster_chart.group(1) != first_cluster_chart.group(1)

    def test_analytics_shows_year_trends_only_for_multiple_reliable_years(self, client):
        syllabus_pdf = make_pdf("1. Relational database design")
        response = client.post(
            "/questionbank/analyze",
            data={
                "syllabus_file": (io.BytesIO(syllabus_pdf), "syllabus.pdf"),
                "pyq_files": [
                    (
                        io.BytesIO(
                            make_pdf("1. Explain relational database design in detail.")
                        ),
                        "exam_2022.pdf",
                    ),
                    (
                        io.BytesIO(
                            make_pdf(
                                "1. Explain relational database design and normalization."
                            )
                        ),
                        "exam_2023.pdf",
                    ),
                ],
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        token = workflow_id_from(response)
        client.post(
            "/questionbank/map",
            data={"workflow_id": token, "topics": "Relational database design"},
        )

        response = client.get("/analytics")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "Year-wise topic trends" in body
        assert 'alt="Line chart of actual mapped question frequencies by year and topic"' in body
        assert "<th>2022</th>" in body and "<th>2023</th>" in body

    def test_csv_export_without_analysis_is_empty_state_404(self, client):
        response = client.get("/analytics/export/topic-frequency.csv")
        assert response.status_code == 404
        assert b"No mapped question analysis is available" in response.data

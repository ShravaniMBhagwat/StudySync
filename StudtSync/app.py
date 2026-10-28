"""Flask application entry point for StudySync.

Run the development server with ``python app.py``.
"""

import logging
import os
import re
import hashlib
import tempfile
import threading
import time
import uuid
from functools import lru_cache

import pandas as pd
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_EXTENSIONS,
    BASE_DIR,
    DEBUG,
    DATASET_DIR,
    MAX_CONTENT_LENGTH,
    SECRET_KEY,
    SUBJECT_LABELS,
    UPLOAD_FOLDER,
)
from ml.classifier import load_demo_dataset, predict as predict_subject, train
from ml.text_extractor import (
    MalformedPDFError,
    ScannedPDFError,
    extract_text,
    extract_pages,
    get_page_count,
)
from ml.similarity_mapper import (
    compute_coverage_gaps,
    extract_syllabus_topic_records,
    map_questions_to_topics,
    parse_topic_heading_and_subheadings,
    split_questions,
)
from ml.trend_analyzer import (
    compute_topic_frequency,
    compute_year_trend,
    plot_classification_comparison,
    plot_confusion_matrix,
    plot_topic_frequency,
    plot_year_trend,
)
from ml.feature_selection import compare_feature_selection
from ml.dimensionality_reduction import reduce_tfidf
from ml.clustering import cluster_documents
from ml.regression import run_experiment as run_regression

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = SECRET_KEY
logger = logging.getLogger(__name__)
_QUESTIONBANK_STATE: dict[str, dict] = {}
_QUESTIONBANK_STATE_LOCK = threading.Lock()
_QUESTIONBANK_STATE_TTL_SECONDS = 30 * 60
_QUESTIONBANK_MAX_STATES = 32
_QUESTIONBANK_MAX_TEXT_BYTES = 32 * 1024 * 1024


class UploadValidationError(ValueError):
    """An uploaded file does not meet the application upload requirements."""


def allowed_file(filename: str, extensions: set[str] | None = None) -> bool:
    allowed_extensions = ALLOWED_EXTENSIONS if extensions is None else extensions
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def _validate_upload(upload: FileStorage | None, label: str, extensions: set[str]) -> str:
    if upload is None or not upload.filename:
        raise UploadValidationError(f"Please choose a {label} file.")

    original_name = secure_filename(upload.filename)
    if not original_name:
        raise UploadValidationError(f"The {label} filename is invalid.")

    extension = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
    if extension not in extensions:
        allowed = ", ".join(sorted(f".{item}" for item in extensions))
        raise UploadValidationError(f"{label.capitalize()} must be a {allowed} file.")

    upload.stream.seek(0, os.SEEK_END)
    file_size = upload.stream.tell()
    upload.stream.seek(0)
    if file_size == 0:
        raise UploadValidationError(f"The {label} file is empty.")
    if file_size > app.config["MAX_CONTENT_LENGTH"]:
        raise UploadValidationError(
            f"The {label} file exceeds the {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB size limit."
        )

    return extension


def _extract_upload(
    upload: FileStorage,
    label: str,
    extensions: set[str],
    temp_dir: str,
    include_text: bool = False,
    include_pages: bool = False,
) -> dict[str, object]:
    extension = _validate_upload(upload, label, extensions)
    safe_name = secure_filename(upload.filename or "")
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.{extension}")
    upload.save(temp_path)

    if os.path.getsize(temp_path) == 0:
        raise UploadValidationError(f"The {label} file is empty.")

    extracted_pages = None
    try:
        if include_pages and extension == "pdf":
            extracted_pages = extract_pages(temp_path)
            text = "\n".join(page["text"] for page in extracted_pages)
        else:
            text = extract_text(temp_path)
    except ScannedPDFError:
        raise UploadValidationError(
            f"The {label} PDF contains no extractable text. Scanned PDFs are not supported yet; OCR is not available."
        ) from None
    except MalformedPDFError:
        raise UploadValidationError(
            f"The {label} PDF could not be read. Please upload a valid, unencrypted PDF."
        ) from None

    if not text.strip():
        raise UploadValidationError(f"The {label} file contains no text.")

    result: dict[str, object] = {
        "filename": safe_name,
        "characters": len(text),
        "pages": get_page_count(temp_path) if extension == "pdf" else None,
    }
    if include_text:
        result["text"] = text
    if include_pages:
        result["extracted_pages"] = extracted_pages or []
    return result


@lru_cache(maxsize=1)
def _get_notemap_model() -> dict:
    dataset_path = os.path.join(BASE_DIR, "data", "training", "demo_notes.csv")
    texts, labels = load_demo_dataset(dataset_path)
    return train(texts, labels)


def _year_from_filename(filename: str) -> str:
    match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", filename)
    return match.group(0) if match else "Unknown"


def _extract_question_paper(
    upload: FileStorage,
    temp_dir: str,
) -> tuple[dict, str]:
    extension = _validate_upload(upload, "question paper", {"pdf"})
    safe_name = secure_filename(upload.filename or "")
    content = upload.read()
    if not content:
        raise UploadValidationError("The question paper file is empty.")
    digest = hashlib.sha256(content).hexdigest()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.{extension}")
    with open(temp_path, "wb") as saved_upload:
        saved_upload.write(content)
    try:
        pages = extract_pages(temp_path)
    except ScannedPDFError:
        raise UploadValidationError(
            f"{safe_name} contains no extractable text. Scanned PDFs are not supported yet; OCR is not available."
        ) from None
    except MalformedPDFError:
        raise UploadValidationError(
            f"{safe_name} could not be read. Please upload a valid, unencrypted PDF."
        ) from None

    text = "\n".join(str(page["text"]) for page in pages)
    if not text.strip():
        raise UploadValidationError(f"{safe_name} contains no extractable text.")
    return (
        {
            "filename": safe_name,
            "characters": len(text),
            "pages": pages,
            "year": _year_from_filename(safe_name),
        },
        digest,
    )


def _store_questionbank_state(state: dict) -> str:
    now = time.monotonic()
    state_bytes = sum(
        len(str(question.get("text", "")).encode("utf-8"))
        for question in state["questions"]
    )
    state_bytes += sum(
        len(str(value).encode("utf-8"))
        for value in state["topics"]
    )
    state_bytes += sum(
        len(str(source.get("topic", "")).encode("utf-8"))
        + len(str(source.get("source_file", "")).encode("utf-8"))
        for source in state["topic_sources"]
    )
    if state_bytes > _QUESTIONBANK_MAX_TEXT_BYTES:
        raise UploadValidationError(
            "The extracted question text is too large to keep for topic review. Upload smaller or fewer papers."
        )
    with _QUESTIONBANK_STATE_LOCK:
        expired = [
            key
            for key, entry in _QUESTIONBANK_STATE.items()
            if now - entry["created_at"] > _QUESTIONBANK_STATE_TTL_SECONDS
        ]
        for key in expired:
            _QUESTIONBANK_STATE.pop(key, None)
        while _QUESTIONBANK_STATE and (
            len(_QUESTIONBANK_STATE) >= _QUESTIONBANK_MAX_STATES
            or sum(entry["state_bytes"] for entry in _QUESTIONBANK_STATE.values())
            + state_bytes
            > _QUESTIONBANK_MAX_TEXT_BYTES
        ):
            oldest = min(
                _QUESTIONBANK_STATE,
                key=lambda key: _QUESTIONBANK_STATE[key]["created_at"],
            )
            _QUESTIONBANK_STATE.pop(oldest, None)
        token = uuid.uuid4().hex
        _QUESTIONBANK_STATE[token] = {
            **state,
            "created_at": now,
            "state_bytes": state_bytes,
        }
        return token


def _get_questionbank_state(token: str) -> dict | None:
    now = time.monotonic()
    with _QUESTIONBANK_STATE_LOCK:
        state = _QUESTIONBANK_STATE.get(token)
        if state is None:
            return None
        if now - state["created_at"] > _QUESTIONBANK_STATE_TTL_SECONDS:
            _QUESTIONBANK_STATE.pop(token, None)
            return None
        return state


def _latest_questionbank_state() -> tuple[str | None, dict | None]:
    now = time.monotonic()
    with _QUESTIONBANK_STATE_LOCK:
        expired = [
            key
            for key, entry in _QUESTIONBANK_STATE.items()
            if now - entry["created_at"] > _QUESTIONBANK_STATE_TTL_SECONDS
        ]
        for key in expired:
            _QUESTIONBANK_STATE.pop(key, None)
        if not _QUESTIONBANK_STATE:
            return None, None
        token = max(
            _QUESTIONBANK_STATE,
            key=lambda key: _QUESTIONBANK_STATE[key]["created_at"],
        )
        return token, _QUESTIONBANK_STATE[token]


def _render_questionbank_results(token: str, state: dict, topics: list[str]):
    questions = state["questions"]
    topic_subheadings = state.get("topic_subheadings", {})
    topic_contexts = state.get("topic_contexts", {})
    try:
        mapped_df = map_questions_to_topics(
            topics,
            questions,
            topic_subheadings=topic_subheadings,
            topic_contexts=topic_contexts,
        )
    except ValueError as exc:
        return _upload_error(str(exc), "questionbank")
    corrections = request.form.getlist("mapping")
    question_ids = [str(question["question_id"]) for question in questions]
    corrected = False
    if corrections:
        if len(corrections) != len(question_ids):
            return _upload_error(
                "The mapping review was incomplete. Please review every question.",
                "questionbank",
            )
        valid_topics = set(topics) | {"Unmatched"}
        if any(value not in valid_topics for value in corrections):
            return _upload_error(
                "A mapping selected an unknown syllabus topic. Please review the mappings again.",
                "questionbank",
            )
        mapped_df["matched_topic"] = corrections
        mapped_df["needs_review"] = [
            value == "Unmatched" for value in corrections
        ]
        mapped_df["review_reason"] = ""
        corrected = True

    frequency_df = compute_topic_frequency(mapped_df, topics)
    trend_df = compute_year_trend(mapped_df)
    gaps = compute_coverage_gaps(topics, mapped_df)
    rows = mapped_df.to_dict(orient="records")
    state["topics"] = topics
    state["mapped_rows"] = rows
    state["last_analysis_at"] = time.time()
    state["corrections_applied"] = corrected or state.get("corrections_applied", False)
    years = [str(year) for year in trend_df.columns] if not trend_df.empty else []
    trend_rows = (
        [
            {
                "topic": str(topic),
                "counts": [int(value) for value in trend_df.loc[topic].tolist()],
            }
            for topic in trend_df.index
        ]
        if years
        else []
    )

    topic_question_map = {str(topic): [] for topic in topics}
    unmatched_questions = []
    for q in rows:
        m_topic = str(q.get("matched_topic", ""))
        if m_topic in topic_question_map:
            topic_question_map[m_topic].append(q)
        else:
            unmatched_questions.append(q)

    return render_template(
        "questionbank/result.html",
        workflow_id=token,
        topics=topics,
        papers=state["papers"],
        topic_sources=state["topic_sources"],
        duplicate_files=state["duplicate_files"],
        questions=rows,
        frequency=frequency_df.to_dict(orient="records"),
        trend_years=years,
        trend_rows=trend_rows,
        topic_chart=plot_topic_frequency(frequency_df),
        year_chart=plot_year_trend(trend_df),
        gaps=gaps,
        corrected=corrected,
        threshold=0.15,
        question_count=len(rows),
        topic_question_map=topic_question_map,
        unmatched_questions=unmatched_questions,
        topic_subheadings=topic_subheadings,
    )


def _upload_error(message: str, endpoint: str):
    flash(message, "danger")
    return redirect(url_for(endpoint))


@app.errorhandler(413)
def request_too_large(error):
    logger.warning(
        "Rejected oversized upload request on %s (%s)",
        request.path,
        type(error).__name__,
    )
    return (
        "The upload is too large. The maximum request size is "
        f"{app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB.",
        413,
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# Dashboard and main pages
@app.route("/")
@app.route("/dashboard")
def index():
    return render_template("index.html")


@app.route("/notemap")
def notemap():
    return render_template("notemap/upload.html")


@app.route("/notemap/analyze", methods=["POST"])
def notemap_analyze():
    note_files = [
        f for f in (request.files.getlist("note_file") + request.files.getlist("note_files"))
        if f and f.filename
    ]
    if not note_files:
        return _upload_error("Please choose a note file.", "notemap")

    try:
        with tempfile.TemporaryDirectory(dir=app.config["UPLOAD_FOLDER"]) as temp_dir:
            if len(note_files) == 1:
                details = _extract_upload(
                    note_files[0],
                    "note",
                    {"pdf", "txt"},
                    temp_dir,
                    include_text=True,
                )
                extracted_text = str(details.pop("text"))
            else:
                combined_texts = []
                total_chars = 0
                total_pages = 0
                file_names = []
                for nf in note_files:
                    d = _extract_upload(
                        nf,
                        "note",
                        {"pdf", "txt"},
                        temp_dir,
                        include_text=True,
                    )
                    combined_texts.append(str(d.pop("text")))
                    total_chars += int(d.get("characters", 0))
                    total_pages += int(d.get("pages") or 1)
                    file_names.append(str(d.get("filename", "")))
                details = {
                    "filename": f"{len(file_names)} files ({', '.join(file_names)})",
                    "characters": total_chars,
                    "pages": total_pages,
                }
                extracted_text = "\n\n".join(combined_texts)
    except UploadValidationError as exc:
        logger.info("Rejected NoteMap upload: %s", exc)
        return _upload_error(str(exc), "notemap")

    model = _get_notemap_model()
    prediction = predict_subject(extracted_text, model)
    selected_metrics = model["comparisons"][model["selected_key"]]
    result = {
        **details,
        **prediction,
        "prediction": prediction["label"],
        "model_name": model["selected_name"],
        "model_metrics": selected_metrics,
        "is_demo_data": model["is_demo_data"],
        "message": (
            f"Text extraction completed ({details['characters']} characters). "
            "This prediction is based on a small demonstration dataset, not a model "
            "validated for all student notes."
        ),
    }
    return render_template("notemap/result.html", result=result, subjects=SUBJECT_LABELS)


@app.route("/notemap/correct", methods=["POST"])
def notemap_correct():
    predicted = request.form.get("predicted_subject", "")
    corrected = request.form.get("corrected_subject", "")
    allowed_predictions = set(SUBJECT_LABELS) | {
        "Unknown / outside the demonstration vocabulary",
        "Unknown / insufficient text",
    }
    if predicted not in allowed_predictions or corrected not in SUBJECT_LABELS:
        flash("Choose a subject from the available list to submit a correction.", "danger")
        return redirect(url_for("notemap"))
    return render_template(
        "notemap/correction.html",
        predicted=predicted,
        corrected=corrected,
    )


@app.route("/questionbank")
def questionbank():
    return render_template("questionbank/upload.html")


@app.route("/questionbank/analyze", methods=["POST"])
def questionbank_analyze():
    syllabus_files = [
        f for f in (request.files.getlist("syllabus_file") + request.files.getlist("syllabus_files"))
        if f and f.filename
    ]
    pyqs = [upload for upload in request.files.getlist("pyq_files") if upload and upload.filename]

    missing_inputs = []
    if not syllabus_files:
        missing_inputs.append("Please choose a syllabus file.")
    if not pyqs:
        missing_inputs.append(
            "Please upload at least one previous-year question paper (PDF)."
        )
    if missing_inputs:
        return _upload_error(
            " ".join(missing_inputs),
            "questionbank",
        )

    try:
        with tempfile.TemporaryDirectory(dir=app.config["UPLOAD_FOLDER"]) as temp_dir:
            all_topics = []
            all_topic_records = []
            combined_filenames = []
            total_characters = 0
            total_pages = 0
            seen_topic_keys = set()
            for syllabus in syllabus_files:
                ext = _validate_upload(syllabus, "syllabus", {"pdf"})
                safe_name = secure_filename(syllabus.filename or "")
                temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.{ext}")
                syllabus.save(temp_path)

                if ext == "pdf":
                    try:
                        pages = extract_pages(temp_path)
                        text = "\n".join(p["text"] for p in pages)
                        page_count = get_page_count(temp_path)
                    except ScannedPDFError:
                        raise UploadValidationError(
                            "The syllabus PDF contains no extractable text. Scanned PDFs are not supported yet; OCR is not available."
                        )
                    except MalformedPDFError:
                        raise UploadValidationError(
                            "The syllabus PDF could not be read. Please upload a valid, unencrypted PDF."
                        )
                else:
                    text = extract_text(temp_path)
                    pages = [{"page_number": 1, "text": text}]
                    page_count = 1

                if not text.strip():
                    msg = "The syllabus file is empty." if len(syllabus_files) == 1 else f"The syllabus file '{safe_name}' contains no text."
                    raise UploadValidationError(msg)

                combined_filenames.append(safe_name)
                total_characters += len(text)
                total_pages += (page_count or 1)

                records = extract_syllabus_topic_records(pages, safe_name)
                for rec in records:
                    key = rec["topic"].casefold()
                    if key not in seen_topic_keys:
                        seen_topic_keys.add(key)
                        all_topic_records.append(rec)
                        all_topics.append(rec["topic"])

            if not all_topics:
                raise UploadValidationError(
                    "No syllabus topics could be extracted. Please upload a text-based PDF with topic headings."
                )

            syllabus_result = {
                "filename": combined_filenames[0] if len(combined_filenames) == 1 else f"{len(combined_filenames)} files ({', '.join(combined_filenames)})",
                "characters": total_characters,
                "pages": total_pages,
            }
            topic_records = all_topic_records
            topics = all_topics
            topic_subheadings_map = {}
            topic_context_map = {}
            for rec in all_topic_records:
                t = rec["topic"]
                if rec.get("subtopics"):
                    topic_subheadings_map[t] = rec["subtopics"]
                if rec.get("context_text"):
                    topic_context_map[t] = rec["context_text"]

            paper_results = []
            questions = []
            duplicate_files = []
            seen_digests: set[str] = set()
            for upload in pyqs:
                paper, digest = _extract_question_paper(upload, temp_dir)
                if digest in seen_digests:
                    duplicate_files.append(paper["filename"])
                    continue
                seen_digests.add(digest)
                paper_results.append(
                    {
                        "filename": paper["filename"],
                        "characters": paper["characters"],
                        "pages": len(paper["pages"]),
                        "year": paper["year"],
                    }
                )
                candidates = split_questions(
                    paper["pages"],
                    paper["filename"],
                    paper["year"],
                )
                for candidate in candidates:
                    candidate["question_id"] = uuid.uuid4().hex
                    questions.append(candidate)
    except UploadValidationError as exc:
        logger.info("Rejected QuestionBank upload: %s", exc)
        return _upload_error(str(exc), "questionbank")

    if not paper_results:
        return _upload_error(
            "All uploaded question papers were duplicates. Upload at least one unique paper.",
            "questionbank",
        )

    try:
        workflow_id = _store_questionbank_state(
            {
                "topics": topics,
                "topic_subheadings": topic_subheadings_map,
                "topic_contexts": topic_context_map,
                "questions": questions,
                "papers": paper_results,
                "duplicate_files": duplicate_files,
                "syllabus": syllabus_result,
                "topic_sources": topic_records,
            }
        )
    except UploadValidationError as exc:
        return _upload_error(str(exc), "questionbank")
    return render_template(
        "questionbank/review_topics.html",
        workflow_id=workflow_id,
        topics=topics,
        topic_subheadings=topic_subheadings_map,
        syllabus=syllabus_result,
        topic_sources=topic_records,
        papers=paper_results,
        question_count=len(questions),
        duplicate_files=duplicate_files,
    )


@app.route("/questionbank/map", methods=["POST"])
def questionbank_map():
    token = request.form.get("workflow_id", "")
    state = _get_questionbank_state(token)
    if state is None:
        return _upload_error(
            "This review has expired or the server restarted. Please upload the documents again.",
            "questionbank",
        )

    if request.form.getlist("topics"):
        submitted_topics = request.form.getlist("topics")
        topic_values = (
            submitted_topics
            if len(submitted_topics) > 1
            else submitted_topics[0].splitlines()
        )
        topics = [
            re.sub(r"\s+", " ", topic).strip()
            for topic in topic_values
            if topic.strip()
        ]
        if not topics:
            return _upload_error(
                "Enter at least one syllabus topic before mapping questions.",
                "questionbank",
            )
        # Gracefully handle large topic lists and auto-truncate excessively long lines
        clean_topics = []
        topic_subheadings = state.get("topic_subheadings", {})
        for topic in topics:
            clean_t = topic[:500].strip() if len(topic) > 500 else topic
            if clean_t:
                heading, subs = parse_topic_heading_and_subheadings(clean_t)
                heading = heading if heading else clean_t
                if subs and heading not in topic_subheadings:
                    topic_subheadings[heading] = subs
                clean_topics.append(heading)
        state["topic_subheadings"] = topic_subheadings
        topics = clean_topics[:1000]
        unique_topics = []
        seen_topics = set()
        for topic in topics:
            if topic.casefold() not in seen_topics:
                unique_topics.append(topic)
                seen_topics.add(topic.casefold())
        state["topics"] = unique_topics

    return _render_questionbank_results(token, state, state["topics"])


@app.route("/analytics")
def analytics():
    workflow_id, state = _latest_questionbank_state()
    has_questionbank = state is not None
    mapped_df = pd.DataFrame(state.get("mapped_rows", [])) if state else pd.DataFrame()
    topics = state.get("topics", []) if state else []

    question_data = {
        "available": has_questionbank,
        "workflow_id": workflow_id,
        "questions": mapped_df.to_dict(orient="records") if not mapped_df.empty else [],
        "frequency": [],
        "topic_chart": None,
        "year_chart": None,
        "trend_years": [],
        "trend_rows": [],
        "year_trends_available": False,
        "year_trends_reason": "Upload and map papers from at least two distinct years to see a trend.",
        "paper_count": len(state.get("papers", [])) if state else 0,
    }

    if state:
        if mapped_df.empty:
            mapped_df = map_questions_to_topics(topics, state["questions"])
            state["mapped_rows"] = mapped_df.to_dict(orient="records")
        frequency_df = compute_topic_frequency(mapped_df, topics)
        question_data["frequency"] = frequency_df.to_dict(orient="records")
        if not frequency_df.empty:
            question_data["topic_chart"] = plot_topic_frequency(frequency_df)

        papers = state.get("papers", [])
        years = [paper.get("year", "Unknown") for paper in papers]
        reliable_years = (
            len(papers) >= 2
            and all(year != "Unknown" for year in years)
            and len(set(years)) >= 2
        )
        if reliable_years and not mapped_df.empty:
            trend_df = compute_year_trend(mapped_df)
            if not trend_df.empty:
                question_data["year_trends_available"] = True
                question_data["trend_years"] = [str(year) for year in trend_df.columns]
                question_data["trend_rows"] = [
                    {
                        "topic": str(topic),
                        "counts": [int(count) for count in trend_df.loc[topic].tolist()],
                    }
                    for topic in trend_df.index
                ]
                question_data["year_chart"] = plot_year_trend(trend_df)
        elif papers:
            question_data["year_trends_reason"] = (
                "Year trends require every unique paper filename to contain a four-digit year "
                "and at least two distinct years. Trends use the year inferred from filenames."
            )

    classification = {"available": False}
    try:
        model = _get_notemap_model()
        selected = model["comparisons"][model["selected_key"]]
        classification = {
            "available": True,
            "is_demo_data": model["is_demo_data"],
            "selected_name": model["selected_name"],
            "classes": model["classes"],
            "comparisons": model["comparisons"],
            "selected_metrics": selected,
            "comparison_chart": plot_classification_comparison(
                model["comparisons"],
                ["mnb", "svm"],
            ),
            "confusion_chart": plot_confusion_matrix(
                selected["confusion_matrix"],
                model["classes"],
            ),
        }
    except (OSError, ValueError) as exc:
        logger.exception("Could not build NoteMap analytics")
        classification["reason"] = f"Classification metrics are unavailable: {exc}"

    clustering = {"available": False}
    if state and state.get("questions"):
        texts = [str(question.get("text", "")) for question in state["questions"]]
        cluster_count = min(4, len(texts) - 1)
        clustering = cluster_documents(texts, n_clusters=cluster_count)
    if not clustering.get("available") and has_questionbank:
        clustering["reason"] = clustering.get(
            "reason",
            "There are no non-empty candidate questions to cluster.",
        )

    try:
        regression = run_regression(DATASET_DIR, random_state=42)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.exception("Could not build regression analytics")
        regression = {
            "available": False,
            "reason": f"Regression results are unavailable because the dataset could not be evaluated ({type(exc).__name__}).",
        }

    return render_template(
        "dsbd/dashboard.html",
        question_data=question_data,
        classification=classification,
        clustering=clustering,
        regression=regression,
    )


@app.route("/analytics/export/<report>.csv")
def export_analytics_csv(report: str):
    workflow_id = request.args.get("workflow_id")
    if workflow_id:
        state = _get_questionbank_state(workflow_id)
        if state is None:
            return "The selected question analysis has expired. Upload and map papers again.", 404
    else:
        workflow_id, state = _latest_questionbank_state()
        if state is None:
            return "No mapped question analysis is available to export.", 404

    mapped_df = pd.DataFrame(state.get("mapped_rows", []))
    if mapped_df.empty:
        mapped_df = map_questions_to_topics(state["topics"], state["questions"])
    if report == "question-mappings":
        export_df = mapped_df[
            [
                "question_text",
                "matched_topic",
                "similarity_score",
                "needs_review",
                "source_file",
                "page_number",
                "year",
            ]
        ]
    elif report == "topic-frequency":
        export_df = compute_topic_frequency(mapped_df, state["topics"])
    else:
        return "Unknown analytics export. Use question-mappings or topic-frequency.", 404

    response = Response(export_df.to_csv(index=False), mimetype="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="{report}.csv"'
    return response


def _model_lab_results(custom_text: str = "", requested_clusters: int = 4) -> dict:
    dataset_path = os.path.join(BASE_DIR, "data", "training", "demo_notes.csv")
    demo_texts, demo_labels = load_demo_dataset(dataset_path)
    sample_texts = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", custom_text)
        if paragraph.strip()
    ]
    documents = sample_texts or demo_texts
    feature_selection = compare_feature_selection(demo_texts, demo_labels)
    dimensionality = reduce_tfidf(documents, random_state=42)
    clustering = cluster_documents(
        documents,
        n_clusters=requested_clusters,
        random_state=42,
    )
    regression = run_regression(DATASET_DIR, random_state=42)
    return {
        "feature_selection": feature_selection,
        "dimensionality": dimensionality,
        "clustering": clustering,
        "regression": regression,
        "document_count": len(documents),
        "using_custom_text": bool(sample_texts),
        "custom_text": custom_text,
        "requested_clusters": requested_clusters,
    }


@app.route("/model-lab", methods=["GET", "POST"])
def model_lab():
    custom_text = request.form.get("documents", "") if request.method == "POST" else ""
    if len(custom_text) > 50000:
        flash("Keep the optional Model Lab text under 50,000 characters.", "danger")
        return redirect(url_for("model_lab"))
    try:
        requested_clusters = int(
            request.form.get("n_clusters", "4") if request.method == "POST" else request.args.get("n_clusters", "4")
        )
    except ValueError:
        flash("Cluster count must be a whole number.", "danger")
        return redirect(url_for("model_lab"))
    if not 2 <= requested_clusters <= 10:
        flash("Choose between 2 and 10 clusters.", "danger")
        return redirect(url_for("model_lab"))
    results = _model_lab_results(custom_text, requested_clusters)
    return render_template("model_lab.html", **results)


@app.route("/dsbd")
def dsbd():
    return redirect(url_for("analytics"))


@app.route("/about")
def about():
    return render_template("about.html")


def _source_code_catalog() -> dict[str, str]:
    """Return a whitelist of project source files available to the source browser."""
    roots = ("ml", "templates", "static", "tests")
    extensions = {".py", ".html", ".css", ".js"}
    relative_paths = ["app.py", "config.py"]
    for root in roots:
        root_path = os.path.join(BASE_DIR, root)
        for current_dir, directories, filenames in os.walk(root_path, followlinks=False):
            directories[:] = sorted(
                directory
                for directory in directories
                if not os.path.islink(os.path.join(current_dir, directory))
            )
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in extensions:
                    relative_paths.append(
                        os.path.relpath(
                            os.path.join(current_dir, filename),
                            BASE_DIR,
                        )
                    )

    return {
        relative_path.replace(os.sep, "/"): os.path.join(BASE_DIR, relative_path)
        for relative_path in sorted(set(relative_paths))
    }


@app.route("/source-code")
def source_code():
    files = _source_code_catalog()
    raw_file = request.args.get("file")
    selected_path = raw_file if raw_file is not None else ("app.py" if "app.py" in files else "")
    selected_file = None
    source_text = None
    file_stats = None
    if selected_path:
        source_path = files.get(selected_path)
        if source_path is None:
            return "Source file not found.", 404
        try:
            with open(source_path, encoding="utf-8") as source_file:
                source_text = source_file.read()
            size_bytes = os.path.getsize(source_path)
            size_formatted = f"{size_bytes / 1024:.1f} KB" if size_bytes >= 1024 else f"{size_bytes} B"
            file_stats = {
                "lines": len(source_text.splitlines()),
                "size_formatted": size_formatted,
                "ext": os.path.splitext(selected_path)[1].lstrip(".").upper(),
            }
        except (OSError, UnicodeError):
            logger.exception("Could not read source file %s", selected_path)
            return "This source file is currently unavailable.", 404
        selected_file = selected_path

    return render_template(
        "source_code.html",
        source_files=list(files),
        selected_file=selected_file,
        source_text=source_text,
        file_stats=file_stats,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(debug=DEBUG, host=host, port=port)

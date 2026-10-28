"""TF-IDF subject classification for NoteMap.

The fitted pipeline stays in memory. Model selection uses cross-validation on
the training partition, while the held-out test partition is used only for the
reported final metrics.
"""

import csv
import logging
import math
from collections import Counter
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import SUBJECT_LABELS, TEST_SPLIT_SIZE, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

logger = logging.getLogger(__name__)
RANDOM_SEED = 42
CLASSIFIER_NAMES = {
    "mnb": "Multinomial Naive Bayes",
    "svm": "Linear SVM",
}


def load_demo_dataset(dataset_path: str) -> tuple[list[str], list[str]]:
    """Load labeled demonstration notes from the project's CSV dataset."""
    texts: list[str] = []
    labels: list[str] = []
    with open(dataset_path, newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if reader.fieldnames is None or not {"subject", "text"}.issubset(reader.fieldnames):
            raise ValueError("The NoteMap dataset must have subject and text columns.")
        for row_number, row in enumerate(reader, start=2):
            subject = (row["subject"] or "").strip()
            text = (row["text"] or "").strip()
            if not subject or not text:
                raise ValueError(f"The NoteMap dataset has an empty value on row {row_number}.")
            if subject not in SUBJECT_LABELS:
                raise ValueError(
                    f"Unknown subject '{subject}' in NoteMap dataset row {row_number}."
                )
            labels.append(subject)
            texts.append(text)
    return texts, labels


def _make_pipeline(classifier_type: str, max_features: int, ngram_range: tuple[int, int]) -> Pipeline:
    if classifier_type == "mnb":
        classifier = MultinomialNB()
    elif classifier_type == "svm":
        classifier = LinearSVC(random_state=RANDOM_SEED, max_iter=5000)
    else:
        raise ValueError("classifier_type must be 'mnb' or 'svm'.")

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    max_features=max_features,
                    ngram_range=ngram_range,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def train(
    texts: list[str],
    labels: list[str],
    model_dir: str | None = None,
    classifier_type: str | None = None,
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    test_size: float = TEST_SPLIT_SIZE,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Compare both models and return the selected fitted pipeline and metrics.

    ``model_dir`` remains an optional compatibility argument but is not used:
    loading serialized model files can execute code, so NoteMap does not persist
    or deserialize model artifacts.
    """
    del model_dir
    if len(texts) != len(labels) or not texts:
        raise ValueError("Provide the same non-zero number of texts and labels.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Training texts must be non-empty strings.")
    unknown_labels = set(labels) - set(SUBJECT_LABELS)
    if unknown_labels:
        raise ValueError(f"Unknown subject labels: {', '.join(sorted(unknown_labels))}.")

    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("Training requires at least two subject categories.")
    if min(counts.values()) < 2:
        raise ValueError("Each subject needs at least two examples for a stratified test split.")

    indices = list(range(len(texts)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    train_texts = [texts[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    test_texts = [texts[index] for index in test_indices]
    test_labels = [labels[index] for index in test_indices]
    classes = sorted(counts)

    min_train_class_count = min(Counter(train_labels).values())
    fold_count = min(cv_folds, min_train_class_count)
    use_cross_validation = fold_count >= 2
    if not use_cross_validation:
        logger.warning(
            "NoteMap training set is too small for cross-validation; selecting Naive Bayes by default."
        )

    comparisons: dict[str, dict[str, Any]] = {}
    for model_key in ("mnb", "svm"):
        pipeline = _make_pipeline(model_key, max_features, ngram_range)
        cv_scores: list[float] = []
        if use_cross_validation:
            folds = StratifiedKFold(
                n_splits=fold_count,
                shuffle=True,
                random_state=RANDOM_SEED,
            )
            cv_scores = cross_val_score(
                pipeline,
                train_texts,
                train_labels,
                cv=folds,
                scoring="f1_macro",
            ).tolist()

        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(test_texts)
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            test_labels,
            predictions,
            labels=classes,
            average="macro",
            zero_division=0,
        )
        cv_mean = sum(cv_scores) / len(cv_scores) if cv_scores else None
        cv_std = (
            math.sqrt(sum((score - cv_mean) ** 2 for score in cv_scores) / len(cv_scores))
            if cv_mean is not None
            else None
        )
        comparisons[model_key] = {
            "name": CLASSIFIER_NAMES[model_key],
            "pipeline": pipeline,
            "accuracy": float(accuracy_score(test_labels, predictions)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "macro_f1": float(macro_f1),
            "confusion_matrix": confusion_matrix(
                test_labels, predictions, labels=classes
            ).tolist(),
            "cv_scores": cv_scores,
            "cv_macro_f1_mean": float(cv_mean) if cv_mean is not None else None,
            "cv_macro_f1_std": float(cv_std) if cv_std is not None else None,
        }

    selected_key = classifier_type
    if selected_key is not None and selected_key not in comparisons:
        raise ValueError("classifier_type must be 'mnb', 'svm', or None.")
    if selected_key is None:
        selected_key = (
            max(
                ("mnb", "svm"),
                key=lambda key: (
                    comparisons[key]["cv_macro_f1_mean"]
                    if comparisons[key]["cv_macro_f1_mean"] is not None
                    else -1.0,
                    key == "mnb",
                ),
            )
            if use_cross_validation
            else "mnb"
        )

    return {
        "selected_key": selected_key,
        "selected_name": CLASSIFIER_NAMES[selected_key],
        "pipeline": comparisons[selected_key]["pipeline"],
        "classes": classes,
        "comparisons": comparisons,
        "test_size": len(test_indices) / len(texts),
        "train_size": len(train_indices),
        "test_count": len(test_indices),
        "cv_folds": fold_count if use_cross_validation else 0,
        "random_seed": RANDOM_SEED,
        "is_demo_data": True,
    }


def predict(text: str, trained_model: dict[str, Any]) -> dict[str, Any]:
    """Predict a subject and return ranked, explicitly non-probability scores."""
    if not isinstance(text, str) or not text.strip():
        return {
            "label": "Unknown / insufficient text",
            "scores": [],
            "score_type": None,
            "needs_review": True,
            "reason": "No readable text was available for classification.",
            "short_text": True,
        }

    pipeline: Pipeline = trained_model["pipeline"]
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    vector = vectorizer.transform([text])
    word_count = len(vectorizer.build_analyzer()(text))
    if vector.nnz == 0:
        return {
            "label": "Unknown / outside the demonstration vocabulary",
            "scores": [],
            "score_type": None,
            "needs_review": True,
            "reason": "The note has no terms in the demonstration training vocabulary.",
            "short_text": word_count < 8,
        }

    classifier = pipeline.named_steps["classifier"]
    raw_scores = pipeline.decision_function([text]) if trained_model["selected_key"] == "svm" else classifier.predict_log_proba(vector)[0]
    ranked_scores = sorted(
        (
            {"label": label, "score": float(score)}
            for label, score in zip(trained_model["classes"], raw_scores)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    margin = ranked_scores[0]["score"] - ranked_scores[1]["score"]
    short_text = word_count < 8
    close_scores = margin < (0.15 if trained_model["selected_key"] == "svm" else 0.5)
    reasons = []
    if short_text:
        reasons.append("The note is very short.")
    if close_scores:
        reasons.append("The top subject scores are close.")

    return {
        "label": ranked_scores[0]["label"],
        "scores": ranked_scores,
        "score_type": (
            "Linear SVM decision margin"
            if trained_model["selected_key"] == "svm"
            else "Naive Bayes log score"
        ),
        "needs_review": bool(reasons),
        "reason": " ".join(reasons) if reasons else "No simple uncertainty warning was triggered.",
        "short_text": short_text,
    }

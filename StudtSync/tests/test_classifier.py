"""Tests for the NoteMap demonstration classifier."""

import os
import sys
from collections import Counter

import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR
from ml.classifier import load_demo_dataset, predict, train


@pytest.fixture(scope="module")
def trained_model():
    dataset_path = os.path.join(BASE_DIR, "data", "training", "demo_notes.csv")
    texts, labels = load_demo_dataset(dataset_path)
    return texts, labels, train(texts, labels)


def test_demo_dataset_has_ten_examples_per_subject(trained_model):
    texts, labels, _ = trained_model
    assert len(texts) == 60
    assert set(Counter(labels).values()) == {10}


def test_comparison_has_reproducible_holdout_metrics_and_cv(trained_model):
    texts, _, result = trained_model
    assert result["random_seed"] == 42
    assert result["train_size"] + result["test_count"] == len(texts)
    assert result["cv_folds"] == 5
    assert set(result["comparisons"]) == {"mnb", "svm"}
    for model_metrics in result["comparisons"].values():
        assert 0 <= model_metrics["accuracy"] <= 1
        assert 0 <= model_metrics["precision_macro"] <= 1
        assert 0 <= model_metrics["recall_macro"] <= 1
        assert 0 <= model_metrics["macro_f1"] <= 1
        assert len(model_metrics["confusion_matrix"]) == len(result["classes"])
        assert sum(map(sum, model_metrics["confusion_matrix"])) == result["test_count"]
        assert len(model_metrics["cv_scores"]) == result["cv_folds"]


def test_predict_returns_non_probability_ranked_scores(trained_model):
    _, _, model = trained_model
    prediction = predict(
        "A queue processes items in first in first out order. "
        "Enqueue inserts an item and dequeue removes the front item.",
        model,
    )
    assert prediction["label"] == "Data Structures"
    assert prediction["score_type"] in {
        "Linear SVM decision margin",
        "Naive Bayes log score",
    }
    assert prediction["scores"][0]["label"] == prediction["label"]
    assert "probability" not in prediction["score_type"].lower()


def test_short_text_is_flagged_for_review(trained_model):
    _, _, model = trained_model
    prediction = predict("array queue", model)
    assert prediction["needs_review"] is True
    assert prediction["short_text"] is True


def test_unknown_vocabulary_is_not_forced_into_a_subject(trained_model):
    _, _, model = trained_model
    prediction = predict("quasar xenolith zephyr glaucous", model)
    assert prediction["label"] == "Unknown / outside the demonstration vocabulary"
    assert prediction["needs_review"] is True
    assert prediction["scores"] == []


def test_empty_text_returns_unknown_without_model_score(trained_model):
    _, _, model = trained_model
    prediction = predict("", model)
    assert prediction["label"] == "Unknown / insufficient text"
    assert prediction["scores"] == []


def test_training_rejects_unknown_categories():
    with pytest.raises(ValueError, match="Unknown subject labels"):
        train(
            ["example about trees", "example about SQL"],
            ["Unknown Subject", "Database Management Systems"],
        )


def test_test_only_term_is_not_in_tfidf_training_vocabulary():
    labels = ["Data Structures"] * 5 + ["Database Management Systems"] * 5
    indices = list(range(len(labels)))
    _, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    texts = [
        "array tree graph educational example"
        if label == "Data Structures"
        else "database sql relation query educational example"
        for label in labels
    ]
    texts[test_indices[0]] += " uniquelyheldouttoken"

    model = train(texts, labels)
    vocabulary = model["pipeline"].named_steps["tfidf"].vocabulary_
    assert "uniquelyheldouttoken" not in vocabulary

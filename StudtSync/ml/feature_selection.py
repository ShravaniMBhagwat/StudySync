"""Leakage-aware chi-squared feature selection for the NoteMap demo corpus."""

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_SEED = 42


def compare_feature_selection(
    texts: list[str],
    labels: list[str],
    requested_features: int = 30,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Compare a full TF-IDF baseline with a chi-squared reduced feature set.

    The vectorizer and selector are pipeline steps, so each is fit only on the
    training split. The held-out test split is used only for the final scores.
    """
    if len(texts) != len(labels) or not texts:
        raise ValueError("Provide the same non-zero number of texts and labels.")
    if requested_features < 1:
        raise ValueError("requested_features must be positive.")
    if len(set(labels)) < 2:
        raise ValueError("Feature selection requires at least two classes.")

    indices = np.arange(len(texts))
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

    baseline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("classifier", LinearSVC(random_state=RANDOM_SEED, max_iter=5000)),
        ]
    )
    baseline.fit(train_texts, train_labels)
    baseline_predictions = baseline.predict(test_texts)
    baseline_vocabulary = baseline.named_steps["tfidf"].get_feature_names_out()

    selected_count = min(requested_features, len(baseline_vocabulary))
    if selected_count < 1:
        raise ValueError("Training text did not produce any TF-IDF features.")
    selected = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("select", SelectKBest(score_func=chi2, k=selected_count)),
            ("classifier", LinearSVC(random_state=RANDOM_SEED, max_iter=5000)),
        ]
    )
    selected.fit(train_texts, train_labels)
    selected_predictions = selected.predict(test_texts)
    support = selected.named_steps["select"].get_support()
    selected_terms = (
        selected.named_steps["tfidf"].get_feature_names_out()[support].tolist()
    )

    return {
        "baseline_accuracy": float(accuracy_score(test_labels, baseline_predictions)),
        "baseline_macro_f1": float(
            f1_score(test_labels, baseline_predictions, average="macro", zero_division=0)
        ),
        "selected_accuracy": float(accuracy_score(test_labels, selected_predictions)),
        "selected_macro_f1": float(
            f1_score(test_labels, selected_predictions, average="macro", zero_division=0)
        ),
        "requested_features": requested_features,
        "selected_features": selected_count,
        "total_features": len(baseline_vocabulary),
        "selected_terms": selected_terms,
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "random_seed": RANDOM_SEED,
    }

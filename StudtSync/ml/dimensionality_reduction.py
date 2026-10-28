"""Sparse TF-IDF dimensionality reduction using TruncatedSVD (LSA)."""

import base64
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def reduce_tfidf(
    texts: list[str],
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit TF-IDF and SVD; return a plot only if a 2-D projection is meaningful."""
    clean_texts = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if len(clean_texts) < 3:
        return {
            "available": False,
            "reason": "At least three non-empty documents are needed for a 2-D illustration.",
        }

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(clean_texts)
    except ValueError:
        return {
            "available": False,
            "reason": "The documents do not contain enough searchable terms for TF-IDF.",
        }

    components = min(2, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if components < 2:
        return {
            "available": False,
            "reason": "The TF-IDF matrix has insufficient rank for a 2-D projection.",
            "document_count": matrix.shape[0],
            "feature_count": matrix.shape[1],
        }

    svd = TruncatedSVD(n_components=components, random_state=random_state)
    coordinates = svd.fit_transform(matrix)
    variance = svd.explained_variance_ratio_.tolist()
    has_variation = bool(
        np.isfinite(coordinates).all()
        and np.ptp(coordinates, axis=0).min() > 1e-10
    )
    if not has_variation:
        return {
            "available": False,
            "reason": "The fitted projection has no meaningful variation to display.",
            "document_count": matrix.shape[0],
            "feature_count": matrix.shape[1],
            "explained_variance": variance,
        }

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.8, color="#5267d8")
    for index, (x_coord, y_coord) in enumerate(coordinates):
        axis.annotate(str(index + 1), (x_coord, y_coord), fontsize=7, alpha=0.75)
    axis.set_xlabel(f"Component 1 ({variance[0]:.1%} variance)")
    axis.set_ylabel(f"Component 2 ({variance[1]:.1%} variance)")
    axis.set_title("TruncatedSVD projection of TF-IDF documents")
    buffer = BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    return {
        "available": True,
        "chart": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "explained_variance": variance,
        "total_explained_variance": float(sum(variance)),
        "document_count": matrix.shape[0],
        "feature_count": matrix.shape[1],
        "random_state": random_state,
    }

"""Exploratory KMeans clustering for educational text documents."""

import base64
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def cluster_documents(
    texts: list[str],
    n_clusters: int = 4,
    max_features: int = 5000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Cluster TF-IDF documents and summarize representative terms/examples."""
    clean_texts = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if len(clean_texts) < 3:
        return {
            "available": False,
            "reason": "At least three non-empty documents are needed for clustering.",
        }
    if not 2 <= n_clusters < len(clean_texts):
        return {
            "available": False,
            "reason": f"Choose between 2 and {len(clean_texts) - 1} clusters for {len(clean_texts)} documents.",
        }

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
    )
    try:
        matrix = vectorizer.fit_transform(clean_texts)
    except ValueError:
        return {
            "available": False,
            "reason": "The documents do not contain enough searchable terms for TF-IDF.",
        }
    if matrix.shape[1] < 2:
        return {
            "available": False,
            "reason": "At least two distinct TF-IDF terms are required for clustering.",
        }
    canonical = matrix.tocsr()
    distinct_rows = {
        (
            tuple(canonical.indices[canonical.indptr[row]:canonical.indptr[row + 1]]),
            tuple(
                np.round(
                    canonical.data[canonical.indptr[row]:canonical.indptr[row + 1]],
                    decimals=10,
                )
            ),
        )
        for row in range(canonical.shape[0])
    }
    distinct_count = len(distinct_rows)
    if distinct_count < n_clusters:
        return {
            "available": False,
            "reason": f"Only {distinct_count} distinct TF-IDF document vectors are available; reduce the cluster count.",
        }

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(matrix)
    actual_clusters = len(set(labels))
    silhouette = (
        float(silhouette_score(matrix, labels, metric="cosine"))
        if 2 <= actual_clusters < len(clean_texts)
        else None
    )
    terms = vectorizer.get_feature_names_out()
    cluster_summaries = []
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(labels == cluster_id)
        center = model.cluster_centers_[cluster_id]
        top_indices = np.argsort(center)[::-1]
        top_terms = [
            str(terms[index])
            for index in top_indices
            if center[index] > 0
        ][:8]
        member_vectors = matrix[members]
        similarities_to_centroid = np.asarray(member_vectors @ center).ravel()
        representative_index = int(members[np.argmax(similarities_to_centroid)])
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "document_count": int(len(members)),
                "top_terms": top_terms,
                "example": clean_texts[representative_index],
            }
        )

    coordinates = None
    if min(matrix.shape) >= 3:
        try:
            coordinates = TruncatedSVD(
                n_components=2,
                random_state=random_state,
            ).fit_transform(matrix)
        except ValueError:
            coordinates = None
    chart = None
    if (
        coordinates is not None
        and np.isfinite(coordinates).all()
        and np.ptp(coordinates, axis=0).min() > 1e-10
    ):
        figure, axis = plt.subplots(figsize=(7, 5))
        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.8,
        )
        axis.set_xlabel("SVD component 1")
        axis.set_ylabel("SVD component 2")
        axis.set_title("Exploratory clusters in a 2-D TF-IDF projection")
        figure.colorbar(scatter, ax=axis, label="KMeans cluster")
        buffer = BytesIO()
        figure.tight_layout()
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(figure)
        chart = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "available": True,
        "labels": [int(label) for label in labels],
        "silhouette_score": silhouette,
        "inertia": float(model.inertia_),
        "clusters": cluster_summaries,
        "n_clusters": n_clusters,
        "document_count": len(clean_texts),
        "chart": chart,
        "random_state": random_state,
        "diagnostic_note": (
            "Silhouette score compares within-cluster cohesion to nearest-cluster separation; "
            "it is descriptive and does not establish that the clusters are meaningful syllabus units."
            if silhouette is not None
            else "Silhouette score is unavailable because fewer than two distinct clusters were formed."
        ),
    }


def reduce_for_plot(tfidf_matrix, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Apply TruncatedSVD to a sparse TF-IDF matrix, validating feasible rank."""
    if n_components < 1:
        raise ValueError("n_components must be positive.")
    if min(tfidf_matrix.shape) <= n_components:
        raise ValueError("The TF-IDF matrix is too small for the requested SVD components.")
    reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(tfidf_matrix)

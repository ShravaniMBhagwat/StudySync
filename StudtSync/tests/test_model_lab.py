"""Model Lab experiment and UI tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app
from config import BASE_DIR
from ml.clustering import cluster_documents, reduce_for_plot
from ml.dimensionality_reduction import reduce_tfidf
from ml.feature_selection import compare_feature_selection
from ml.regression import FEATURE_COLUMNS, TARGET_COLUMN, evaluate_frame, run_experiment
from ml.classifier import load_demo_dataset


@pytest.fixture(scope="module")
def demo_corpus():
    dataset_path = os.path.join(BASE_DIR, "data", "training", "demo_notes.csv")
    return load_demo_dataset(dataset_path)


class TestFeatureSelection:
    def test_select_kbest_reports_real_split_and_selected_terms(self, demo_corpus):
        texts, labels = demo_corpus
        result = compare_feature_selection(texts, labels, requested_features=25)
        assert result["random_seed"] == 42
        assert result["train_count"] + result["test_count"] == len(texts)
        assert result["selected_features"] == 25
        assert result["selected_features"] <= result["total_features"]
        assert len(result["selected_terms"]) == 25
        assert 0 <= result["baseline_accuracy"] <= 1
        assert 0 <= result["selected_macro_f1"] <= 1


class TestDimensionalityReduction:
    def test_svd_returns_visualization_for_sufficient_varied_corpus(self, demo_corpus):
        texts, _ = demo_corpus
        result = reduce_tfidf(texts)
        assert result["available"]
        assert result["document_count"] == len(texts)
        assert result["feature_count"] > 2
        assert result["chart"].startswith("data:image/png;base64,")
        assert 0 <= result["total_explained_variance"] <= 1

    def test_svd_skips_plot_for_insufficient_documents(self):
        result = reduce_tfidf(["array structure", "database systems"])
        assert not result["available"]
        assert "three non-empty documents" in result["reason"]

    def test_sparse_svd_helper_rejects_impossible_rank(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        matrix = TfidfVectorizer().fit_transform(
            ["array tree graph", "database table query", "network packet router"]
        )
        reduced = reduce_for_plot(matrix)
        assert reduced.shape == (3, 2)


class TestClustering:
    def test_kmeans_reports_clusters_representatives_and_diagnostics(self, demo_corpus):
        texts, _ = demo_corpus
        result = cluster_documents(texts, n_clusters=4)
        assert result["available"]
        assert len(result["labels"]) == len(texts)
        assert len(result["clusters"]) == 4
        assert all(cluster["top_terms"] for cluster in result["clusters"])
        assert all(cluster["example"] for cluster in result["clusters"])
        assert result["silhouette_score"] is not None
        assert result["chart"].startswith("data:image/png;base64,")
        assert result["random_state"] == 42

    def test_kmeans_handles_too_few_documents(self):
        result = cluster_documents(["database table", "network packet"], n_clusters=2)
        assert not result["available"]
        assert "At least three" in result["reason"]

    def test_kmeans_handles_identical_documents(self):
        result = cluster_documents(["database table", "database table", "database table"], n_clusters=2)
        assert not result["available"]
        assert "distinct TF-IDF" in result["reason"]


class TestRegression:
    def test_missing_uci_dataset_reports_no_fabricated_metrics(self, tmp_path):
        result = run_experiment(str(tmp_path))
        assert not result["available"]
        assert "no records or metrics are generated" in result["reason"]
        assert "mae" not in result

    def test_linear_regression_evaluates_valid_numeric_dataset(self):
        # A compact synthetic fixture validates calculations only; it is not
        # used by the app or presented as student-performance evidence.
        rows = []
        for index in range(50):
            rows.append(
                {
                    "studytime": index % 4 + 1,
                    "failures": index % 3,
                    "absences": index,
                    "G1": index % 20,
                    "G2": (index * 2) % 20,
                    "G3": index * 0.5,
                }
            )
        result = evaluate_frame(pd.DataFrame(rows))
        assert result["available"]
        assert result["n_train"] + result["n_test"] == result["n_used"] == 50
        assert result["features_used"] == FEATURE_COLUMNS
        assert result["target"] == TARGET_COLUMN
        assert np.isfinite([result["mae"], result["rmse"], result["r2"]]).all()
        assert result["actual_predicted_chart"].startswith("data:image/png;base64,")

    def test_regression_rejects_missing_columns_and_tiny_data(self):
        with pytest.raises(ValueError, match="missing required columns"):
            evaluate_frame(pd.DataFrame({"G3": [1] * 12}))
        rows = [
            {
                column: value
                for column, value in zip(FEATURE_COLUMNS + [TARGET_COLUMN], range(6))
            }
            for _ in range(5)
        ]
        with pytest.raises(ValueError, match="At least 10 rows"):
            evaluate_frame(pd.DataFrame(rows))


class TestModelLabRoute:
    def test_model_lab_renders_all_experiment_sections(self, demo_corpus, tmp_path):
        application = flask_app.app
        previous_dataset_dir = flask_app.DATASET_DIR if hasattr(flask_app, "DATASET_DIR") else None
        flask_app.DATASET_DIR = str(tmp_path)
        try:
            application.config["TESTING"] = True
            with application.test_client() as client:
                response = client.get("/model-lab")
                assert response.status_code == 200
                for heading in (
                    b"Feature selection",
                    b"TruncatedSVD",
                    b"KMeans clustering",
                    b"Linear regression baseline",
                ):
                    assert heading in response.data
                assert b"no records or metrics are generated" in response.data
        finally:
            if previous_dataset_dir is not None:
                flask_app.DATASET_DIR = previous_dataset_dir

    def test_custom_cluster_run_uses_entered_documents(self, tmp_path):
        previous_dataset_dir = flask_app.DATASET_DIR
        flask_app.DATASET_DIR = str(tmp_path)
        try:
            flask_app.app.config["TESTING"] = True
            with flask_app.app.test_client() as client:
                response = client.post(
                    "/model-lab",
                    data={
                        "documents": (
                            "array tree graph traversal algorithms\n\n"
                            "database sql tables query indexes\n\n"
                            "network packets router protocol ethernet\n\n"
                            "operating system process memory scheduling\n"
                        ),
                        "n_clusters": "2",
                    },
                )
                assert response.status_code == 200
                assert b"(entered for this request)" in response.data
                assert b"into 2 discovered clusters" in response.data
        finally:
            flask_app.DATASET_DIR = previous_dataset_dir

    def test_invalid_cluster_count_is_rejected(self, tmp_path):
        previous_dataset_dir = flask_app.DATASET_DIR
        flask_app.DATASET_DIR = str(tmp_path)
        try:
            flask_app.app.config["TESTING"] = True
            with flask_app.app.test_client() as client:
                response = client.post(
                    "/model-lab",
                    data={"documents": "", "n_clusters": "1"},
                    follow_redirects=True,
                )
                assert response.status_code == 200
                assert b"Choose between 2 and 10 clusters" in response.data
        finally:
            flask_app.DATASET_DIR = previous_dataset_dir

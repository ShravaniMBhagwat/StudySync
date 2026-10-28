"""Linear regression baseline for the documented UCI Student Performance data.

The experiment runs only when the real ``student-mat.csv`` file is present.
No synthetic student records or fallback metrics are generated.
"""

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["studytime", "failures", "absences", "G1", "G2"]
TARGET_COLUMN = "G3"
DATASET_CITATION = (
    "P. Cortez and A. Silva. 'Using Data Mining to Predict Secondary School "
    "Student Performance.' EUROSIS, 2008. UCI Machine Learning Repository: "
    "https://archive.ics.uci.edu/dataset/320/student+performance"
)
HONEST_DISCLAIMER = (
    "This public dataset experiment is an educational example. It does not "
    "predict an individual's personal marks or future exam performance."
)


def load_dataset(dataset_dir: str) -> pd.DataFrame | None:
    """Load the real UCI mathematics-course file, or return None if absent."""
    path = os.path.join(dataset_dir, "student-mat.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, sep=";")


def evaluate_frame(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit and evaluate Linear Regression on a validated real-data frame."""
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}.")
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required for a train/test experiment.")

    data = frame[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 10:
        raise ValueError("At least 10 complete numeric rows are required.")
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The numerical target must contain at least two distinct values.")

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "available": True,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "n_used": len(data),
        "features_used": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "random_state": random_state,
        "actual_predicted_chart": plot_regression(
            y_test.to_numpy(),
            predictions,
        ),
        "citation": DATASET_CITATION,
        "disclaimer": HONEST_DISCLAIMER,
    }


def run_experiment(
    dataset_dir: str,
    output_dir: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run on the actual UCI CSV if available; otherwise report unavailable."""
    del output_dir
    frame = load_dataset(dataset_dir)
    if frame is None:
        return {
            "available": False,
            "reason": (
                "The real UCI student-mat.csv file is not installed. Download it "
                "from the documented UCI source and place it in data/datasets; "
                "no records or metrics are generated as a fallback."
            ),
            "citation": DATASET_CITATION,
            "disclaimer": HONEST_DISCLAIMER,
        }
    return evaluate_frame(frame, test_size=test_size, random_state=random_state)


def plot_regression(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | None = None,
    filename: str = "regression_scatter.png",
) -> str:
    """Return a chart data URI for actual versus predicted held-out values."""
    del output_dir, filename
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(y_test, y_pred, alpha=0.7)
    lower = float(min(np.min(y_test), np.min(y_pred)))
    upper = float(max(np.max(y_test), np.max(y_pred)))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#d45d5d")
    axis.set_xlabel("Actual held-out grade")
    axis.set_ylabel("Predicted grade")
    axis.set_title("Linear regression: held-out predictions")
    figure.tight_layout()
    from io import BytesIO
    import base64

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

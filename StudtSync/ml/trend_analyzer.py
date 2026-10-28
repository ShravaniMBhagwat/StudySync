"""Pandas summaries and private, in-memory Matplotlib charts for QuestionBank."""

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def compute_topic_frequency(
    mapped_df: pd.DataFrame,
    syllabus_topics: list[str] | None = None,
) -> pd.DataFrame:
    """Count accepted mappings, including zero-count syllabus topics."""
    accepted = mapped_df[mapped_df["matched_topic"] != "Unmatched"]
    counts = accepted["matched_topic"].value_counts()
    topics = syllabus_topics if syllabus_topics is not None else sorted(counts.index)
    return pd.DataFrame(
        {
            "topic": topics,
            "question_count": [int(counts.get(topic, 0)) for topic in topics],
        }
    )


def compute_year_trend(mapped_df: pd.DataFrame) -> pd.DataFrame:
    """Return a topic-by-year Pandas pivot, omitting papers with unknown years."""
    if mapped_df.empty:
        return pd.DataFrame()
    known = mapped_df[
        (mapped_df["year"] != "Unknown") & (mapped_df["matched_topic"] != "Unmatched")
    ]
    if known.empty:
        return pd.DataFrame()
    return pd.crosstab(known["matched_topic"], known["year"]).sort_index(axis=1)


def _chart_data_uri(figure) -> str:
    buffer = BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_topic_frequency(freq_df: pd.DataFrame) -> str:
    """Render a frequency chart into a data URI (no public static upload/chart file)."""
    height = max(3.0, min(0.55 * len(freq_df) + 1.0, 10.0))
    figure, axis = plt.subplots(figsize=(9, height))
    axis.barh(freq_df["topic"], freq_df["question_count"], color="#5267d8")
    axis.set_xlabel("Mapped question count")
    axis.set_title("Question counts by syllabus topic")
    axis.invert_yaxis()
    return _chart_data_uri(figure)


def plot_year_trend(trend_df: pd.DataFrame) -> str | None:
    """Render available year-wise counts, or return None when years are unavailable."""
    if trend_df.empty:
        return None
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for topic in trend_df.index:
        axis.plot(
            trend_df.columns.astype(str),
            trend_df.loc[topic].values,
            marker="o",
            label=topic,
        )
    axis.set_ylabel("Mapped question count")
    axis.set_title("Year-wise question counts")
    axis.legend(loc="best", fontsize="small")
    return _chart_data_uri(figure)


def plot_classification_comparison(comparisons: dict, model_keys: list[str]) -> str | None:
    """Plot observed held-out test metrics for the trained classifiers."""
    metric_names = ["accuracy", "precision_macro", "recall_macro", "macro_f1"]
    metric_labels = ["Accuracy", "Macro precision", "Macro recall", "Macro-F1"]
    available = [
        key for key in model_keys
        if key in comparisons and comparisons[key].get("accuracy") is not None
    ]
    if not available:
        return None
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = list(range(len(metric_names)))
    width = 0.8 / len(available)
    for model_index, key in enumerate(available):
        metrics = comparisons[key]
        values = [metrics[metric] for metric in metric_names]
        offsets = [
            position - 0.4 + width / 2 + model_index * width
            for position in positions
        ]
        axis.bar(offsets, values, width=width, label=metrics["name"])
    axis.set_xticks(positions, metric_labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out score")
    axis.set_title("Actual test-set classification metrics")
    axis.legend()
    return _chart_data_uri(figure)


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> str | None:
    """Render the selected model's measured confusion matrix."""
    if not matrix or not labels:
        return None
    figure_width = max(6.0, min(0.9 * len(labels) + 3, 14))
    figure, axis = plt.subplots(figsize=(figure_width, figure_width * 0.8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted subject")
    axis.set_ylabel("Actual subject")
    axis.set_title("Selected model: held-out confusion matrix")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Question-note count")
    return _chart_data_uri(figure)

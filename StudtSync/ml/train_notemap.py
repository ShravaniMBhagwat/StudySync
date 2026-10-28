"""Train, compare, and print evaluation for the NoteMap demo dataset.

Run from the project root with ``python -m ml.train_notemap``.
"""

from pathlib import Path

from config import BASE_DIR
from ml.classifier import load_demo_dataset, train


def main() -> None:
    dataset_path = Path(BASE_DIR) / "data" / "training" / "demo_notes.csv"
    texts, labels = load_demo_dataset(str(dataset_path))
    result = train(texts, labels)

    print(f"Dataset: {dataset_path}")
    print(f"Examples: {len(texts)} | Train: {result['train_size']} | Test: {result['test_count']}")
    print(f"Stratified CV folds on training partition: {result['cv_folds']}")
    print(f"Selected model (training-only CV macro-F1): {result['selected_name']}")
    print("All metrics below are from the held-out test partition.")

    for model_key, metrics in result["comparisons"].items():
        cv_mean = metrics["cv_macro_f1_mean"]
        cv_std = metrics["cv_macro_f1_std"]
        cv_text = f"{cv_mean:.3f} +/- {cv_std:.3f}" if cv_mean is not None else "not run"
        print(f"\n{metrics['name']} (CV macro-F1: {cv_text})")
        print(f"  Accuracy:      {metrics['accuracy']:.3f}")
        print(f"  Precision:     {metrics['precision_macro']:.3f} (macro)")
        print(f"  Recall:        {metrics['recall_macro']:.3f} (macro)")
        print(f"  Macro-F1:      {metrics['macro_f1']:.3f}")
        print("  Confusion matrix (rows=actual, columns=predicted):")
        print("  Labels: " + " | ".join(result["classes"]))
        for label, row in zip(result["classes"], metrics["confusion_matrix"]):
            print(f"  {label}: {row}")

    print(
        "\nLimitations: these are small, newly generated demonstration examples. "
        "Scores describe this split only and do not establish performance on real student notes."
    )


if __name__ == "__main__":
    main()

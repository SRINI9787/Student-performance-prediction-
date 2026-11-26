"""Train and save the student pass/fail Random Forest model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "study_hours",
]
TARGET = "result"


def train_model() -> float:
    """Train the classifier, report held-out metrics, and save the model."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    required_columns = FEATURES + [TARGET]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "dataset.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    data = data[required_columns].copy()
    if data.isnull().any().any():
        raise ValueError("dataset.csv contains missing values.")

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data[TARGET] = data[TARGET].astype(str).str.strip().str.title()
    allowed_results = {"Pass", "Fail"}
    unexpected_results = sorted(set(data[TARGET]) - allowed_results)
    if unexpected_results:
        raise ValueError(
            "The result column may contain only Pass or Fail; found: "
            + ", ".join(unexpected_results)
        )
    if data[TARGET].nunique() != 2 or data[TARGET].value_counts().min() < 2:
        raise ValueError(
            "dataset.csv must include at least two rows for both Pass and Fail."
        )

    features = data[FEATURES]
    labels = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows: actual, columns: predicted; Fail, Pass):")
    print(confusion_matrix(y_test, predictions, labels=["Fail", "Pass"]))

    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"\nTrained model saved to: {MODEL_PATH}")
    return accuracy


if __name__ == "__main__":
    train_model()

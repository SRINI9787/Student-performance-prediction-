"""Flask interface for the student performance prediction model."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "study_hours",
]
INPUT_LIMITS = {
    "attendance": (0, 100),
    "internal_marks": (0, 100),
    "assignment_marks": (0, 100),
    "study_hours": (0, 24),
}

app = Flask(__name__)

if MODEL_PATH.is_file():
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    if model_bundle["features"] != FEATURES:
        raise ValueError(
            "The feature columns in model.pkl do not match the application."
        )
else:
    model = None


def read_student_inputs(form_data):
    """Return validated numeric inputs or a user-readable validation error."""
    values = {}
    for field in FEATURES:
        raw_value = form_data.get(field, "").strip()
        if not raw_value:
            return None, f"Please enter a value for {field.replace('_', ' ')}."
        try:
            value = float(raw_value)
        except ValueError:
            return None, f"{field.replace('_', ' ').title()} must be a number."

        minimum, maximum = INPUT_LIMITS[field]
        if not minimum <= value <= maximum:
            return (
                None,
                f"{field.replace('_', ' ').title()} must be between "
                f"{minimum} and {maximum}.",
            )
        values[field] = value
    return values, None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    submitted_values = {}

    if request.method == "POST":
        submitted_values = {
            field: request.form.get(field, "").strip() for field in FEATURES
        }
        values, error = read_student_inputs(request.form)
        if error is None:
            if model is None:
                error = "The model file is missing. Run `python train_model.py` first."
            else:
                input_frame = pd.DataFrame([values], columns=FEATURES)
                prediction = model.predict(input_frame)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=submitted_values,
        model_ready=model is not None,
    )


if __name__ == "__main__":
    app.run()

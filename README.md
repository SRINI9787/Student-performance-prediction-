# Student Performance Prediction

A Flask web app that predicts **Pass** or **Fail** from attendance, internal marks, assignment marks, and daily study hours. It uses a Random Forest Classifier trained and evaluated with a held-out test split.

> **Dataset note:** The supplied workspace did not contain the referenced `dataset.csv`, so this project includes a small synthetic demonstration dataset. Its labels are illustrative, and its test accuracy is not evidence of real-world performance. Replace it with representative, properly labeled student data before using predictions for any consequential decision.

## Project files

- `dataset.csv` — training data; must have `attendance`, `internal_marks`, `assignment_marks`, `study_hours`, and `result` columns.
- `train_model.py` — validates the CSV, splits training and test rows, trains the Random Forest, reports accuracy and other metrics, and writes `model.pkl`.
- `model.pkl` — generated trained model; created by the training command.
- `app.py` — Flask application and input validation.
- `templates/index.html` — responsive prediction form and result display.
- `static/style.css` — page styling.
- `requirements.txt` — Python dependencies.

## Run in VS Code

1. Install Python 3.12, then open this project folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use the VS Code terminal with Command Prompt instead:

   ```bat
   .venv\Scripts\activate.bat
   ```

4. In VS Code, select the `.venv` interpreter using **Python: Select Interpreter**.
5. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

6. Train, evaluate, and save the model:

   ```bash
   python train_model.py
   ```

   The command prints held-out test accuracy, a classification report, and a confusion matrix, then saves `model.pkl` in the project folder.
7. Start Flask:

   ```bash
   python app.py
   ```

8. Open the local address printed in the terminal (usually `http://127.0.0.1:5000`) in a browser. Stop the server with **Ctrl+C**.

## Dataset format

Use one row per student. The four input columns should be numeric; `attendance`, `internal_marks`, and `assignment_marks` are percentages from 0 to 100, while `study_hours` is hours per day from 0 to 24. The target column `result` must contain `Pass` or `Fail`. Keep both classes represented in the dataset.

The app applies the same input ranges. Retrain the model after changing `dataset.csv`. Accuracy depends on the quality, size, and representativeness of the data, and should be assessed on a suitable independent dataset before real use.

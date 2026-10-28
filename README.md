# Student Performance Prediction — AI/ML using Python

## Project overview
A Streamlit web application that trains a Random Forest regression model to estimate a student's final score from study hours, attendance, previous score, sleep hours, and assignment completion.

**Important:** `data/student_performance.csv` is a synthetic demonstration dataset, not real student data. Model metrics are educational examples and should not be presented as real-world performance.

## Features
- Score prediction through interactive sliders
- Optional upload of a CSV dataset
- Dataset charts and correlation visualization
- MAE, RMSE, and R² evaluation on a held-out test split
- Download prediction and dataset as CSV

## Requirements
- Python 3.10–3.12 recommended
- VS Code
- Internet connection for initial package installation

## Run in VS Code (Windows)
1. Extract the ZIP file.
2. Open the `Student_Performance_AI_ML_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Alternatively use Command Prompt: `.venv\Scripts\activate.bat`.
6. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Start the web app:
   ```powershell
   streamlit run app.py
   ```
8. Streamlit normally opens the local app in your browser. Stop it with `Ctrl+C`.

## Dataset columns
| Column | Meaning | Example range |
|---|---|---|
| study_hours | Study hours per day | 0–12 |
| attendance | Attendance percentage | 0–100 |
| previous_score | Previous score percentage | 0–100 |
| sleep_hours | Sleep hours per night | 0–12 |
| assignments_completed | Assignment completion percentage | 0–100 |
| final_score | Target score to predict | 0–100 |

## Model workflow
1. Load the CSV with Pandas.
2. Select five input features and `final_score` as the target.
3. Split data into training (80%) and testing (20%) sets.
4. Train `RandomForestRegressor` on the training data.
5. Predict test-set scores.
6. Calculate MAE, RMSE, and R².
7. Use the trained model to predict a new input from the Streamlit sidebar.

## Uploading a custom dataset
The CSV must contain all six required columns above and at least 30 valid rows. Keep the feature units consistent. Only upload data you have permission to use and remove personal identifiers. The app will fall back to the included demo dataset if the uploaded file is invalid.

## Suggested future improvements
- Use a real, properly licensed dataset with a documented source
- Add cross-validation and hyperparameter tuning
- Compare Random Forest with Linear Regression and Gradient Boosting
- Add tests, logging, and model versioning
- Add explainability with permutation importance
- Deploy the app to a hosting platform after reviewing privacy and security

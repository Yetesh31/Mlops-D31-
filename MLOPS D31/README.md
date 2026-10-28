# Titanic Survival Prediction — FastAPI Assignment

## Objective
Train a machine-learning pipeline and demonstrate online, batch, streaming inference and FastAPI model serving.

## Dataset
- `train.csv` — supplied Titanic training dataset (891 rows).
- Original source: https://www.kaggle.com/competitions/titanic/data
- Features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- Target: `Survived`

## Model and results
- Algorithm: Random Forest classifier in a scikit-learn Pipeline.
- Split: stratified 80% training / 20% testing (random_state=42).
- Test accuracy: **0.8156 (81.56%)**
- Preprocessing and classifier are saved together in `model.pkl`.
- `model_original_uploaded.pkl` is the original uploaded artifact kept for reference. The included `model.pkl` was rebuilt from the provided dataset because the uploaded pickle could not be loaded in the current environment due to a scikit-learn internal-version compatibility error.

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run tasks
```bash
python train_model.py
python online_inference.py
python batch_inference.py
python streaming_inference.py
```
`batch_inference.py` writes `batch_predictions.csv`. The supplied `outputs/unseen_passengers.csv` contains held-out records used by the demonstration scripts.

## Start FastAPI
```bash
uvicorn app:app --reload
```
- Health check: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- In another terminal: `python client.py`

For a validation-error test, send a request with an invalid value such as `"Sex": "unknown"`; Pydantic should return HTTP 422.

## Files
- `train_model.py`: train/evaluate/save pipeline
- `model.pkl`: runnable saved pipeline
- `model_original_uploaded.pkl`: original uploaded model, retained for comparison
- `online_inference.py`, `batch_inference.py`, `streaming_inference.py`: inference demonstrations
- `app.py`, `client.py`: API and client
- `batch_predictions.csv`: generated batch results
- `assignment_report.pdf`: observations and execution notes
- `screenshots/`: screenshots supplied by the student

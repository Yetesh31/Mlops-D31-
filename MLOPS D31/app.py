from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.pkl")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="REST API for Titanic passenger survival prediction",
    version="1.0.0",
)

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str

    @field_validator("Sex")
    @classmethod
    def valid_sex(cls, value):
        if value not in {"male", "female"}:
            raise ValueError("Sex must be 'male' or 'female'")
        return value

    @field_validator("Embarked")
    @classmethod
    def valid_embarked(cls, value):
        if value not in {"C", "Q", "S"}:
            raise ValueError("Embarked must be one of C, Q, S")
        return value

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Titanic Prediction API is running"}

@app.post("/predict")
def predict(passenger: Passenger):
    input_data = pd.DataFrame([passenger.model_dump()])
    prediction = int(model.predict(input_data)[0])
    return {
        "prediction": prediction,
        "result": "Survived" if prediction == 1 else "Did Not Survive"
    }

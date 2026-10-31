from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_stroke_model.pkl"
SCALER_PATH = ROOT_DIR / "models" / "scaler.pkl"
THRESHOLD = 0.5

FEATURES = [
    "Chest Pain",
    "Shortness of Breath",
    "Irregular Heartbeat",
    "Fatigue & Weakness",
    "Dizziness",
    "Swelling (Edema)",
    "Pain in Neck/Jaw/Shoulder/Back",
    "Excessive Sweating",
    "Persistent Cough",
    "Nausea/Vomiting",
    "High Blood Pressure",
    "Chest Discomfort (Activity)",
    "Cold Hands/Feet",
    "Snoring/Sleep Apnea",
    "Anxiety/Feeling of Doom",
    "Age",
]

if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
    raise RuntimeError(f"Model artifacts not found: {MODEL_PATH} and {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    prediction: int = Field(ge=0, le=1)
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(gt=0, lt=1)


app = FastAPI(title="Stroke Risk Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "null",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Stroke Risk Prediction API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    missing_features = sorted(set(FEATURES) - request.features.keys())
    if missing_features:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing_features}",
        )

    unknown_features = sorted(set(request.features) - set(FEATURES))
    if unknown_features:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown features: {unknown_features}",
        )

    binary_features = [feature for feature in FEATURES if feature != "Age"]
    invalid_binary_features = [
        feature
        for feature in binary_features
        if request.features[feature] not in (0, 1)
    ]
    if invalid_binary_features:
        raise HTTPException(
            status_code=422,
            detail=f"Binary features must be 0 or 1: {invalid_binary_features}",
        )

    age = request.features["Age"]
    if not age.is_integer() or not 1 <= age <= 120:
        raise HTTPException(status_code=422, detail="Age must be an integer from 1 to 120.")

    input_df = pd.DataFrame(
        [{feature: request.features[feature] for feature in FEATURES}]
    )
    try:
        input_df["Age"] = scaler.transform(input_df[["Age"]])
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(probability >= THRESHOLD)
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        threshold=THRESHOLD,
    )

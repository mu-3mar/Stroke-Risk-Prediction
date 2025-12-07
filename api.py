import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

# Define the FastAPI app
app = FastAPI(
    title="Stroke Risk Prediction API",
    description="API to predict stroke risk based on patient symptoms and demographic data.",
    version="1.0.0"
)

# Paths to artifacts
MODEL_PATH = "artifacts/best_stroke_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

# Global variables for model and scaler
model = None
scaler = None

# Input Schema
class PatientData(BaseModel):
    # Categorical Features (0 for Absent, 1 for Present)
    chest_pain: int = Field(..., description="0: Absent, 1: Present")
    shortness_of_breath: int = Field(..., description="0: Absent, 1: Present")
    irregular_heartbeat: int = Field(..., description="0: Absent, 1: Present")
    fatigue_weakness: int = Field(..., description="0: Absent, 1: Present")
    dizziness: int = Field(..., description="0: Absent, 1: Present")
    swelling_edema: int = Field(..., description="0: Absent, 1: Present")
    pain_neck_jaw_shoulder_back: int = Field(..., description="0: Absent, 1: Present")
    excessive_sweating: int = Field(..., description="0: Absent, 1: Present")
    persistent_cough: int = Field(..., description="0: Absent, 1: Present")
    nausea_vomiting: int = Field(..., description="0: Absent, 1: Present")
    high_blood_pressure: int = Field(..., description="0: Absent, 1: Present")
    chest_discomfort_activity: int = Field(..., description="0: Absent, 1: Present")
    cold_hands_feet: int = Field(..., description="0: Absent, 1: Present")
    snoring_sleep_apnea: int = Field(..., description="0: Absent, 1: Present")
    anxiety_feeling_of_doom: int = Field(..., description="0: Absent, 1: Present")
    
    # Numerical Features
    age: int = Field(..., ge=0, le=120, description="Patient's age in years")

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError("Artifacts not found. Please run the training notebook first.")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")

@app.get("/")
def home():
    return {"message": "Stroke Risk Prediction API is running. Use /predict to get a prediction."}

@app.post("/predict")
def predict(data: PatientData):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # Prepare input vector (order must match training)
    # The order in the dataset was: Chest Pain... Anxiety, Age.
    # We must construct a DataFrame or array in that precise order.
    
    input_data = {
        'Chest Pain': [data.chest_pain],
        'Shortness of Breath': [data.shortness_of_breath],
        'Irregular Heartbeat': [data.irregular_heartbeat],
        'Fatigue & Weakness': [data.fatigue_weakness],
        'Dizziness': [data.dizziness],
        'Swelling (Edema)': [data.swelling_edema],
        'Pain in Neck/Jaw/Shoulder/Back': [data.pain_neck_jaw_shoulder_back],
        'Excessive Sweating': [data.excessive_sweating],
        'Persistent Cough': [data.persistent_cough],
        'Nausea/Vomiting': [data.nausea_vomiting],
        'High Blood Pressure': [data.high_blood_pressure],
        'Chest Discomfort (Activity)': [data.chest_discomfort_activity],
        'Cold Hands/Feet': [data.cold_hands_feet],
        'Snoring/Sleep Apnea': [data.snoring_sleep_apnea],
        'Anxiety/Feeling of Doom': [data.anxiety_feeling_of_doom],
        'Age': [data.age]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Preprocessing: Scale Age
    # Note: Scaler expects a 2D array for the column it was trained on. 
    # In the notebook: X_train['Age'] = scaler.fit_transform(X_train[['Age']])
    # So we apply it to the 'Age' column.
    
    input_df['Age'] = scaler.transform(input_df[['Age']])
    
    # Predict
    prediction_prob = model.predict_proba(input_df)[0][1] # Probability of Class 1 (At Risk)
    prediction_class = model.predict(input_df)[0]
    
    result = "At Risk" if prediction_class == 1 else "Not At Risk"
    
    return {
        "prediction": result,
        "risk_probability": float(prediction_prob),
        "input_summary": data.dict()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

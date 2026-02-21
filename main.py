from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Preventive Health AI API")

# Allow Flutter / external apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Global ML Objects
# ----------------------------
model = None
scaler = None
feature_columns = None
label_encoders = None

# ----------------------------
# Load Model on Startup
# ----------------------------
@app.on_event("startup")
def load_ml_models():
    global model, scaler, feature_columns, label_encoders

    print("Loading ML model...")

    model = tf.keras.models.load_model("final_advanced_multi_domain_model.keras")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    label_encoders = joblib.load("multi_label_encoders.pkl")

    print("Model loaded successfully!")

# ----------------------------
# Root Route (Health Check)
# ----------------------------
@app.get("/")
def root():
    return {"status": "Preventive Health API Running"}

# ----------------------------
# Input Schema
# ----------------------------
class HealthInput(BaseModel):
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    blood_sugar: int
    cholesterol: int
    spo2: int
    body_temperature: float
    smoking: str
    alcohol: str
    exercise_level: str
    diet_type: str
    sleep_hours: float
    stress_level: str
    screen_time_hours: float
    water_intake_l: float

    fatigue: int
    mild_headache: int
    occasional_chest_discomfort: int
    frequent_urination: int
    mild_breathlessness: int
    dry_cough: int
    weight_gain: int
    weight_loss: int
    blurred_vision: int
    dizziness: int
    sleep_disturbance: int
    irregular_heartbeat: int
    leg_swelling: int
    loss_of_appetite: int

# ----------------------------
# Recommendation Engine
# ----------------------------
def generate_recommendations(risks):
    recs = []

    if risks["heart_risk"] == "High":
        recs += [
            "Consult a cardiologist",
            "Reduce salt intake",
            "Monitor blood pressure weekly"
        ]

    if risks["metabolic_risk"] == "High":
        recs += [
            "Reduce sugar intake",
            "Increase daily physical activity",
            "Check fasting blood sugar"
        ]

    if risks["stress_risk"] == "High":
        recs += [
            "Improve sleep schedule",
            "Practice stress management techniques"
        ]

    if risks["lung_risk"] == "High":
        recs += [
            "Avoid smoking",
            "Consult doctor if breathlessness continues"
        ]

    if risks["lifestyle_risk"] == "High":
        recs += [
            "Start structured exercise routine",
            "Improve diet quality"
        ]

    if not recs:
        recs.append("Maintain your current healthy lifestyle.")

    return recs

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
def predict(data: HealthInput):

    input_dict = data.dict()

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Calculate BMI
    df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    predictions = model.predict(df_scaled)

    risk_names = [
        "heart_risk",
        "metabolic_risk",
        "stress_risk",
        "lung_risk",
        "lifestyle_risk"
    ]

    output = {}

    for i, risk in enumerate(risk_names):
        pred_class = np.argmax(predictions[i], axis=1)
        label = label_encoders[risk].inverse_transform(pred_class)[0]
        confidence = float(np.max(predictions[i]))

        output[risk] = label
        output[f"{risk}_confidence"] = round(confidence, 3)

    # Generate recommendations
    recommendations = generate_recommendations(output)

    return {
        "risks": output,
        "recommendations": recommendations
    }

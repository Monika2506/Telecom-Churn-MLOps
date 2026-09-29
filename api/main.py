from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0"
)

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Load feature names
features = joblib.load("models/features.pkl")


class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Telecom Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([{
        "SeniorCitizen": data.SeniorCitizen,
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges
    }])

    # Add missing features with default value
    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Keep the same feature order used during training
    input_data = input_data[features]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "result": "Customer likely to churn"
        if prediction == 1
        else "Customer likely to stay"
    }
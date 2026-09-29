import joblib
import pandas as pd


def test_model_load():
    model = joblib.load("models/churn_model.pkl")
    assert model is not None


def test_model_prediction():
    model = joblib.load("models/churn_model.pkl")
    features = joblib.load("models/features.pkl")

    data = pd.DataFrame([{
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0
    }])

    for feature in features:
        if feature not in data.columns:
            data[feature] = 0

    data = data[features]

    prediction = model.predict(data)

    assert prediction[0] in [0, 1]
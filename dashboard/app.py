import streamlit as st
import pandas as pd
import joblib


# Page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model and features
model = joblib.load("models/churn_model.pkl")
features = joblib.load("models/features.pkl")


# Title
st.title("📊 Telecom Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")


# Sidebar
st.sidebar.header("Customer Information")

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.5
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=846.0
)


# Prediction button
if st.sidebar.button("Predict Churn"):

    # Create input data
    input_data = pd.DataFrame([{
        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Add missing features
    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Arrange features in training order
    input_data = input_data[features]

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    # Progress bar
    st.write("Churn Probability")
    st.progress(float(probability))
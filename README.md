# Telecom Customer Churn Prediction using MLOps

## 📌 Project Overview

Telecom customer churn is a major challenge for telecommunication companies. Customers may leave a service because of high charges, poor network quality, competitor offers, or dissatisfaction.

This project uses Machine Learning and MLOps practices to predict whether a telecom customer is likely to churn.

The project implements an end-to-end MLOps workflow including:

- Data preprocessing
- Machine learning model training
- Random Forest classification
- MLflow experiment tracking
- FastAPI model deployment
- Streamlit dashboard
- Docker containerization
- Docker Compose
- Automated testing
- GitHub Actions CI/CD
- Model performance monitoring
- Data drift monitoring
- Model retraining
- Model comparison

---

## 🎯 Objectives

1. Build a machine learning model to predict telecom customer churn.
2. Identify customers who are likely to leave the service.
3. Provide churn probability for individual customers.
4. Track machine learning experiments using MLflow.
5. Deploy the model using FastAPI.
6. Provide an interactive Streamlit dashboard.
7. Containerize the application using Docker.
8. Automate testing and Docker builds using GitHub Actions.
9. Monitor model performance and data drift.
10. Support model retraining and comparison.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Random Forest | Churn prediction model |
| MLflow | Experiment tracking |
| FastAPI | REST API |
| Streamlit | Interactive dashboard |
| Docker | Containerization |
| Docker Compose | Multi-container deployment |
| Pytest | Automated testing |
| Git | Version control |
| GitHub | Source code repository |
| GitHub Actions | CI/CD |

---

## 📂 Project Structure

```text
Telecom-Churn-MLOps/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── telco_churn.csv
│   └── processed_telco_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── features.pkl
│   ├── churn_model_retrained.pkl
│   └── features_retrained.pkl
│
├── monitoring/
│   ├── monitor.py
│   ├── data_drift.py
│   ├── monitoring_results.csv
│   ├── data_drift_results.csv
│   └── model_comparison.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── mlflow_train.py
│   ├── retrain.py
│   └── compare_models.py
│
├── tests/
│   └── test_model.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
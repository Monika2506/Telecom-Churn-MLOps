import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load processed dataset
df = pd.read_csv("data/processed_telco_churn.csv")

X = df.drop("Churn_1", axis=1)
y = df["Churn_1"]


# Use the same test data for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Load original model
original_model = joblib.load("models/churn_model.pkl")

# Load retrained model
retrained_model = joblib.load("models/churn_model_retrained.pkl")


# Predictions
original_pred = original_model.predict(X_test)
retrained_pred = retrained_model.predict(X_test)


# Calculate metrics
original_metrics = {
    "Accuracy": accuracy_score(y_test, original_pred),
    "Precision": precision_score(y_test, original_pred),
    "Recall": recall_score(y_test, original_pred),
    "F1 Score": f1_score(y_test, original_pred)
}

retrained_metrics = {
    "Accuracy": accuracy_score(y_test, retrained_pred),
    "Precision": precision_score(y_test, retrained_pred),
    "Recall": recall_score(y_test, retrained_pred),
    "F1 Score": f1_score(y_test, retrained_pred)
}


# Display comparison
comparison = pd.DataFrame({
    "Original Model": original_metrics,
    "Retrained Model": retrained_metrics
})

print("===== MODEL COMPARISON =====")
print(comparison.round(4))


# Save comparison
comparison.to_csv(
    "monitoring/model_comparison.csv"
)

print("\nModel comparison saved to:")
print("monitoring/model_comparison.csv")
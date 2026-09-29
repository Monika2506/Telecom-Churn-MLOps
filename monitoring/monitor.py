import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load processed dataset
df = pd.read_csv("data/processed_telco_churn.csv")

# Separate features and target
X = df.drop("Churn_1", axis=1)
y = df["Churn_1"]

# Use the same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("===== MODEL PERFORMANCE MONITORING =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Performance threshold
if f1 < 0.50:
    print("\nWARNING: Model performance is low!")
else:
    print("\nModel performance is acceptable.")
# Save monitoring results
results = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

results.to_csv("monitoring/monitoring_results.csv", index=False)

print("\nMonitoring results saved to monitoring/monitoring_results.csv")
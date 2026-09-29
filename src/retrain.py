import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load processed dataset
df = pd.read_csv("data/processed_telco_churn.csv")

# Separate features and target
X = df.drop("Churn_1", axis=1)
y = df["Churn_1"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("===== MODEL RETRAINING =====")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Create new Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the new model
model.fit(X_train, y_train)


# Evaluate new model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nNew Model Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save retrained model
joblib.dump(model, "models/churn_model_retrained.pkl")

# Save feature names
joblib.dump(X.columns.tolist(), "models/features_retrained.pkl")


print("\nRetrained model saved successfully!")
print("Model: models/churn_model_retrained.pkl")
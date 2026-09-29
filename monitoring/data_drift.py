import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load processed dataset
df = pd.read_csv("data/processed_telco_churn.csv")

# Separate features and target
X = df.drop("Churn_1", axis=1)
y = df["Churn_1"]

# Split data
X_train, X_test, _, _ = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index (PSI)."""

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Create bins using training data
    breakpoints = np.unique(
        np.percentile(expected, np.linspace(0, 100, bins + 1))
    )

    # Constant columns need special handling
    if len(breakpoints) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert counts to proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return psi


print("===== DATA DRIFT MONITORING =====")

drift_results = []

for column in X_train.columns:

    psi_value = calculate_psi(
        X_train[column],
        X_test[column]
    )

    if psi_value < 0.10:
        status = "No significant drift"
    elif psi_value < 0.25:
        status = "Moderate drift"
    else:
        status = "Significant drift"

    drift_results.append({
        "feature": column,
        "psi": round(psi_value, 4),
        "status": status
    })

    print(
        f"{column}: PSI={psi_value:.4f} -> {status}"
    )


# Save drift results
drift_df = pd.DataFrame(drift_results)

drift_df.to_csv(
    "monitoring/data_drift_results.csv",
    index=False
)

print("\nData drift results saved to:")
print("monitoring/data_drift_results.csv")
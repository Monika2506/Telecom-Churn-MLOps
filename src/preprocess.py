import pandas as pd

# Load dataset
df = pd.read_csv("data/telco_churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove customer ID
df = df.drop("customerID", axis=1)

# Convert Yes/No columns
df = df.replace({"Yes": 1, "No": 0})

# Convert categorical columns using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Handle any missing values created during conversion
df = df.fillna(0)

print("Processed Dataset Shape:", df.shape)

print("\nProcessed Columns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

# Save processed dataset
df.to_csv("data/processed_telco_churn.csv", index=False)

print("\nProcessed dataset saved successfully!")
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib
# Load Dataset
df = pd.read_csv("Customer-Churn.csv")

print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())
# -----------------------------------
# Data Cleaning
# -----------------------------------

# Remove customerID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values (if any)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\nData Cleaning Completed Successfully!")

print("\nUpdated Data Types:")
print(df.dtypes)
# -----------------------------------
# Encode Categorical Columns
# -----------------------------------

label_encoder = LabelEncoder()

for column in df.select_dtypes(include=["object", "string"]).columns:
    df[column] = label_encoder.fit_transform(df[column])

print("\nCategorical Data Encoded Successfully!")

print(df.head())
# -----------------------------------
# Separate Features and Target
# -----------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
# -----------------------------------
# Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# -----------------------------------
# Train Random Forest Model
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")
# -----------------------------------
# Make Predictions
# -----------------------------------

y_pred = model.predict(X_test)

print("\nPredictions Completed Successfully!")
# -----------------------------------
# Model Accuracy
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
# -----------------------------------
# Classification Report
# -----------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# -----------------------------------
# Confusion Matrix
# -----------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
# -----------------------------------
# Save Trained Model
# -----------------------------------

joblib.dump(model, "models/customer_churn_model.pkl")

print("\nModel Saved Successfully!")
# -----------------------------------
# Feature Importance
# -----------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))
# -----------------------------------
# Confusion Matrix Visualization
# -----------------------------------



sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Customer Churn Prediction - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()
# -----------------------------------
# Feature Importance Graph
# -----------------------------------

# -----------------------------------
# Feature Importance Graph
# -----------------------------------

importance.head(10).plot(
    x="Feature",
    y="Importance",
    kind="bar",
    legend=False,
    figsize=(10,6)
)

plt.title("Top 10 Important Features")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)

plt.show()

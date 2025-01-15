# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score
import zipfile
import os


dev_data = pd.read_csv(r"C:\Users\Anooj Dilip Archana\Downloads\Dev_data_to_be_shared 3")  # Replace with actual file name
val_data = pd.read_csv(r"C:\Users\Anooj Dilip Archana\Downloads\Dev_data_to_be_shared 3")  # Replace with actual file name

# Display first few rows of the development data
print("Development Data Overview:")
print(dev_data.head())

print("\nValidation Data Overview:")
print(val_data.head())

# Step 3: Exploratory Data Analysis (EDA)
# Basic statistics and summary
print("\nDevelopment Data Summary:")
print(dev_data.describe())

# Check for missing values
print("\nMissing Values in Development Data:")
print(dev_data.isnull().sum())

# Distribution of the target variable (bad_flag)
sns.countplot(x="bad_flag", data=dev_data)
plt.title("Distribution of Default Flag (bad_flag)")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(dev_data.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Step 4: Data Preprocessing
# Drop columns with too many missing values or irrelevant columns
dev_data = dev_data.dropna(thresh=0.8 * len(dev_data), axis=1)  # Drop columns with >20% missing
val_data = val_data[dev_data.columns.difference(["bad_flag"])]  # Keep only relevant columns in validation data

# Fill missing values with mean/median
for col in dev_data.columns:
    if dev_data[col].isnull().sum() > 0:
        dev_data[col].fillna(dev_data[col].median(), inplace=True)

# Encode categorical variables (if any exist)
# Assuming all columns are numerical for simplicity

# Split Development Data into Train and Test
X = dev_data.drop(columns=["bad_flag", "account_number"])  # Features
y = dev_data["bad_flag"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Development
# Using Logistic Regression as baseline
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# ROC-AUC Score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {auc_score}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="Logistic Regression (AUC = {:.2f})".format(auc_score))
plt.plot([0, 1], [0, 1], "k--")  # Random model line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Step 7: Prediction on Validation Data
val_data_scaled = scaler.transform(val_data.drop(columns=["account_number"]))
val_pred_proba = model.predict_proba(val_data_scaled)[:, 1]

# Step 8: Export Results
output = pd.DataFrame({
    "account_number": val_data["account_number"],
    "predicted_probability": val_pred_proba
})
output.to_csv("validation_predictions.csv", index=False)
print("Predictions saved to validation_predictions.csv")

# Step 9: Conclusions (Markdown cell)
# "The model achieved an AUC-ROC of XX, showing good discriminatory power..."

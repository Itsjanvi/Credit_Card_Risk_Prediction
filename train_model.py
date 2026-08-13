import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

data = pd.read_csv("dataset/credit_risk_dataset.csv")

print("Dataset loaded successfully!\n")
print(data.head())
print("\nColumns:")
print(data.columns)


# -----------------------------------
# 2. Remove Missing Values
# -----------------------------------

data = data.dropna()


# -----------------------------------
# 3. Separate Features and Target
# -----------------------------------

target_column = "Risk"

X = data.drop(target_column, axis=1)
y = data[target_column]


# -----------------------------------
# 4. Encode Target
# -----------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nRisk classes:")
for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)


# -----------------------------------
# 5. Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 6. Create Random Forest Model
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -----------------------------------
# 7. Train Model
# -----------------------------------

model.fit(X_train, y_train)


# -----------------------------------
# 8. Prediction
# -----------------------------------

y_pred = model.predict(X_test)


# -----------------------------------
# 9. Accuracy
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# -----------------------------------
# 10. Classification Report
# -----------------------------------

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))


# -----------------------------------
# 11. Create Model Folder
# -----------------------------------

os.makedirs("model", exist_ok=True)


# -----------------------------------
# 12. Save Everything
# -----------------------------------

model_data = {
    "model": model,
    "label_encoder": label_encoder,
    "features": list(X.columns)
}

joblib.dump(
    model_data,
    "model/credit_risk_model.pkl"
)


print("\n-----------------------------------")
print("Model saved successfully!")
print("File: model/credit_risk_model.pkl")
print("-----------------------------------")
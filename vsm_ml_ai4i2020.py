import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the REAL dataset
df = pd.read_csv('ai4i2020.csv')

# 2. Select Features (Sensor readings) and Target (Breakdown)
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[features]
y = df['Machine failure']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("Training Industrie 4.0 Predictive Maintenance Model...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
classifier.fit(X_train, y_train)

# 5. Evaluate
preds = classifier.predict(X_test)
print("\n--- Model Accuracy ---")
print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
print("\nClassification Report:\n", classification_report(y_test, preds))

# 6. Save the model so Streamlit can use it instantly
joblib.dump(classifier, 'pdm_model.pkl')
print("\nModel saved as pdm_model.pkl")

# 7. Print Feature Importance
importances = pd.DataFrame({
    'Sensor': features,
    'Impact on Failure (%)': classifier.feature_importances_ * 100
}).sort_values(by='Impact on Failure (%)', ascending=False)
print("\n--- Sensor Feature Importance ---")
print(importances.round(2))
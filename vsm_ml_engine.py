import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# 1. Load the synthetic powder coating data
df = pd.read_csv('predictive_vsm_data.csv')

# 2. Feature Engineering
X = df[['Order_Quantity', 'Parts_Per_Hanger', 'Color_Change', 'Booth_Swap_Available', 'Uptime_Coating', 'Uptime_Curing']]
y_lead_time = df['Total_Lead_Time_Mins']
y_bottleneck = df['Bottleneck_Station']

# Split data into 80% training and 20% testing
X_train, X_test, y_lead_train, y_lead_test = train_test_split(X, y_lead_time, test_size=0.2, random_state=42)
_, _, y_bottle_train, y_bottle_test = train_test_split(X, y_bottleneck, test_size=0.2, random_state=42)

# 3. Model 1: Predicting Total Lead Time (Regression)
print("Training Lead Time Regressor...")
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_lead_train)
lead_time_preds = regressor.predict(X_test)

print("\n--- Lead Time Prediction Performance ---")
print(f"Mean Absolute Error (Minutes): {mean_absolute_error(y_lead_test, lead_time_preds):.2f}")
print(f"R-squared Score: {r2_score(y_lead_test, lead_time_preds):.3f}")

# 4. Model 2: Predicting Bottleneck Station (Classification)
print("\nTraining Bottleneck Classifier...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_bottle_train)
bottle_preds = classifier.predict(X_test)

print("\n--- Bottleneck Prediction Performance ---")
print(f"Classification Accuracy: {accuracy_score(y_bottle_test, bottle_preds):.2%}")

# 5. Extract Feature Importance
importances = pd.DataFrame({
    'Production Variable': X.columns,
    'Impact on Lead Time (%)': regressor.feature_importances_ * 100
}).sort_values(by='Impact on Lead Time (%)', ascending=False)

print("\n--- Feature Importance ---")
print(importances.round(2))
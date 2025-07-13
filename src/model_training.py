import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preprocessing import load_data, preprocess_data, split_data

# Paths
DATA_PATH = os.path.join("..", "data", "AmesHousing.csv")
MODEL_DIR = os.path.join("..", "models")

# Make sure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and preprocess data
df = load_data(DATA_PATH)
X, y, preprocessor = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Stacking Regressor": StackingRegressor(
        estimators=[
            ("lr", LinearRegression()),
            ("dt", DecisionTreeRegressor(random_state=42))
        ],
        final_estimator=RandomForestRegressor(random_state=42)
    )
}

# Train and evaluate
results = []
best_model = None
best_score = -float('inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append((name, mae, mse, rmse, r2))

    if r2 > best_score:
        best_score = r2
        best_model = model

# Save best model
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_random_forest.pkl"))

# Save preprocessor
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

# Print results
print("Model Evaluation Results:")
for name, mae, mse, rmse, r2 in results:
    print(f"{name:20s} | MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

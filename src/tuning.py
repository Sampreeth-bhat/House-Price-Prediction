import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from data_preprocessing import load_data, preprocess_data, split_data

# Load and preprocess the data
data_path = os.path.join("..", "data", "AmesHousing.csv")
df = load_data(data_path)
X, y, preprocessor = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Fit and evaluate
grid_search.fit(X_train, y_train)

print("✅ Best Parameters:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

print("🎯 R² Score (tuned Random Forest):", r2_score(y_test, y_pred_best))

import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from data_preprocessing import load_data, preprocess_data, split_data

# Load and preprocess data
data_path = os.path.join("..", "data", "AmesHousing.csv")
df = load_data(data_path)
X, y, preprocessor = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        "R2": r2_score(y_test, preds)
    })

# Optional: Stacking model
stacking_model = StackingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('dt', DecisionTreeRegressor(random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ],
    final_estimator=LinearRegression()
)

stacking_model.fit(X_train, y_train)
stack_preds = stacking_model.predict(X_test)
results.append({
    "Model": "Stacking Regressor",
    "MAE": mean_absolute_error(y_test, stack_preds),
    "MSE": mean_squared_error(y_test, stack_preds),
    "RMSE": mean_squared_error(y_test, preds) ** 0.5,
    "R2": r2_score(y_test, stack_preds)
})

# Show results in a clean table
results_df = pd.DataFrame(results)
print("\nModel Evaluation Results:")
print(results_df.sort_values(by="R2", ascending=False).reset_index(drop=True))


# Visualizations for model evaluation....
import matplotlib.pyplot as plt
import seaborn as sns

# Use the best model (Random Forest)
best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)

# 1. Actual vs Predicted Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Price - Random Forest")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()

# 2. Residuals Histogram
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Distribution of Residuals (Random Forest)")
plt.xlabel("Error")
plt.tight_layout()
plt.show()

# 3. Feature Importances (Top 15)
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(15)

    plt.figure(figsize=(10,6))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.tight_layout()
    plt.show()

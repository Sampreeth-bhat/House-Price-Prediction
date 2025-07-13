import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Load data
def load_data(path):
    return pd.read_csv(path)

# Preprocessing function
def preprocess_data(df):
    # Select top 20 relevant features
    selected_columns = [
        'SalePrice', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF',
        '1st Flr SF', 'Year Built', 'Full Bath', 'Year Remod/Add', 'Garage Yr Blt',
        'Mas Vnr Area', 'TotRms AbvGrd', 'Fireplaces', 'BsmtFin SF 1', 'Lot Frontage',
        'Kitchen Qual', 'Exter Qual', 'Bsmt Qual', 'Roof Style', 'Neighborhood'
    ]

    df = df[selected_columns].copy()

    # Separate target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # Identify numerical and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y, preprocessor

# Train-test split
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ✅ Only runs when this file is executed directly
if __name__ == "__main__":
    import joblib

    data_path = os.path.join("..", "data", "AmesHousing.csv")
    df = load_data(data_path)

    X, y, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save preprocessor ONLY when this script runs directly
    os.makedirs(os.path.join("..", "models"), exist_ok=True)
    joblib.dump(preprocessor, os.path.join("..", "models", "preprocessor.pkl"))

    print("✅ Preprocessing complete")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

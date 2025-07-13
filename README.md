🏡 House Price Prediction App
A machine learning-based web application that predicts house sale prices using the Ames Housing Dataset. The project includes data preprocessing, model training, hyperparameter tuning, and a Streamlit web interface for real-time predictions.

Project Overview
This project aims to build a regression model to predict the sale price of residential houses based on their features. The final model is deployed as an interactive web application using Streamlit, where users can input property details and receive price predictions instantly.

Technologies Used
Python 3.12
Scikit-learn
Pandas / NumPy
Joblib
Streamlit
Matplotlib & Seaborn (for optional visualizations)

Machine Learning Models
The following models were evaluated:
| Model                   | MAE                          | RMSE | R² Score |
| ----------------------- | ---------------------------- | ---- | -------- |
| Linear Regression       | ✅ Baseline model             |      |          |
| Decision Tree Regressor | 🔍 Compared for performance  |      |          |
| Random Forest Regressor | 🏆 Final selected model      |      |          |
| Stacking Regressor      | ⚙️ Tuned model with ensemble |      |          |

Features Used for Prediction
Overall Qual (Overall Material and Finish Quality)
Gr Liv Area (Above Grade Living Area)
Garage Cars (Garage Capacity)
Year Built, Year Remod/Add, Garage Yr Blt
Total Bsmt SF, 1st Flr SF, Mas Vnr Area, etc.
Categorical: Neighborhood, Roof Style, Kitchen Qual, Exter Qual, etc.

Streamlit App Features
📥 User input form for 20+ key house attributes
🔐 Error handling for edge cases (e.g., remodel year before construction year)
📈 Live prediction of sale price with formatted output
🧼 Integrated preprocessing pipeline using joblib
⚙️ Uses the best-tuned Random Forest model


How to Run the App Locally:
1.Clone the repo

2.Create a virtual environment:
python -m venv venv
venv\Scripts\activate     # On Windows

3.Install requirements:
pip install -r requirements.txt

4.Run the app:
streamlit run app.py


Thank You...





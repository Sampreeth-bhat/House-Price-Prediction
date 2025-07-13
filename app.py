import streamlit as st
import pandas as pd
import joblib
import os

# Load model and preprocessor
model_path = os.path.join("models", "best_random_forest.pkl")
preprocessor_path = os.path.join("models", "preprocessor.pkl")

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
except Exception as e:
    st.error(f"Model or preprocessor could not be loaded: {e}")
    st.stop()

st.title("🏠 House Price Prediction App")
st.markdown("""
    This app uses a trained Random Forest model to predict house prices based on your input.
    Please enter the following property details:
""")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        OverallQual = st.slider("Overall Quality", 1, 10, 5)
        GrLivArea = st.number_input("Above Grade Living Area (sq ft)", min_value=100, max_value=10000, value=1500)
        GarageCars = st.slider("Garage Capacity (Cars)", 0, 5, 2)
        RoofStyle = st.selectbox("Roof Style", ['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed'])
        TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
        FirstFlrSF = st.number_input("1st Floor Area (sq ft)", min_value=100, max_value=3000, value=1000)
        YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
        FullBath = st.slider("Full Bathrooms", 0, 4, 2)
        YearRemodAdd = st.number_input("Year Remodeled", min_value=1800, max_value=2024, value=2005)

    with col2:
        GarageYrBlt = st.number_input("Garage Year Built", min_value=1800, max_value=2024, value=2000)
        MasVnrArea = st.number_input("Masonry Veneer Area (sq ft)", min_value=0, max_value=1500, value=100)
        TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 1, 15, 6)
        Fireplaces = st.slider("Number of Fireplaces", 0, 5, 1)
        BsmtFinSF1 = st.number_input("Finished Basement Area 1 (sq ft)", min_value=0, max_value=3000, value=400)
        LotFrontage = st.number_input("Lot Frontage (ft)", min_value=0, max_value=200, value=60)
        KitchenQual = st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa'])
        ExterQual = st.selectbox("Exterior Quality", ['Ex', 'Gd', 'TA', 'Fa'])
        BsmtQual = st.selectbox("Basement Quality", ['Ex', 'Gd', 'TA', 'Fa'])
        Neighborhood = st.selectbox("Neighborhood", ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel',
                                                     'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer',
                                                     'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV',
                                                     'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr'])
        SaleCondition = st.selectbox("Sale Condition", ['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'])

    submit = st.form_submit_button("Predict Sale Price")

if submit:
    if YearRemodAdd < YearBuilt:
        st.error("Remodel year cannot be earlier than the year the house was built.")
    elif GarageYrBlt < YearBuilt:
        st.error("Garage year built cannot be earlier than the house year built.")
    else:
        try:
            input_data = pd.DataFrame([{
                'Overall Qual': OverallQual,
                'Gr Liv Area': GrLivArea,
                'Garage Cars': GarageCars,
                'Roof Style': RoofStyle,
                'Total Bsmt SF': TotalBsmtSF,
                '1st Flr SF': FirstFlrSF,
                'Year Built': YearBuilt,
                'Full Bath': FullBath,
                'Year Remod/Add': YearRemodAdd,
                'Garage Yr Blt': GarageYrBlt,
                'Mas Vnr Area': MasVnrArea,
                'TotRms AbvGrd': TotRmsAbvGrd,
                'Fireplaces': Fireplaces,
                'BsmtFin SF 1': BsmtFinSF1,
                'Lot Frontage': LotFrontage,
                'Kitchen Qual': KitchenQual,
                'Exter Qual': ExterQual,
                'Bsmt Qual': BsmtQual,
                'Neighborhood': Neighborhood,
                'Sale Condition': SaleCondition
            }])

            prediction = model.predict(preprocessor.transform(input_data))
            st.success(f"🏡 Predicted Sale Price: ${prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

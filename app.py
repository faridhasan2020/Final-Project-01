import joblib
import pandas as pd
import streamlit as st

REFERENCE_YEAR = 2026  # same reference year used in EDA and training: Car_Age = 2026 - Year

# Ranges seen in the training data (used for input bounds and warnings)
PRESENT_PRICE_RANGE = (0.32, 92.6)
KMS_RANGE = (500, 500_000)
CAR_AGE_RANGE = (8, 23)

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


pipeline = load_model()

st.title("🚗 Used Car Price Predictor")
st.write(
    "Estimate the selling price of a used car (in lakhs) from its showroom price, age, "
    "mileage and specifications. Model: tuned XGBoost trained on the CarDekho dataset."
)

col1, col2 = st.columns(2)
with col1:
    present_price = st.number_input(
        "Present Price (ex-showroom, in lakhs)",
        min_value=0.0, max_value=100.0, value=6.5, step=0.1,
        help="Current ex-showroom price of the same model when new.",
    )
    kms_driven = st.number_input(
        "Kilometres Driven", min_value=0, max_value=500_000, value=30_000, step=1_000
    )
    car_age = st.number_input(
        f"Car Age (years, as of {REFERENCE_YEAR})",
        min_value=0, max_value=30, value=12, step=1,
        help=f"Car_Age = {REFERENCE_YEAR} - manufacturing year. Training data covers ages "
             f"{CAR_AGE_RANGE[0]}-{CAR_AGE_RANGE[1]}.",
    )
    st.caption(f"Manufacturing year ≈ {REFERENCE_YEAR - car_age}")
with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

if st.button("Predict Selling Price", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Car_Age": car_age,
        "Owner": owner,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
    }])
    prediction = max(float(pipeline.predict(input_df)[0]), 0.0)
    st.success(f"Estimated Selling Price: **{prediction:.2f} Lakhs**")

    notes = []
    if not CAR_AGE_RANGE[0] <= car_age <= CAR_AGE_RANGE[1]:
        notes.append(f"car age outside the training range ({CAR_AGE_RANGE[0]}-{CAR_AGE_RANGE[1]} years)")
    if not PRESENT_PRICE_RANGE[0] <= present_price <= PRESENT_PRICE_RANGE[1]:
        notes.append("present price outside the training range")
    if notes:
        st.warning("Prediction may be less reliable: " + "; ".join(notes) + ".")

st.divider()
st.caption("Full Stack AI & Data Science — Final Project 01 · Used Car Price Prediction")

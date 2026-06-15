import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("crop_yield_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌾 Crop Yield Prediction App")
st.write("Predict crop yield using weather and soil data")

# -----------------------------
# INPUT FIELDS
# -----------------------------
avg_temp_c = st.number_input("Average Temperature (°C)")
total_rainfall_mm = st.number_input("Total Rainfall (mm)")
avg_humidity_percent = st.number_input("Average Humidity (%)")
n = st.number_input("Nitrogen (N)")
p = st.number_input("Phosphorus (P)")
k = st.number_input("Potassium (K)")

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Yield"):

    # Feature engineering (same as training)
    total_npk = n + p + k
    rain_temp_ratio = total_rainfall_mm / (avg_temp_c + 1)
    n_p_ratio = n / (p + 1)
    growing_index = total_rainfall_mm * avg_temp_c
    humidity_temp_index = avg_humidity_percent * avg_temp_c

    # Create input array
    input_data = np.array([[
        avg_temp_c,
        total_rainfall_mm,
        avg_humidity_percent,
        n,
        p,
        k,
        total_npk,
        rain_temp_ratio,
        n_p_ratio,
        growing_index,
        humidity_temp_index
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"🌾 Predicted Crop Yield: {prediction[0]:.2f}")
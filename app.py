import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Load the model and mappings
@st.cache_resource
def load_model_and_mappings():
    model = joblib.load("xgb_model.pkl")
    with open("label_encoder_mapping.json") as f:
        label_mapping = json.load(f)
    with open("pain_clusters.json") as f:
        pain_clusters = json.load(f)
    with open("columnwise_pain_options.json") as f:
        columnwise_pain_options = json.load(f)
    return model, label_mapping, pain_clusters, columnwise_pain_options

model, label_mapping, pain_clusters, columnwise_pain_options = load_model_and_mappings()
label_mapping_rev = {int(v): k for k, v in label_mapping.items()}

st.title("🩺 Lower Back Pain Predictor")

# Input fields
st.header("Enter Patient Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", min_value=1, max_value=200, value=60)
with col2:
    duration = st.number_input("Duration of Pain (months)", min_value=1, max_value=365, value=5)
    side_of_pain = st.selectbox("Side of Pain", ["Left", "Right", "Both", "Center", "Other"])

# Pain feature dropdowns with column-specific options
st.subheader("Pain Site and Features")
pain_inputs = []
for i in range(1, 8):
    col_name = f"Site and Features of Pain {i}"
    options = ["Unknown"] + columnwise_pain_options.get(col_name, [])
    selected = st.selectbox(col_name, options)
    pain_inputs.append(pain_clusters.get(selected, 0))

# Encode inputs
gender_encoded = 0 if gender == "Male" else 1
side_mapping = {v: i for i, v in enumerate(["Left", "Right", "Both", "Center", "Other"])}
side_encoded = side_mapping[side_of_pain]

# Prepare model input
input_array = np.array([[age, gender_encoded, weight, duration, side_encoded, *pain_inputs]])

# Debug info (optional)
# st.write("Model input array:", input_array)

# Predict
if st.button("🔍 Predict Diagnosis"):
    pred = model.predict(input_array)[0]
    # st.write("Prediction ID:", pred)
    diagnosis = label_mapping_rev.get(int(pred), "Unknown")
    st.success(f"✅ Predicted Probable Diagnosis: **{diagnosis}**")

import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# Load model
model = joblib.load("cancer_model.pkl")

# Load dataset metadata
data = load_breast_cancer()
feature_names = data.feature_names

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺"
)

st.title("🩺 Breast Cancer Prediction App")

st.write(
    "Enter tumor measurements and click Predict."
)

# Create inputs
user_inputs = []

for feature in feature_names:
    value = st.number_input(
        label=feature,
        min_value=0.0,
        value=1.0,
        format="%.4f"
    )
    user_inputs.append(value)

# Predict button
if st.button("Predict"):

    input_data = np.array(user_inputs).reshape(1, -1)

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.error("Prediction: MALIGNANT TUMOR")
    else:
        st.success("Prediction: BENIGN TUMOR")

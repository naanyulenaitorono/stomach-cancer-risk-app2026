
import streamlit as st
import pandas as pd
import joblib

# Load saved model pipeline
model = joblib.load("stomach_cancer_risk_model.joblib")

st.title("Early Risk Detection for Stomach Cancer Patients")

st.write(
    "This application uses a Gaussian Naive Bayes model "
    "to estimate patient outcome based on the provided characteristics."
)

st.header("Patient Information")

age = st.text_input("Age", "60")

sex = st.text_input("Sex", "Male")

primary_site = st.text_input(
    "Primary Site",
    "Stomach NOS"
)

county = st.text_input(
    "County",
    "Nyeri County"
)

tnm_stage = st.text_input(
    "TNM Stage",
    "Stage IV"
)

if st.button("Predict"):

    patient_data = pd.DataFrame({
        "age_numeric": [int(age)],
        "sex": [sex],
        "primary_site": [primary_site],
        "county": [county],
        "tnm_stage": [tnm_stage]
    })

    prediction = model.predict(patient_data)

    probabilities = model.predict_proba(patient_data)

    alive_index = list(model.classes_).index("Alive")
    dead_index = list(model.classes_).index("Dead")

    alive_probability = probabilities[0, alive_index]
    dead_probability = probabilities[0, dead_index]

    st.subheader("Prediction")

    st.write(
        "Predicted Status:",
        prediction[0]
    )

    st.subheader("Class Probabilities")

    st.write(
        f"Alive Probability: {alive_probability * 100:.2f}%"
    )

    st.write(
        f"Dead Probability: {dead_probability * 100:.2f}%"
    )

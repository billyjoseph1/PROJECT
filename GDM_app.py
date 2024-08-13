import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model_path = 'logistic_regression_gdm_model.joblib'
model = joblib.load(model_path)

st.title('Gestational Diabetes Mellitus (GDM) Risk Prediction')

# Sidebar for user input
st.sidebar.header('Enter Your Details Here:')

age = st.sidebar.slider('Age', 20, 45, 30)
bmi = st.sidebar.slider('Pre-pregnancy BMI', 15.0, 50.0, 25.0)
previous_gdm = st.sidebar.number_input('Previous GDM', min_value=0, max_value=1, value=0, step=1)
family_history_diabetes = st.sidebar.number_input('Family History of Diabetes', min_value=0, max_value=1, value=0, step=1)
pcos = st.sidebar.number_input('PCOS', min_value=0, max_value=1, value=0, step=1)
hypertension = st.sidebar.number_input('Hypertension', min_value=0, max_value=1, value=0, step=1)
fasting_glucose = st.sidebar.slider('Fasting Plasma Glucose', 50.0, 200.0, 100.0)
physical_activity_options = ['High', 'Low', 'Moderate']
physical_activity = st.sidebar.radio('Physical Activity', physical_activity_options)

# Prepare input data for prediction
input_data = {
    'Age': [age],
    'Pre-pregnancy BMI': [bmi],
    'Previous GDM': [previous_gdm],
    'Family History of Diabetes': [family_history_diabetes],
    'PCOS': [pcos],
    'Hypertension': [hypertension],
    'Fasting Plasma Glucose': [fasting_glucose],
    # One-hot encoding for Physical Activity
}
for option in physical_activity_options:
    input_data[f'Physical Activity_{option}'] = [1 if physical_activity == option else 0]

input_df = pd.DataFrame(input_data)
input_df = input_df[['Age', 'Pre-pregnancy BMI', 'Previous GDM', 'Family History of Diabetes', 'PCOS', 'Hypertension', 'Fasting Plasma Glucose', 'Physical Activity_High', 'Physical Activity_Low', 'Physical Activity_Moderate']]

# Button to make prediction
if st.sidebar.button('Predict GDM Risk'):
    # Getting probability scores for risk levels
    risk_prob = model.predict_proba(input_df)[0][1]
    if risk_prob < 0.33:
        risk_level = 'Low Risk'
        recommendation = "Maintain a healthy lifestyle and regular check-ups."
    elif risk_prob < 0.66:
        risk_level = 'Intermediate Risk'
        recommendation = "Consider moderating lifestyle factors and seek more frequent monitoring."
    else:
        risk_level = 'High Risk'
        recommendation = "Consult a healthcare provider for comprehensive management and monitoring."

    st.success(f'{risk_level} of GDM')
    st.info(f'Recommendation: {recommendation}')

# Explain the model
st.write('## Model Explanation:')


st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">This model predicts the risk of gestational diabetes using logistic regression based on several clinical and personal health indicators.</p>', unsafe_allow_html=True)

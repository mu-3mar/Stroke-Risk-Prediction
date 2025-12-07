import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# Paths to artifacts
MODEL_PATH = "artifacts/best_stroke_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

# Load artifacts with caching
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error(f"Artifacts not found at {MODEL_PATH} or {SCALER_PATH}. Please run the training notebook first.")
        return None, None
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

model, scaler = load_artifacts()

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main_header {
        font-size: 40px;
        font-weight: bold;
        color: #e63946;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-box-safe {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        font-size: 24px;
        font-weight: bold;
    }
    .result-box-risk {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Header
st.markdown("<p class='main_header'>Heart & Stroke Health Check</p>", unsafe_allow_html=True)
st.write("Please answer the following questions to assess your potential risk factor.")

# Form inputs
with st.form("risk_form"):
    
    # Section 1: Demographics
    st.subheader("👤 Personal Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=30, help="Enter your age in years.")
    
    st.markdown("---")
    
    # Section 2: Symptoms & Conditions
    st.subheader("🏥 Symptoms & History")
    
    # Use columns to organize the checkboxes cleanly
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**General Symptoms**")
        fatigue = st.checkbox("Fatigue & Weakness", help="Do you often feel unusually tired or weak?")
        dizziness = st.checkbox("Dizziness", help="Do you experience sudden dizziness or lightheadedness?")
        sweating = st.checkbox("Excessive Sweating", help="Do you sweat more than usual without exercise?")
        cold_limbs = st.checkbox("Cold Hands/Feet", help="Are your hands or feet frequently cold?")
        anxiety = st.checkbox("Anxiety / Feeling of Doom", help="Do you experience sudden severe anxiety?")
        
        st.markdown("**Startling Symptoms**")
        chest_pain = st.checkbox("Chest Pain", help="Do you experience pain in your chest?")
        shortness_breath = st.checkbox("Shortness of Breath", help="Do you feel winded easily?")
        irregular_heart = st.checkbox("Irregular Heartbeat", help="Does your heart feel like it's fluttering or beating irregularly?")

    with col2:
        st.markdown("**Physical Indications**")
        swelling = st.checkbox("Swelling (Edema)", help="Do you have swelling in your legs or ankles?")
        pain_radiating = st.checkbox("Pain in Neck/Jaw/Back", help="Do you have unexplained pain in these areas?")
        cough = st.checkbox("Persistent Cough", help=" do you have a cough that won't go away?")
        nausea = st.checkbox("Nausea / Vomiting", help="Do you feel sick to your stomach often?")
        
        st.markdown("**Conditions**")
        bp = st.checkbox("High Blood Pressure", help="Have you been diagnosed with hypertension?")
        chest_discomfort = st.checkbox("Chest Discomfort (Activity)", help="Do you feel pressure in your chest during physical activity?")
        snoring = st.checkbox("Snoring / Sleep Apnea", help="Do you snore loudly or gasp for air during sleep?")

    st.markdown("---")
    
    # Submit Button
    submitted = st.form_submit_button("Analyze Risk", use_container_width=True)

    if submitted:
        if model is None or scaler is None:
             st.error("Model or Scaler is missing. Cannot perform prediction.")
        else:
            # Prepare input data matching the order expected by the model
            # Note: The keys must match exactly what the model was trained on
            input_data = {
                'Chest Pain': [int(chest_pain)],
                'Shortness of Breath': [int(shortness_breath)],
                'Irregular Heartbeat': [int(irregular_heart)],
                'Fatigue & Weakness': [int(fatigue)],
                'Dizziness': [int(dizziness)],
                'Swelling (Edema)': [int(swelling)],
                'Pain in Neck/Jaw/Shoulder/Back': [int(pain_radiating)],
                'Excessive Sweating': [int(sweating)],
                'Persistent Cough': [int(cough)],
                'Nausea/Vomiting': [int(nausea)],
                'High Blood Pressure': [int(bp)],
                'Chest Discomfort (Activity)': [int(chest_discomfort)],
                'Cold Hands/Feet': [int(cold_limbs)],
                'Snoring/Sleep Apnea': [int(snoring)],
                'Anxiety/Feeling of Doom': [int(anxiety)],
                'Age': [int(age)]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Preprocessing: Scale Age using the loaded scaler
            # The scaler expects a 2D array for the column it was trained on
            try:
                input_df['Age'] = scaler.transform(input_df[['Age']])
                
                # Predict
                prediction_prob = model.predict_proba(input_df)[0][1] # Probability of Class 1 (At Risk)
                prediction_class = model.predict(input_df)[0]
                
                result = "At Risk" if prediction_class == 1 else "Not At Risk"
                probability = prediction_prob * 100
                
                if result == "At Risk":
                    st.markdown(f"<div class='result-box-risk'>Result: High Risk⚠️<br><span style='font-size:18px'>Probability: {probability:.1f}%</span></div>", unsafe_allow_html=True)
                    st.warning("Your inputs suggest patterns often found in high-risk patients. Please consult a medical professional immediately.")
                else:
                    st.markdown(f"<div class='result-box-safe'>Result: Low Risk✅<br><span style='font-size:18px'>Probability: {probability:.1f}%</span></div>", unsafe_allow_html=True)
                    st.success("Your inputs align with lower risk profiles. Maintain a healthy lifestyle!")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

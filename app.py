import streamlit as st
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# API Endpoint
API_URL = "http://localhost:8000/predict"

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
        # Prepare payload
        # Map boolean (True/False) to int (1/0)
        payload = {
            "chest_pain": int(chest_pain),
            "shortness_of_breath": int(shortness_breath),
            "irregular_heartbeat": int(irregular_heart),
            "fatigue_weakness": int(fatigue),
            "dizziness": int(dizziness),
            "swelling_edema": int(swelling),
            "pain_neck_jaw_shoulder_back": int(pain_radiating),
            "excessive_sweating": int(sweating),
            "persistent_cough": int(cough),
            "nausea_vomiting": int(nausea),
            "high_blood_pressure": int(bp),
            "chest_discomfort_activity": int(chest_discomfort),
            "cold_hands_feet": int(cold_limbs),
            "snoring_sleep_apnea": int(snoring),
            "anxiety_feeling_of_doom": int(anxiety),
            "age": int(age)
        }
        
        try:
            with st.spinner("Analyzing data..."):
                response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                probability = result["risk_probability"] * 100
                
                if prediction == "At Risk":
                    st.markdown(f"<div class='result-box-risk'>Result: High Risk⚠️<br><span style='font-size:18px'>Probability: {probability:.1f}%</span></div>", unsafe_allow_html=True)
                    st.warning("Your inputs suggest patterns often found in high-risk patients. Please consult a medical professional immediately.")
                else:
                    st.markdown(f"<div class='result-box-safe'>Result: Low Risk✅<br><span style='font-size:18px'>Probability: {probability:.1f}%</span></div>", unsafe_allow_html=True)
                    st.success("Your inputs align with lower risk profiles. Maintain a healthy lifestyle!")
                    
            else:
                st.error(f"Error communicating with API. Status Code: {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Is `api.py` running?")
            st.info("Run in terminal: `python3 api.py`")

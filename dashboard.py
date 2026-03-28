import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sepsis Early Warning System", page_icon="hospital", layout="wide")

@st.cache_resource
def load_model():
    model  = joblib.load("sepsis_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.title("Sepsis Early Warning System")
st.markdown("### AI-Powered Early Sepsis Prediction for ICU Patients")
st.markdown("---")
st.success("Model loaded successfully — AUC-ROC: 0.953")

st.sidebar.header("Patient Vital Signs")
HR         = st.sidebar.slider("Heart Rate", 0, 300, 90)
O2Sat      = st.sidebar.slider("O2 Saturation (%)", 0, 100, 97)
Temp       = st.sidebar.slider("Temperature (C)", 25.0, 45.0, 37.0)
SBP        = st.sidebar.slider("Systolic BP", 0, 300, 120)
MAP        = st.sidebar.slider("Mean Arterial Pressure", 0, 200, 85)
DBP        = st.sidebar.slider("Diastolic BP", 0, 200, 80)
Resp       = st.sidebar.slider("Respiratory Rate", 0, 60, 18)
WBC        = st.sidebar.slider("WBC Count", 0.0, 50.0, 8.0)
Lactate    = st.sidebar.slider("Lactate", 0.0, 20.0, 1.0)
Creatinine = st.sidebar.slider("Creatinine", 0.0, 20.0, 1.0)
Glucose    = st.sidebar.slider("Glucose", 0.0, 500.0, 100.0)
ICULOS     = st.sidebar.slider("ICU Length of Stay (hrs)", 0, 100, 24)
Age        = st.sidebar.slider("Age", 18, 100, 65)

if st.sidebar.button("Predict Sepsis Risk", type="primary"):
    features = {
        "HR": HR, "O2Sat": O2Sat, "Temp": Temp, "SBP": SBP,
        "MAP": MAP, "DBP": DBP, "Resp": Resp, "EtCO2": 35,
        "BaseExcess": 0, "HCO3": 24, "FiO2": 0.21, "pH": 7.4,
        "PaCO2": 40, "SaO2": 97, "AST": 30, "BUN": 15,
        "Alkalinephos": 80, "Calcium": 9, "Chloride": 100,
        "Creatinine": Creatinine, "Bilirubin_direct": 0.3,
        "Glucose": Glucose, "Lactate": Lactate, "Magnesium": 2,
        "Phosphate": 3.5, "Potassium": 4, "Bilirubin_total": 0.8,
        "TroponinI": 0.01, "Hct": 38, "Hgb": 13, "PTT": 30,
        "WBC": WBC, "Fibrinogen": 300, "Platelets": 200,
        "Age": Age, "Gender": 1, "Unit1": 1, "Unit2": 0,
        "HospAdmTime": -24, "ICULOS": ICULOS,
        "Shock_Index": HR / (SBP + 1),
        "Pulse_Pressure": SBP - DBP,
        "MAP_HR_ratio": MAP / (HR + 1),
        "Resp_O2_ratio": Resp / (O2Sat + 1),
        "HR_rolling_mean": HR,
        "HR_rolling_std": 0,
        "Resp_rolling_mean": Resp
    }
    input_df     = pd.DataFrame([features])
    input_scaled = scaler.transform(input_df)
    prob         = model.predict_proba(input_scaled)[0][1]
    risk_pct     = prob * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sepsis Risk Score", f"{risk_pct:.1f}%")
    with col2:
        if risk_pct >= 60:
            st.error("HIGH RISK — Immediate attention!")
        elif risk_pct >= 30:
            st.warning("MODERATE RISK — Monitor closely")
        else:
            st.success("LOW RISK — Routine monitoring")
    with col3:
        st.metric("Model AUC-ROC", "0.953")

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(0, 30, color="#2ecc71", height=0.5)
    ax.barh(0, 30, left=30, color="#f39c12", height=0.5)
    ax.barh(0, 40, left=60, color="#e74c3c", height=0.5)
    ax.barh(0, 2, left=max(risk_pct - 1, 0), color="black", height=0.9)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Risk Score (%)")
    ax.set_title(f"Patient Sepsis Risk: {risk_pct:.1f}%", fontweight="bold")
    st.pyplot(fig)
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model AUC-ROC", "0.953")
    with col2:
        st.metric("Patients Trained On", "40,336")
    with col3:
        st.metric("Clinical Features", "47")
    st.info("Enter patient vitals in the sidebar and click Predict.")
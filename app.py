import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CardioXAI | Clinical Intelligence Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Immersive Glassmorphic Presentation Overrides
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    .glass-header {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .glass-header h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .glass-header p {
        color: #9ca3af;
        margin: 5px 0 0 0;
        font-size: 0.95rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .card-title {
        color: #38bdf8;
        font-weight: 600;
        font-size: 1.15rem;
        margin-bottom: 25px;
        letter-spacing: 0.5px;
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        width: 100%;
        font-weight: 600;
        height: 52px;
        font-size: 1.05rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.4);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%);
    }
    
    .diagnostic-panel {
        padding: 20px;
        border-radius: 8px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='glass-header'>
        <h1>CardioXAI Diagnostics Engine</h1>
        <p>Explainable AI Interface for Quantitative Cardiovascular Risk Assessment | Engine: LightGBM + SHAP</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_pipeline():
    return joblib.load("cardiac_xai_pipeline.pkl")

try:
    pipeline = load_model_pipeline()
    model = pipeline["model"]
    explainer = pipeline["explainer"]
    feature_names = pipeline["feature_names"]
    training_means = pipeline.get("training_means", {})
except FileNotFoundError:
    st.error("FATAL ERROR: Production pipeline asset bundle 'cardiac_xai_pipeline.pkl' not found.")
    st.stop()

col_input, col_diagnostics = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Patient Physiological Metrics</div>", unsafe_allow_html=True)
    
    age = st.slider("Patient Age", min_value=1, max_value=100, value=54)
    sex = st.selectbox("Biological Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Severity Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=131)
    chol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=242)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    custom_restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", ["0", "1", "2", "3", "4"])
    thal = st.selectbox("Thalassemia Diagnostic Type", ["Normal", "Fixed Defect", "Reversible Defect", "Unspecified"])
    
    # Structural Mapping Interface to isolate numeric alignment boundaries
    sex_val = 1 if sex == "Male" else 0
    cp_val = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
    fbs_val = 1 if fbs == "True" else 0
    restecg_val = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[custom_restecg]
    exang_val = 1 if exang == "Yes" else 0
    slope_val = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
    ca_val = int(ca)
    thal_val = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3, "Unspecified": 0}[thal]
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_diagnostics:
    st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Diagnostic Inference & Attribution Layer</div>", unsafe_allow_html=True)
    
    execute_analysis = st.button("RUN QUANTITATIVE CLINICAL EVALUATION")
    
    if execute_analysis:
        input_data = pd.DataFrame([{
            "age": age, "sex": sex_val, "cp": cp_val, "trestbps": trestbps,
            "chol": chol, "fbs": fbs_val, "restecg": restecg_val, "thalach": thalach,
            "exang": exang_val, "oldpeak": oldpeak, "slope": slope_val, "ca": ca_val, "thal": thal_val
        }])
        
        # Enforce column matrix order exactly as expected by LightGBM
        input_data = input_data[feature_names]
        
        risk_probability = float(model.predict_proba(input_data)[0][1])
        risk_class = int(model.predict(input_data)[0])
        
        st.markdown("### Risk Profiler Output")
        
        if risk_class == 1:
            st.markdown(f"""
                <div class='diagnostic-panel' style='background-color: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; color: #fca5a5;'>
                    <span style='font-weight: 700;'>[ HIGH RISK DETECTED ]</span> Patient profiles clinical indicators consistent with cardiovascular disease pathology. Risk Probability: <b>{risk_probability*100:.2f}%</b>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='diagnostic-panel' style='background-color: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; color: #a7f3d0;'>
                    <span style='font-weight: 700;'>[ LOW RISK PATTERN ]</span> Patient metrics fall inside baseline margins. No immediate pathological indication identified. Risk Probability: <b>{risk_probability*100:.2f}%</b>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>### Patient-Specific Attribution Plot (SHAP Probability Breakdown)", unsafe_allow_html=True)
        
        # Calculate tree structural explanations
        shap_values_raw = explainer(input_data)
        
        if len(shap_values_raw.shape) == 3:
            raw_values = shap_values_raw.values[0, :, 1]
            raw_base = shap_values_raw.base_values[0, 1]
        else:
            raw_values = shap_values_raw.values[0]
            raw_base = shap_values_raw.base_values[0]
            
        # Sigmoid Link Function Layer for mapping log-odds seamlessly into probabilities
        total_log_odds = raw_base + np.sum(raw_values)
        prob_final = 1 / (1 + np.exp(-total_log_odds))
        prob_base = 1 / (1 + np.exp(-raw_base))
        
        sum_raw = np.sum(raw_values)
        scaled_values = raw_values * ((prob_final - prob_base) / (sum_raw if sum_raw != 0 else 1e-6))
        
        exp = shap.Explanation(
            values=scaled_values,
            base_values=prob_base,
            data=input_data.iloc[0].values,
            feature_names=feature_names
        )
        
        # Render clean visual dashboard plot
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        shap.plots.waterfall(exp, max_display=7, show=False)
        
        fig.axes[0].tick_params(colors='white', labelsize=9)
        fig.axes[0].xaxis.label.set_color('white')
        
        for text in fig.axes[0].texts:
            text.set_color('white')
            
        st.pyplot(fig, bbox_inches='tight')
        
    st.markdown("</div>", unsafe_allow_html=True)
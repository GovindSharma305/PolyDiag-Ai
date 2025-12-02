# -*- coding: utf-8 -*-
"""
PolyDiag AI - Multiple Disease Prediction System
@author: Govind
"""

import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="PolyDiag AI - Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# --------------------------------------------------
# GLOBAL DARK THEME + UI CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark cyber UI */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #000000 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .main {
        padding: 1.5rem 2rem 2.5rem 2rem;
    }

    h1, h2, h3, h4 {
        color: #f9fafb !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(15, 23, 42, 0.88);
        border-radius: 1.2rem;
        padding: 1.3rem 1.6rem;
        border: 1px solid rgba(148, 163, 184, 0.6);
        box-shadow: 0 18px 40px rgba(0,0,0,0.7);
        backdrop-filter: blur(10px);
    }

    /* Pills / badges */
    .pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(56, 189, 248, 0.5);
        background: rgba(15, 23, 42, 0.8);
        color: #7dd3fc;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
    }

    /* Main buttons – blue/purple gradient */
    .stButton > button {
        border-radius: 999px;
        padding: 0.45rem 1.6rem;
        font-weight: 600;
        border: 1px solid rgba(148, 163, 184, 0.6);
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
        color: #f9fafb !important;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 16px 40px rgba(56, 189, 248, 0.45);
    }

    /* Result box */
    .result-box {
        margin-top: 1rem;
        padding: 1rem 1.25rem;
        border-radius: 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.7);
        font-weight: 600;
        font-size: 0.98rem;
    }

    .result-ok {
        background: radial-gradient(circle at top left, #022c22 0, #064e3b 50%, #022c22 100%);
        color: #bbf7d0;
    }

    .result-bad {
        background: radial-gradient(circle at top left, #450a0a 0, #7f1d1d 50%, #450a0a 100%);
        color: #fecaca;
    }

    .result-neutral {
        background: radial-gradient(circle at top left, #020617 0, #0f172a 50%, #020617 100%);
        color: #e5e7eb;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: radial-gradient(circle at top, #020617 0, #0b1120 50%, #020617 100%);
        border-right: 1px solid rgba(31, 41, 55, 0.9);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Inputs on dark background */
    .stTextInput input, .stNumberInput input {
        background: rgba(15, 23, 42, 0.95) !important;
        color: #e5e7eb !important;
        border-radius: 0.6rem !important;
        border: 1px solid rgba(148, 163, 184, 0.7) !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    label, .stMarkdown p {
        color: #e5e7eb !important;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 0.9rem;
        overflow: hidden;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.8);
    }

    /* Helper text */
    .helper-text {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: -0.4rem;
        margin-bottom: 0.5rem;
    }

    /* Mini highlight chip */
    .hint-chip {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.12);
        color: #7dd3fc;
        font-size: 0.7rem;
        margin-left: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Session State for reports (to enable download)
# --------------------------------------------------
for key in ["diab_report", "heart_report", "park_report"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# --------------------------------------------------
# Default values for inputs (for Clear All)
# --------------------------------------------------
DEFAULT_DIAB = {
    "diab_preg": 0.0,
    "diab_gluc": 0.0,
    "diab_bp": 0.0,
    "diab_skin": 0.0,
    "diab_ins": 0.0,
    "diab_bmi": 0.0,
    "diab_dpf": 0.0,
    "diab_age": 0.0,
}

DEFAULT_HEART = {
    "heart_age": 0.0,
    "heart_sex": 0.0,
    "heart_cp": 0.0,
    "heart_trestbps": 0.0,
    "heart_chol": 0.0,
    "heart_fbs": 0.0,
    "heart_restecg": 0.0,
    "heart_thalach": 0.0,
    "heart_exang": 0.0,
    "heart_oldpeak": 0.0,
    "heart_slope": 0.0,
    "heart_ca": 0.0,
    "heart_thal": 0.0,
}

DEFAULT_PARK = {
    "par_fo": 0.0,
    "par_fhi": 0.0,
    "par_flo": 0.0,
    "par_jper": 0.0,
    "par_jabs": 0.0,
    "par_rap": 0.0,
    "par_ppq": 0.0,
    "par_ddp": 0.0,
    "par_shim": 0.0,
    "par_shimdb": 0.0,
    "par_apq3": 0.0,
    "par_apq5": 0.0,
    "par_apq": 0.0,
    "par_dda": 0.0,
    "par_nhr": 0.0,
    "par_hnr": 0.0,
    "par_rpde": 0.0,
    "par_dfa": 0.0,
    "par_sp1": 0.0,
    "par_sp2": 0.0,
    "par_d2": 0.0,
    "par_ppe": 0.0,
}

# --------------------------------------------------
# Helper: Load Models (cached + safe + portable)
# --------------------------------------------------
# Use pathlib for portable model path resolution
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODELS_DIR = BASE_DIR / "saved_models"

@st.cache_resource
def load_model(filename: str):
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        st.error(
            f"🔴 Model file not found: `{model_path}`. "
            f"Place `{filename}` inside the `saved_models` folder."
        )
        return None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Failed to load model {filename}: {e}")
        return None

diabetes_model = load_model("diabetes_model.sav")
heart_disease_model = load_model("heart_disease_model.sav")
parkinsons_model = load_model("parkinsons_model.sav")

# --------------------------------------------------
# Small helpers
# --------------------------------------------------
def risk_meter(prob: float, label: str = "Risk level"):
    """Render a small risk meter with a progress bar + text."""
    if prob is None:
        return
    # Ensure probability is between 0 and 1 before converting to percent
    pct = int(min(max(prob, 0.0), 1.0) * 100)
    st.write(f"**{label}:** {pct}%")
    st.progress(pct)


def result_box(text: str, good: bool | None):
    """good=True -> green, False -> red, None -> neutral."""
    if not text:
        return
    if good is True:
        css = "result-box result-ok"
    elif good is False:
        css = "result-box result-bad"
    else:
        css = "result-box result-neutral"
    st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)


# Demo value loaders
def demo_diabetes():
    st.session_state.diab_preg = 2.0
    st.session_state.diab_gluc = 150.0
    st.session_state.diab_bp = 80.0
    st.session_state.diab_skin = 35.0
    st.session_state.diab_ins = 120.0
    st.session_state.diab_bmi = 31.5
    st.session_state.diab_dpf = 0.52
    st.session_state.diab_age = 42.0


def demo_heart():
    st.session_state.heart_age = 55.0
    st.session_state.heart_sex = 1.0
    st.session_state.heart_cp = 2.0
    st.session_state.heart_trestbps = 140.0
    st.session_state.heart_chol = 250.0
    st.session_state.heart_fbs = 0.0
    st.session_state.heart_restecg = 1.0
    st.session_state.heart_thalach = 150.0
    st.session_state.heart_exang = 0.0
    st.session_state.heart_oldpeak = 1.0
    st.session_state.heart_slope = 1.0
    st.session_state.heart_ca = 0.0
    st.session_state.heart_thal = 2.0


def demo_parkinson():
    st.session_state.par_fo = 120.0
    st.session_state.par_fhi = 150.0
    st.session_state.par_flo = 90.0
    st.session_state.par_jper = 0.005
    st.session_state.par_jabs = 0.00005
    st.session_state.par_rap = 0.003
    st.session_state.par_ppq = 0.004
    st.session_state.par_ddp = 0.009
    st.session_state.par_shim = 0.03
    st.session_state.par_shimdb = 0.3
    st.session_state.par_apq3 = 0.02
    st.session_state.par_apq5 = 0.025
    st.session_state.par_apq = 0.03
    st.session_state.par_dda = 0.06
    st.session_state.par_nhr = 0.02
    st.session_state.par_hnr = 20.0
    st.session_state.par_rpde = 0.55
    st.session_state.par_dfa = 0.7
    st.session_state.par_sp1 = -4.0
    st.session_state.par_sp2 = 0.3
    st.session_state.par_d2 = 2.0
    st.session_state.par_ppe = 0.2

# ---------- CLEAR ALL HELPERS (Safe with callbacks) ----------
def clear_diabetes():
    for k, v in DEFAULT_DIAB.items():
        st.session_state[k] = v
    st.session_state.diab_report = ""

def clear_heart():
    for k, v in DEFAULT_HEART.items():
        st.session_state[k] = v
    st.session_state.heart_report = ""

def clear_parkinson():
    for k, v in DEFAULT_PARK.items():
        st.session_state[k] = v
    st.session_state.park_report = ""

def clear_all_reports():
    st.session_state.diab_report = ""
    st.session_state.heart_report = ""
    st.session_state.park_report = ""

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 🧑‍⚕️ PolyDiag AI")
    st.caption("Smart multi-disease risk assistant")

    selected = option_menu(
        "Navigation",
        [
            "Overview",
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Parkinsons Prediction",
            "Batch Prediction (CSV)",
            "About Project",
        ],
        icons=[
            "house", "activity", "heart", "person",
            "file-earmark-spreadsheet", "info-circle"
        ],
        menu_icon="hospital-fill",
        default_index=0
    )

    st.markdown("---")
    st.caption("⚠️ For awareness & experimentation only.\nNot a medical device.")

    st.markdown("---")
    st.button("🧹 Reset all reports", on_click=clear_all_reports)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if selected == "Overview":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("PolyDiag AI – Multi-Disease Prediction Hub")
    st.write("")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            Predict health risk for:

            <span class="pill">🩸 Diabetes</span>
            <span class="pill">❤️ Heart Disease</span>
            <span class="pill">🧠 Parkinson's</span>

            <br><br>
            **What you get:**
            - Dark cyber UI with smart hints  
            - Single-patient risk prediction  
            - Risk score meter (when model supports probabilities)  
            - CSV batch prediction & downloadable results  
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <br>
            <span class="hint-chip">Tip</span> Use **Demo values** on each page to instantly feel the flow.
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Models connected", "3")
        st.metric("Modes", "Single · Batch")
        st.metric("Interface", "Streamlit · Dark Glass UI")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Diabetes Prediction
# --------------------------------------------------
if selected == "Diabetes Prediction":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🩸 Diabetes Risk Prediction")

    top_col1, top_col2, top_col3, top_col4 = st.columns([1.6, 1, 1, 1])

    with top_col1:
        st.caption("Fill the clinical values. Use the hints under each input.")

    with top_col2:
        st.button("✨ Demo values", key="demo_diab", on_click=demo_diabetes)

    with top_col3:
        st.button("🧹 Clear all values", key="clear_diab", on_click=clear_diabetes)

    with top_col4:
        st.caption("Model: Pima Indians Diabetes Dataset")

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0.0,
            step=1.0,
            key="diab_preg",
        )
        st.markdown(
            '<div class="helper-text">0 if male • higher values = more risk for mothers</div>',
            unsafe_allow_html=True,
        )

        SkinThickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0,
            step=1.0,
            key="diab_skin",
        )
        st.markdown(
            '<div class="helper-text">Often between 10–50 mm</div>',
            unsafe_allow_html=True,
        )

        DiabetesPedigreeFunction = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            step=0.01,
            key="diab_dpf",
            format="%.3f"  # Added better float format
        )
        st.markdown(
            '<div class="helper-text">Higher = stronger family history</div>',
            unsafe_allow_html=True,
        )

    with col2:
        Glucose = st.number_input(
            "Glucose Level (mg/dL)",
            min_value=0.0,
            step=1.0,
            key="diab_gluc",
        )
        st.markdown(
            '<div class="helper-text">Fasting ≥ 126 mg/dL is considered high</div>',
            unsafe_allow_html=True,
        )

        Insulin = st.number_input(
            "Insulin Level (μU/mL)",
            min_value=0.0,
            step=1.0,
            key="diab_ins",
        )
        st.markdown(
            '<div class="helper-text">Typical range ≈ 15–276 μU/mL</div>',
            unsafe_allow_html=True,
        )

        Age = st.number_input(
            "Age (years)",
            min_value=0.0,
            step=1.0,
            key="diab_age",
        )
        st.markdown(
            '<div class="helper-text">Risk usually increases with age</div>',
            unsafe_allow_html=True,
        )

    with col3:
        BloodPressure = st.number_input(
            "Blood Pressure (mmHg)",
            min_value=0.0,
            step=1.0,
            key="diab_bp",
        )
        st.markdown(
            '<div class="helper-text">~80 is a common diastolic BP value</div>',
            unsafe_allow_html=True,
        )

        BMI = st.number_input(
            "BMI (kg/m²)",
            min_value=0.0,
            step=0.1,
            key="diab_bmi",
        )
        st.markdown(
            '<div class="helper-text">25–29.9 = overweight • ≥30 = obese</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    hint_col1, hint_col2 = st.columns([2, 1])

    if diabetes_model is None:
        with hint_col1:
            st.error("Model not loaded. Fix model path in `saved_models` to enable predictions.")
    else:
        with hint_col1:
            if st.button("🔍 Get Diabetes Test Result"):
                user_input = [
                    Pregnancies,
                    Glucose,
                    BloodPressure,
                    SkinThickness,
                    Insulin,
                    BMI,
                    DiabetesPedigreeFunction,
                    Age,
                ]
                # Predict and calculate probability
                prediction = diabetes_model.predict([user_input])[0]
                try:
                    proba = diabetes_model.predict_proba([user_input])[0][1]
                except:
                    proba = None

                if prediction == 1:
                    text = "🚨 The model indicates a **high likelihood of Diabetes**."
                    good = False
                else:
                    text = "✅ The model indicates a **low likelihood of Diabetes**."
                    good = True

                result_box(text, good)
                risk_meter(proba, "Estimated diabetes risk")

                risk_str = f"{proba:.2f}" if proba is not None else "N/A"

                st.session_state.diab_report = f"""PolyDiag AI - Diabetes Prediction Report

Input values:
- Pregnancies: {Pregnancies}
- Glucose: {Glucose}
- BloodPressure: {BloodPressure}
- SkinThickness: {SkinThickness}
- Insulin: {Insulin}
- BMI: {BMI}
- DiabetesPedigreeFunction: {DiabetesPedigreeFunction}
- Age: {Age}

Prediction:
{text}
Risk Score: {risk_str}

Note: This is an AI-based estimation and NOT a medical diagnosis.
"""

        with hint_col2:
            st.subheader("Smart hints")
            if Glucose >= 126:
                st.markdown("- 🔴 Glucose is in the **high** range.")
            elif Glucose > 0:
                st.markdown("- 🟢 Glucose is below the common diabetes threshold (126 mg/dL).")

            if BMI >= 30:
                st.markdown("- 🔴 BMI is in the **obese** range.")
            elif BMI >= 25:
                st.markdown("- 🟠 BMI is in the **overweight** range.")
            elif BMI > 0:
                st.markdown("- 🟢 BMI is in the **normal** range.")

            if Age >= 45:
                st.markdown("- 🧓 Age is in a higher-risk group for diabetes.")

    if st.session_state.diab_report:
        st.download_button(
            "📄 Download Diabetes Report",
            data=st.session_state.diab_report,
            file_name="diabetes_report.txt",
            mime="text/plain",
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Heart Disease Prediction
# --------------------------------------------------
if selected == "Heart Disease Prediction":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("❤️ Heart Disease Risk Prediction")

    top_col1, top_col2, top_col3, top_col4 = st.columns([1.6, 1, 1, 1])

    with top_col1:
        st.caption("Inputs follow the classic UCI Heart Disease dataset format.")
    with top_col2:
        st.button("✨ Demo values", key="demo_heart", on_click=demo_heart)
    with top_col3:
        st.button("🧹 Clear all values", key="clear_heart", on_click=clear_heart)
    with top_col4:
        st.caption("Outputs: 0 = low risk • 1 = high risk")

    st.write("")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0.0, step=1.0, key="heart_age")
        sex = st.number_input(
            "Sex (1 = male, 0 = female)",
            min_value=0.0,
            max_value=1.0,
            step=1.0,
            key="heart_sex",
        )
        cp = st.number_input(
            "Chest Pain type (0–3)",
            min_value=0.0,
            max_value=3.0,
            step=1.0,
            key="heart_cp",
        )
        trestbps = st.number_input(
            "Resting Blood Pressure (mmHg)",
            min_value=0.0,
            step=1.0,
            key="heart_trestbps",
        )

    with col2:
        chol = st.number_input(
            "Serum Cholesterol (mg/dL)",
            min_value=0.0,
            step=1.0,
            key="heart_chol",
        )
        fbs = st.number_input(
            "Fasting Blood Sugar > 120 (1/0)",
            min_value=0.0,
            max_value=1.0,
            step=1.0,
            key="heart_fbs",
        )
        restecg = st.number_input(
            "Resting ECG (0–2)",
            min_value=0.0,
            max_value=2.0,
            step=1.0,
            key="heart_restecg",
        )
        thalach = st.number_input(
            "Max Heart Rate Achieved",
            min_value=0.0,
            step=1.0,
            key="heart_thalach",
        )

    with col3:
        exang = st.number_input(
            "Exercise Induced Angina (1/0)",
            min_value=0.0,
            max_value=1.0,
            step=1.0,
            key="heart_exang",
        )
        oldpeak = st.number_input(
            "ST depression by exercise",
            step=0.1,
            key="heart_oldpeak",
        )
        slope = st.number_input(
            "Slope of ST segment (0–2)",
            min_value=0.0,
            max_value=2.0,
            step=1.0,
            key="heart_slope",
        )
        ca = st.number_input(
            "Major vessels (0–3)",
            min_value=0.0,
            max_value=3.0,
            step=1.0,
            key="heart_ca",
        )
        thal = st.number_input(
            "Thal (0 normal, 1 fixed, 2 reversible)",
            min_value=0.0,
            max_value=3.0,
            step=1.0,
            key="heart_thal",
        )

    st.markdown("---")
    hint_col1, hint_col2 = st.columns([2, 1])

    if heart_disease_model is None:
        with hint_col1:
            st.error("Model not loaded. Fix model path in `saved_models` to enable predictions.")
    else:
        with hint_col1:
            if st.button("🔍 Get Heart Disease Test Result"):
                user_input = [
                    age,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal,
                ]
                prediction = heart_disease_model.predict([user_input])[0]
                try:
                    proba = heart_disease_model.predict_proba([user_input])[0][1]
                except:
                    proba = None

                if prediction == 1:
                    text = "🚨 The model indicates a **high likelihood of Heart Disease**."
                    good = False
                else:
                    text = "✅ The model indicates a **low likelihood of Heart Disease**."
                    good = True

                result_box(text, good)
                risk_meter(proba, "Estimated heart disease risk")

                risk_str = f"{proba:.2f}" if proba is not None else "N/A"

                st.session_state.heart_report = f"""PolyDiag AI - Heart Disease Prediction Report

Key inputs:
Age: {age}
Sex: {sex}
Chest Pain (cp): {cp}
Resting BP: {trestbps}
Cholesterol: {chol}
Fasting Blood Sugar: {fbs}
Rest ECG: {restecg}
Max Heart Rate: {thalach}
Exercise Induced Angina: {exang}
Oldpeak: {oldpeak}
Slope: {slope}
CA: {ca}
Thal: {thal}

Prediction:
{text}
Risk Score: {risk_str}

Note: This is an AI-based estimation and NOT a medical diagnosis.
"""

        with hint_col2:
            st.subheader("Smart hints")
            if age >= 50:
                st.markdown("- 🧓 Age is in a higher-risk group.")
            if trestbps >= 140:
                st.markdown("- 🔴 Resting blood pressure is **elevated**.")
            if chol >= 240:
                st.markdown("- 🔴 Cholesterol is in the **high** range.")
            if exang == 1:
                st.markdown("- ⚠️ Exercise-induced angina reported.")

    if st.session_state.heart_report:
        st.download_button(
            "📄 Download Heart Report",
            data=st.session_state.heart_report,
            file_name="heart_disease_report.txt",
            mime="text/plain",
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Parkinson's Prediction
# --------------------------------------------------
if selected == "Parkinsons Prediction":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🧠 Parkinson's Disease Prediction")

    top_col1, top_col2, top_col3, top_col4 = st.columns([1.6, 1, 1, 1])

    with top_col1:
        st.caption("Uses acoustic voice features from the Parkinson's dataset.")
    with top_col2:
        st.button("✨ Demo values", key="demo_park", on_click=demo_parkinson)
    with top_col3:
        st.button("🧹 Clear all values", key="clear_park", on_click=clear_parkinson)
    with top_col4:
        st.caption("Higher jitter/shimmer often = more instability")

    st.write("")
    col1, col2, col3, col4, col5 = st.columns(5)

    # Standardizing float formats for better visual consistency
    float_format = "%.5f"
    abs_float_format = "%.6f"

    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", step=0.1, key="par_fo")
        RAP = st.number_input("MDVP:RAP", step=0.0001, format=float_format, key="par_rap")
        APQ3 = st.number_input("Shimmer:APQ3", step=0.001, format=float_format, key="par_apq3")
        NHR = st.number_input("NHR", step=0.001, format=float_format, key="par_nhr")
        RPDE = st.number_input("RPDE", step=0.001, format=float_format, key="par_rpde")

    with col2:
        fhi = st.number_input("MDVP:Fhi(Hz)", step=0.1, key="par_fhi")
        PPQ = st.number_input("MDVP:PPQ", step=0.0001, format=float_format, key="par_ppq")
        APQ5 = st.number_input("Shimmer:APQ5", step=0.001, format=float_format, key="par_apq5")
        HNR = st.number_input("HNR", step=0.1, key="par_hnr")
        DFA = st.number_input("DFA", step=0.001, format=float_format, key="par_dfa")

    with col3:
        flo = st.number_input("MDVP:Flo(Hz)", step=0.1, key="par_flo")
        DDP = st.number_input("Jitter:DDP", step=0.0001, format=float_format, key="par_ddp")
        APQ = st.number_input("MDVP:APQ", step=0.001, format=float_format, key="par_apq")
        spread1 = st.number_input("spread1", step=0.001, format=float_format, key="par_sp1")
        spread2 = st.number_input("spread2", step=0.001, format=float_format, key="par_sp2")

    with col4:
        Jitter_percent = st.number_input(
            "MDVP:Jitter(%)", step=0.0001, format=float_format, key="par_jper"
        )
        Shimmer = st.number_input(
            "MDVP:Shimmer", step=0.001, format=float_format, key="par_shim"
        )
        DDA = st.number_input(
            "Shimmer:DDA", step=0.001, format=float_format, key="par_dda"
        )
        D2 = st.number_input("D2", step=0.001, format=float_format, key="par_d2")

    with col5:
        Jitter_Abs = st.number_input(
            "MDVP:Jitter(Abs)", step=0.00001, format=abs_float_format, key="par_jabs"
        )
        Shimmer_dB = st.number_input(
            "MDVP:Shimmer(dB)", step=0.01, key="par_shimdb"
        )
        PPE = st.number_input("PPE", step=0.001, format=float_format, key="par_ppe")

    st.markdown("---")
    hint_col1, hint_col2 = st.columns([2, 1])

    if parkinsons_model is None:
        with hint_col1:
            st.error("Model not loaded. Fix model path in `saved_models` to enable predictions.")
    else:
        with hint_col1:
            if st.button("🔍 Get Parkinson's Test Result"):
                user_input = [
                    fo,
                    fhi,
                    flo,
                    Jitter_percent,
                    Jitter_Abs,
                    RAP,
                    PPQ,
                    DDP,
                    Shimmer,
                    Shimmer_dB,
                    APQ3,
                    APQ5,
                    APQ,
                    DDA,
                    NHR,
                    HNR,
                    RPDE,
                    DFA,
                    spread1,
                    spread2,
                    D2,
                    PPE,
                ]
                prediction = parkinsons_model.predict([user_input])[0]
                try:
                    proba = parkinsons_model.predict_proba([user_input])[0][1]
                except:
                    proba = None

                if prediction == 1:
                    text = "🚨 The model indicates a **high likelihood of Parkinson's Disease**."
                    good = False
                else:
                    text = "✅ The model indicates a **low likelihood of Parkinson's Disease**."
                    good = True

                result_box(text, good)
                risk_meter(proba, "Estimated Parkinson's risk")

                risk_str = f"{proba:.2f}" if proba is not None else "N/A"

                st.session_state.park_report = f"""PolyDiag AI - Parkinson's Disease Prediction Report

Prediction:
{text}
Risk Score: {risk_str}

(Voice feature values omitted in this short report.)

Note: This is an AI-based estimation and NOT a medical diagnosis.
"""

        with hint_col2:
            st.subheader("Smart hints")
            if HNR > 0:
                if HNR < 15:
                    st.markdown("- 🔴 HNR is relatively **low** (more noise in voice).")
                else:
                    st.markdown("- 🟢 HNR is in a relatively **healthy** range.")
            # Simplified jitter/shimmer hint for clarity
            if Jitter_percent > 0.003 or Shimmer > 0.03: 
                 st.markdown("- ⚠️ Jitter or Shimmer is **elevated**, indicating higher voice instability.")
            elif Jitter_percent > 0 and Shimmer > 0:
                 st.markdown("- ℹ️ Jitter/Shimmer parameters show voice characteristics related to stability.")

    if st.session_state.park_report:
        st.download_button(
            "📄 Download Parkinson's Report",
            data=st.session_state.park_report,
            file_name="parkinsons_report.txt",
            mime="text/plain",
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------
if selected == "Batch Prediction (CSV)":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("📁 Batch Prediction for Multiple Records")

    st.markdown(
        """
        Upload a CSV and run predictions for many rows in one go.  
        The CSV should have the same **feature columns**, in the **same order**, as used while training the model.
        <br><br>
        <span class="hint-chip">Tip:</span> Ensure you do **not** include the target/label column (e.g., 'Outcome') in the batch file.
        """,
        unsafe_allow_html=True
    )

    model_choice = st.selectbox(
        "Select model for batch prediction:",
        ["Diabetes", "Heart Disease", "Parkinson's"],
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            df = None

        if df is not None and st.button("🚀 Run Batch Prediction"):
            model = None
            result_col = ""

            if model_choice == "Diabetes":
                if diabetes_model is None:
                    st.error("Diabetes model not loaded.")
                else:
                    model = diabetes_model
                    result_col = "Diabetes_Prediction"

            elif model_choice == "Heart Disease":
                if heart_disease_model is None:
                    st.error("Heart disease model not loaded.")
                else:
                    model = heart_disease_model
                    result_col = "HeartDisease_Prediction"

            else:  # Parkinson's
                if parkinsons_model is None:
                    st.error("Parkinson's model not loaded.")
                else:
                    model = parkinsons_model
                    result_col = "Parkinsons_Prediction"

            if model is not None:
                try:
                    # Convert DataFrame to numpy array for model prediction
                    preds = model.predict(df.values)
                    df[result_col] = preds
                    st.success(f"Batch prediction completed for {len(df)} records.")
                    st.dataframe(df.head(10))  # Showing fewer rows for better UI fit

                    # Create a buffer and download
                    csv_buffer = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        data=csv_buffer,
                        file_name=f"{model_choice.lower().replace(' ', '_')}_batch_predictions.csv",
                        mime="text/csv",
                    )
                except ValueError as e:
                    st.error(
                        f"Error during prediction (Value Error). Make sure the CSV has the **correct number of columns** "
                        f"({model.n_features_in_} expected for this model) and all are numeric. Error: {e}"
                    )
                except Exception as e:
                    st.error(
                        f"An unexpected error occurred during prediction. Error: {e}"
                    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# About Project
# --------------------------------------------------
if selected == "About Project":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("ℹ️ About PolyDiag AI")

    st.markdown(
        """
        **PolyDiag AI** connects three trained ML models (Diabetes, Heart Disease, Parkinson's)  
        inside a single dark, glass-style Streamlit interface.

        **System Information**
        - **Model Files:** Expects saved models (`.sav` files) to be located in a folder named 
          `saved_models` in the same directory as this main script.
        - **Models Used:** Pre-trained classifiers for Pima Indians Diabetes, UCI Heart Disease, 
          and Parkinson's Voice Features datasets.

        **Highlights**
        - Unified interface for multiple classifiers  
        - Single prediction + batch CSV prediction  
        - Downloadable text reports for each prediction  
        - Smart hints and demo values for quick exploration  
        - Dark neon UI inspired by modern analytics dashboards  
        """
    )

    st.markdown("---")
    st.caption(
        "This tool is meant to support learning, awareness and experimentation. "
        "For real decisions, rely on doctors and verified medical tools."
    )

    st.markdown("</div>", unsafe_allow_html=True)

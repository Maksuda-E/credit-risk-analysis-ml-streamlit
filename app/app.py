import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_model, prepare_input, transform_input
from errorLog import setup_logger

logger = setup_logger()

st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #06111f 0%, #0c1f37 50%, #132b49 100%);
        color: #e5e7eb;
    }

    header[data-testid="stHeader"] {display: none;}
    footer {visibility: hidden;}

    .block-container {
        max-width: 1400px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #e5e7eb;
    }

    .hero-card {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 45%, #38bdf8 100%);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.30);
        margin-bottom: 1 rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.45rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        line-height: 1.6;
        color: rgba(255,255,255,0.92);
        margin: 0;
    }

    .panel {
        background: rgba(8, 22, 47, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 22px;
        padding: 1.25rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        backdrop-filter: blur(8px);
    }

    .card-title {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    .section-heading {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.9rem;
    }

    .perf-card {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        text-align: left;
    }

    .perf-label {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
    }

    .perf-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .side-card {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #38bdf8 100%);
        border-radius: 22px;
        padding: 1rem 0.5rem;
        text-align: center;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
        margin-top: 2rem;
    }

    .side-label {
        color: rgba(255,255,255,0.90);
        font-size: 0.95rem;
        margin-bottom: 0.55rem;
    }

    .side-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }

    .result-box {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        font-weight: 600;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    .result-good {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        color: #d1fae5;
    }

    .result-medium {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border: 1px solid #f59e0b;
        color: #fef3c7;
    }

    .result-high {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
        color: #fee2e2;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        font-size: 1rem;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.28);
        margin-top: 0.6rem;
    }

    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {
        color: #d1d5db !important;
        font-weight: 600;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #08162f !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
    }

    input { color: #e5e7eb !important; }

    [data-testid="stDataFrame"] {
        background-color: #0f172a;
        border-radius: 14px;
        border: 1px solid #334155;
        padding: 0.35rem;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    ul[role="listbox"],
    ul[role="listbox"] *,
    li[role="option"],
    li[role="option"] * {
        color: black !important;
    }

    ul[role="listbox"], li[role="option"] {
        background: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# HERO
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Credit Card Default Prediction System</div>
        <p class="hero-subtitle">
            Predict whether a customer is likely to default on credit card payment
            using a trained Random Forest model and engineered financial behavior features.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# PERFORMANCE SECTION
st.markdown("### Model Performance")
p1, p2, p3, p4 = st.columns(4, gap="medium")
with p1:
    st.markdown('<div class="perf-card"><div class="perf-label">Accuracy</div><div class="perf-value">0.82</div></div>', unsafe_allow_html=True)
with p2:
    st.markdown('<div class="perf-card"><div class="perf-label">Precision</div><div class="perf-value">0.80</div></div>', unsafe_allow_html=True)
with p3:
    st.markdown('<div class="perf-card"><div class="perf-label">Recall</div><div class="perf-value">0.78</div></div>', unsafe_allow_html=True)
with p4:
    st.markdown('<div class="perf-card"><div class="perf-label">F1 Score</div><div class="perf-value">0.79</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, ".."))
model_path = os.path.join(root_dir, "model", "random_forest_model.pkl")
features_path = os.path.join(root_dir, "model", "model_features.pkl")
scaler_path = os.path.join(root_dir, "model", "scaler.pkl")

@st.cache_resource
def get_model(model_path_arg, features_path_arg, scaler_path_arg):
    return load_model(model_path_arg, features_path_arg, scaler_path_arg)

try:
    model, feature_names, scaler = get_model(model_path, features_path, scaler_path)
except Exception as e:
    logger.error(f"Error loading model or feature names: {e}")
    st.error(f"Model loading failed: {e}")
    st.stop()

main_left, main_right = st.columns([1.75, 1.0], gap="large")

with main_left:
   
    st.markdown('<div class="card-title">Customer Information</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        LIMIT_BAL = st.number_input("Credit Limit", min_value=0.0, value=50000.0, step=1000.0)
        AGE = st.number_input("Age", min_value=18, value=30, step=1)
        PAY_0 = st.number_input("Recent Payment Status PAY_0", min_value=-1, max_value=6, value=0, step=1)
        AVG_BILL_AMT = st.number_input("Average Bill Amount", min_value=0.0, value=10000.0, step=500.0)
        AVG_PAY_AMT = st.number_input("Average Payment Amount", min_value=0.0, value=5000.0, step=500.0)

    with c2:
        TOTAL_DELAY_MONTHS = st.number_input("Total Delay Months", min_value=0, max_value=6, value=0, step=1)
        MAX_DELAY = st.number_input("Maximum Delay", min_value=-1, max_value=6, value=0, step=1)
        AVG_UTILIZATION = st.slider("Average Credit Utilization", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        PAYMENT_RATIO = st.slider("Payment Ratio", min_value=0.0, max_value=2.0, value=0.50, step=0.01)

    st.markdown('<div class="section-heading">Categorical Information</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        sex_choice = st.selectbox("Sex", ["Male", "Female"])
    with s2:
        education_choice = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
    with s3:
        marriage_choice = st.selectbox("Marriage", ["Married", "Single", "Others"])

    predict_btn = st.button("Predict Default Risk")
    st.markdown('</div>', unsafe_allow_html=True)

SEX_2 = 1 if sex_choice == "Female" else 0
EDUCATION_2 = 1 if education_choice == "University" else 0
EDUCATION_3 = 1 if education_choice == "High School" else 0
EDUCATION_4 = 1 if education_choice == "Others" else 0
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0

with main_right:
    
    st.markdown(
        """
        <div class="side-card">
            <div class="side-label">Model</div>
            <div class="side-value">Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="side-card">
            <div class="side-label">Target</div>
            <div class="side-value">Default Risk</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    result_placeholder = st.empty()
    note_placeholder = st.empty()
    interpretation_placeholder = st.empty()

    st.markdown('</div>', unsafe_allow_html=True)

if predict_btn:
    logger.info("Prediction triggered")
    try:
        input_data = {
            "LIMIT_BAL": LIMIT_BAL,
            "AGE": AGE,
            "PAY_0": PAY_0,
            "AVG_BILL_AMT": AVG_BILL_AMT,
            "AVG_PAY_AMT": AVG_PAY_AMT,
            "TOTAL_DELAY_MONTHS": TOTAL_DELAY_MONTHS,
            "MAX_DELAY": MAX_DELAY,
            "AVG_UTILIZATION": AVG_UTILIZATION,
            "PAYMENT_RATIO": PAYMENT_RATIO,
            "SEX_2": SEX_2,
            "EDUCATION_2": EDUCATION_2,
            "EDUCATION_3": EDUCATION_3,
            "EDUCATION_4": EDUCATION_4,
            "MARRIAGE_2": MARRIAGE_2,
            "MARRIAGE_3": MARRIAGE_3
        }
        
        logger.debug(f"Raw input: {input_data}")
        
        input_df = prepare_input(input_data, feature_names)
        logger.debug(f"Prepared input: {input_df.to_dict()}")
        
        model_input = transform_input(input_df, scaler)
        logger.debug(f"Transformed input: {model_input.to_dict()}")

        probability = model.predict_proba(input_df)[0][1]
        logger.info(f"Prediction probability: {probability}")

        if probability < 0.35:
            result_placeholder.markdown(
                f'<div class="result-box result-good">Customer is not likely to default<br>Default probability: {probability:.2%}</div>',
                unsafe_allow_html=True
            )
            interpretation_placeholder.markdown("**Risk Interpretation:** This customer appears to have relatively low default risk.")
        elif probability < 0.55:
            result_placeholder.markdown(
                f'<div class="result-box result-medium">Customer appears to have moderate default risk<br>Default probability: {probability:.2%}</div>',
                unsafe_allow_html=True
            )
            interpretation_placeholder.markdown("**Risk Interpretation:** The customer shows some financial risk signals and should be monitored carefully.")
        else:
            result_placeholder.markdown(
                f'<div class="result-box result-high">Customer is likely to default<br>Default probability: {probability:.2%}</div>',
                unsafe_allow_html=True
            )
            interpretation_placeholder.markdown("**Risk Interpretation:** This customer appears to have high default risk and may require immediate attention.")

        note_placeholder.info("Probability reflects likelihood, not certainty. Use this result as decision support.")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("### Input Summary")
        st.dataframe(input_df, use_container_width=True)

        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)

            st.markdown("### Top 10 Important Features")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
            ax.set_title("Top 10 Important Features")
            ax.set_xlabel("Importance Score")
            ax.set_ylabel("Feature")
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            ax.tick_params(axis="x", colors="#e5e7eb")
            ax.tick_params(axis="y", colors="#e5e7eb")
            ax.xaxis.label.set_color("#e5e7eb")
            ax.yaxis.label.set_color("#e5e7eb")
            ax.title.set_color("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            st.pyplot(fig)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"Prediction failed: {e}")
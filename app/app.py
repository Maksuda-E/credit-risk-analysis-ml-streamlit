import os  # import os for file path handling
import streamlit as st  # import streamlit for app ui
import pandas as pd  # import pandas for dataframe handling
import matplotlib.pyplot as plt  # import matplotlib for chart plotting
from utils import load_model, prepare_input  # import helper functions
from errorLog import setup_logger  # import logger setup

logger = setup_logger()  # create logger object

st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- STYLING --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #071224 0%, #0e2749 55%, #163b66 100%);
        color: #e5e7eb;
    }
    header[data-testid="stHeader"] { display: none; }
    footer { visibility: hidden; }

    .block-container {
        max-width: 1380px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HERO SECTION --------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Credit Card Default Prediction System</div>
        <p class="hero-subtitle">
            Predict whether a customer will default using a Random Forest model.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- PATH SETUP --------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, ".."))

model_path = os.path.join(root_dir, "model", "random_forest_model.pkl")
features_path = os.path.join(root_dir, "model", "model_features.pkl")

# -------------------- LOAD MODEL --------------------
try:
    model, feature_names = load_model(model_path, features_path)
except Exception as e:
    logger.error(f"Error loading model or feature names: {e}")
    st.error(f"Model loading failed: {e}")
    st.stop()

# -------------------- LAYOUT --------------------
top_left, top_right = st.columns([1.8, 1.15], gap="large")

# -------------------- INPUT FORM --------------------
with top_left:
    st.markdown('<div class="card-title">Customer Information</div>', unsafe_allow_html=True)

    left_num, right_num = st.columns(2)

    with left_num:
        LIMIT_BAL = st.number_input("Credit Limit", 0.0, value=50000.0, step=1000.0)
        AGE = st.number_input("Age", 18, value=30)
        PAY_0 = st.number_input("PAY_0", -1, 6, 0)
        AVG_BILL_AMT = st.number_input("Average Bill", 0.0, 10000.0)
        AVG_PAY_AMT = st.number_input("Average Payment", 0.0, 5000.0)

    with right_num:
        TOTAL_DELAY_MONTHS = st.number_input("Delay Months", 0, 6, 0)
        MAX_DELAY = st.number_input("Max Delay", -1, 6, 0)
        AVG_UTILIZATION = st.slider("Utilization", 0.0, 1.0, 0.3)
        PAYMENT_RATIO = st.slider("Payment Ratio", 0.0, 2.0, 0.5)

    sex_choice = st.selectbox("Sex", ["Male", "Female"])
    education_choice = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
    marriage_choice = st.selectbox("Marriage", ["Married", "Single", "Others"])

    predict_btn = st.button("Predict")

# -------------------- METRICS --------------------
with top_right:
    st.markdown("### Model: Random Forest")
    st.markdown("### Accuracy: 0.82")
    st.markdown("### Target: Default Risk")

# -------------------- ENCODING --------------------
SEX_2 = 1 if sex_choice == "Female" else 0
EDUCATION_2 = 1 if education_choice == "University" else 0
EDUCATION_3 = 1 if education_choice == "High School" else 0
EDUCATION_4 = 1 if education_choice == "Others" else 0
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0

# -------------------- PREDICTION --------------------
if predict_btn:
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

        input_df = prepare_input(input_data, feature_names)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("## Prediction Result")

        if probability < 0.30:
            st.success(f"Low Risk ({probability:.2%})")
            interpretation = "Low default risk"
        elif probability < 0.60:
            st.warning(f"Moderate Risk ({probability:.2%})")
            interpretation = "Moderate default risk"
        else:
            st.error(f"High Risk ({probability:.2%})")
            interpretation = "High default risk"

        st.subheader("Risk Interpretation")
        st.write(interpretation)

        # -------------------- FIXED INDENTATION BLOCK --------------------
        st.subheader("Input Summary")

        styled_df = (
            input_df.style
            .set_properties(**{
                'background-color': '#0b1730',
                'color': '#e5e7eb'
            })
        )

        st.dataframe(styled_df, use_container_width=True)

        # -------------------- FEATURE IMPORTANCE --------------------
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)

            fig, ax = plt.subplots()
            ax.barh(importance_df["Feature"], importance_df["Importance"])
            ax.invert_yaxis()

            st.pyplot(fig)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"Prediction failed: {e}")
import os  # import os for file path handling
import streamlit as st  # import streamlit for app ui
import pandas as pd  # import pandas for dataframe operations
import matplotlib.pyplot as plt  # import matplotlib for plotting
from utils import load_model, prepare_input  # import helper functions
from errorLog import setup_logger  # import logger setup

logger = setup_logger()  # initialize logger

st.set_page_config(  # configure streamlit page
    page_title="Credit Card Default Prediction",  # set browser tab title
    layout="wide",  # use wide layout
    initial_sidebar_state="collapsed"  # collapse sidebar by default
)

st.markdown(  # add custom css for dark premium layout
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #071224 0%, #0f2747 55%, #16345a 100%);
        color: #e5e7eb;
    }

    header[data-testid="stHeader"] {
        display: none;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #e5e7eb;
    }

    .hero-card {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 45%, #38bdf8 100%);
        border-radius: 28px;
        padding: 2rem 2.2rem;
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.4rem;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.55rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.65;
        color: rgba(255, 255, 255, 0.92);
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.92) 0%, rgba(15, 23, 42, 0.96) 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
        min-height: 114px;
    }

    .metric-label {
        color: #a5b4fc;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.95rem;
        font-weight: 800;
        margin: 0;
    }

    .panel-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.92) 0%, rgba(15, 23, 42, 0.96) 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        padding: 1.35rem 1.25rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        color: #f8fafc;
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    .result-good {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        color: #d1fae5;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .result-medium {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border: 1px solid #f59e0b;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        color: #fef3c7;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .result-high {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        color: #fee2e2;
        font-weight: 600;
        margin-bottom: 1rem;
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
    }

    .stButton > button:hover {
        filter: brightness(1.08);
    }

    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {
        color: #d1d5db !important;
        font-weight: 600;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #0b1730 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
    }

    input {
        color: #e5e7eb !important;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 0.45rem;
        padding-bottom: 0.45rem;
    }

    [data-testid="stDataFrame"] {
        background-color: #0f172a;
        border-radius: 14px;
        border: 1px solid #334155;
        padding: 0.35rem;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    .small-space {
        height: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(  # render hero section
    """
    <div class="hero-card">
        <div class="hero-title">Credit Card Default Prediction System</div>
        <p class="hero-subtitle">
            This application predicts whether a customer is likely to default on credit card payment
            using the trained Random Forest model and engineered financial behavior features.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

left_space, metric_col1, metric_col2, metric_col3 = st.columns([0.18, 1, 1, 1])  # create left-aligned metric layout

with metric_col1:  # render first metric card
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col2:  # render second metric card
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">0.82</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col3:  # render third metric card
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Target</div>
            <div class="metric-value">Default Risk</div>
        </div>
        """,
        unsafe_allow_html=True
    )

base_dir = os.path.dirname(os.path.abspath(__file__))  # get current file directory
root_dir = os.path.abspath(os.path.join(base_dir, ".."))  # move to project root
model_path = os.path.join(root_dir, "model", "random_forest_model.pkl")  # create model path
features_path = os.path.join(root_dir, "model", "model_features.pkl")  # create features path

try:  # load saved model and features
    model, feature_names = load_model(model_path, features_path)
except Exception as e:  # handle model loading error
    logger.error(f"Error loading model or feature names: {e}")
    st.error(f"Model loading failed: {e}")
    st.stop()

left_col, right_col = st.columns([1.35, 1])  # create main input layout

with left_col:  # left side for numeric inputs
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Customer Financial Profile</div>', unsafe_allow_html=True)

    num_col1, num_col2 = st.columns(2)  # split numeric area into two columns

    with num_col1:  # first numeric column
        LIMIT_BAL = st.number_input("Credit Limit", min_value=0.0, value=50000.0, step=1000.0)
        AGE = st.number_input("Age", min_value=18, value=30, step=1)
        PAY_0 = st.number_input("Recent Payment Status PAY_0", min_value=-1, max_value=6, value=0, step=1)
        AVG_BILL_AMT = st.number_input("Average Bill Amount", min_value=0.0, value=10000.0, step=500.0)
        AVG_PAY_AMT = st.number_input("Average Payment Amount", min_value=0.0, value=5000.0, step=500.0)

    with num_col2:  # second numeric column
        TOTAL_DELAY_MONTHS = st.number_input("Total Delay Months", min_value=0, max_value=6, value=0, step=1)
        MAX_DELAY = st.number_input("Maximum Delay", min_value=-1, max_value=6, value=0, step=1)
        AVG_UTILIZATION = st.slider("Average Credit Utilization", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        PAYMENT_RATIO = st.slider("Payment Ratio", min_value=0.0, max_value=2.0, value=0.50, step=0.01)

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:  # right side for categorical inputs and action
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Categorical Information</div>', unsafe_allow_html=True)

    sex_choice = st.selectbox("Sex", ["Male", "Female"])
    education_choice = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
    marriage_choice = st.selectbox("Marriage", ["Married", "Single", "Others"])

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Run Prediction</div>', unsafe_allow_html=True)

    predict_btn = st.button("Predict Default Risk")

    st.markdown('</div>', unsafe_allow_html=True)

SEX_2 = 1 if sex_choice == "Female" else 0  # encode sex
EDUCATION_2 = 1 if education_choice == "University" else 0  # encode university
EDUCATION_3 = 1 if education_choice == "High School" else 0  # encode high school
EDUCATION_4 = 1 if education_choice == "Others" else 0  # encode others education
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0  # encode single
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0  # encode others marriage

if predict_btn:  # run prediction after button click
    try:
        input_data = {  # collect model input data
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

        input_df = prepare_input(input_data, feature_names)  # align input with training features
        prediction = model.predict(input_df)[0]  # get class prediction
        probability = model.predict_proba(input_df)[0][1]  # get default class probability

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Prediction Result</div>', unsafe_allow_html=True)

        if probability < 0.30:  # low risk band
            st.markdown(
                f"""
                <div class="result-good">
                    Customer is not likely to default<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            interpretation = "This customer appears to have relatively low default risk."
        elif probability < 0.60:  # moderate risk band
            st.markdown(
                f"""
                <div class="result-medium">
                    Customer appears to have moderate default risk<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            interpretation = "This customer appears to have moderate default risk."
        else:  # high risk band
            st.markdown(
                f"""
                <div class="result-high">
                    Customer is likely to default<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            interpretation = "This customer appears to have high default risk."

        st.subheader("Risk Interpretation")  # show interpretation title
        st.write(interpretation)  # render interpretation text

        st.subheader("Input Summary")  # show input summary title
        st.dataframe(input_df, use_container_width=True)  # display input dataframe

        if hasattr(model, "feature_importances_"):  # render chart for models that support importance
            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": model.feature_importances_
                }
            ).sort_values(by="Importance", ascending=False).head(10)

            st.subheader("Top 10 Important Features")  # show chart title

            fig, ax = plt.subplots(figsize=(10, 5))  # create figure
            ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])  # draw horizontal bars
            ax.set_title("Top 10 Important Features")  # set chart title
            ax.set_xlabel("Importance Score")  # set x label
            ax.set_ylabel("Feature")  # set y label
            fig.patch.set_facecolor("#0f172a")  # set figure background
            ax.set_facecolor("#0f172a")  # set axes background
            ax.tick_params(axis="x", colors="#e5e7eb")  # style x ticks
            ax.tick_params(axis="y", colors="#e5e7eb")  # style y ticks
            ax.xaxis.label.set_color("#e5e7eb")  # style x label
            ax.yaxis.label.set_color("#e5e7eb")  # style y label
            ax.title.set_color("#f8fafc")  # style title
            for spine in ax.spines.values():  # style border lines
                spine.set_color("#334155")
            st.pyplot(fig)  # display chart

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:  # handle prediction errors
        logger.error(f"Prediction error: {e}")
        st.error(f"Prediction failed: {e}")
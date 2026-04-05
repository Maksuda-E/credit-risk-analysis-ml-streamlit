import os  # import os to work with file paths
import streamlit as st  # import streamlit to build the web app
import pandas as pd  # import pandas for dataframe handling
import matplotlib.pyplot as plt  # import matplotlib for charts
from utils import load_model, prepare_input  # import helper functions for model loading and input preparation
from errorLog import setup_logger  # import logger setup function

logger = setup_logger()  # create a logger object for error tracking

st.set_page_config(  # configure the page settings
    page_title="Credit Card Default Prediction",  # set browser tab title
    layout="wide",  # use wide screen layout
    initial_sidebar_state="collapsed"  # keep sidebar collapsed
)

st.markdown(  # inject custom CSS for dark theme and improved layout
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        color: #e5e7eb;
    }

    .main .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #e5e7eb;
    }

    .hero-card {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 24px;
        color: white;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        color: white;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.92);
        font-size: 1.05rem;
        line-height: 1.6;
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 20px;
        padding: 1.3rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 0.6rem;
    }

    .metric-value {
        color: #f9fafb;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }

    .section-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 22px;
        padding: 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.2rem;
    }

    .section-title {
        color: #f9fafb;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .result-good {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        color: #d1fae5;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .result-medium {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border: 1px solid #f59e0b;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        color: #fef3c7;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .result-high {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
        border-radius: 18px;
        padding: 1rem 1.2rem;
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
        padding: 0.85rem 1rem;
        font-size: 1rem;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.25);
    }

    .stButton > button:hover {
        filter: brightness(1.08);
    }

    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {
        color: #d1d5db !important;
        font-weight: 500;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
    }

    input {
        color: #e5e7eb !important;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    [data-testid="stDataFrame"] {
        background-color: #111827;
        border-radius: 14px;
        border: 1px solid #374151;
        padding: 0.35rem;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(  # create the top hero section without any blank card below it
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

metric_col1, metric_col2, metric_col3 = st.columns(3)  # create three columns for summary metrics

with metric_col1:  # first metric card
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col2:  # second metric card
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">0.82</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col3:  # third metric card
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Target</div>
            <div class="metric-value">Default Risk</div>
        </div>
        """,
        unsafe_allow_html=True
    )

base_dir = os.path.dirname(os.path.abspath(__file__))  # get absolute path of current file
root_dir = os.path.abspath(os.path.join(base_dir, ".."))  # move one level up to project root
model_path = os.path.join(root_dir, "model", "random_forest_model.pkl")  # build model file path
features_path = os.path.join(root_dir, "model", "model_features.pkl")  # build feature names file path

try:  # safely load model and feature list
    model, feature_names = load_model(model_path, features_path)  # load trained model and saved features
except Exception as e:  # catch loading errors
    logger.error(f"Error loading model or feature names: {e}")  # write error to log
    st.error(f"Model loading failed: {e}")  # show error message on screen
    st.stop()  # stop app if model is not loaded

left_col, right_col = st.columns([1.3, 1])  # create two main columns for inputs

with left_col:  # left side for numeric features
    st.markdown('<div class="section-card">', unsafe_allow_html=True)  # open financial profile card
    st.markdown('<div class="section-title">Customer Financial Profile</div>', unsafe_allow_html=True)  # add card title

    num_col1, num_col2 = st.columns(2)  # create two columns inside financial profile section

    with num_col1:  # first numeric input column
        LIMIT_BAL = st.number_input("Credit Limit", min_value=0.0, value=50000.0, step=1000.0)  # input credit limit
        AGE = st.number_input("Age", min_value=18, value=30, step=1)  # input customer age
        PAY_0 = st.number_input("Recent Payment Status PAY_0", min_value=-1, max_value=6, value=0, step=1)  # input recent payment status
        AVG_BILL_AMT = st.number_input("Average Bill Amount", min_value=0.0, value=10000.0, step=500.0)  # input average bill amount
        AVG_PAY_AMT = st.number_input("Average Payment Amount", min_value=0.0, value=5000.0, step=500.0)  # input average payment amount

    with num_col2:  # second numeric input column
        TOTAL_DELAY_MONTHS = st.number_input("Total Delay Months", min_value=0, max_value=6, value=0, step=1)  # input total delayed months
        MAX_DELAY = st.number_input("Maximum Delay", min_value=-1, max_value=6, value=0, step=1)  # input maximum delay
        AVG_UTILIZATION = st.slider("Average Credit Utilization", min_value=0.0, max_value=1.0, value=0.30, step=0.01)  # input utilization
        PAYMENT_RATIO = st.slider("Payment Ratio", min_value=0.0, max_value=2.0, value=0.50, step=0.01)  # input payment ratio

    st.markdown('</div>', unsafe_allow_html=True)  # close financial profile card

with right_col:  # right side for categorical features and action button
    st.markdown('<div class="section-card">', unsafe_allow_html=True)  # open categorical card
    st.markdown('<div class="section-title">Categorical Information</div>', unsafe_allow_html=True)  # add card title

    sex_choice = st.selectbox("Sex", ["Male", "Female"])  # input sex category
    education_choice = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])  # input education category
    marriage_choice = st.selectbox("Marriage", ["Married", "Single", "Others"])  # input marriage category

    st.markdown('</div>', unsafe_allow_html=True)  # close categorical card

    st.markdown('<div class="section-card">', unsafe_allow_html=True)  # open action card
    st.markdown('<div class="section-title">Run Prediction</div>', unsafe_allow_html=True)  # add action title
    predict_btn = st.button("Predict Default Risk")  # create prediction button
    st.markdown('</div>', unsafe_allow_html=True)  # close action card

SEX_2 = 1 if sex_choice == "Female" else 0  # encode female as 1 and male as 0
EDUCATION_2 = 1 if education_choice == "University" else 0  # encode university category
EDUCATION_3 = 1 if education_choice == "High School" else 0  # encode high school category
EDUCATION_4 = 1 if education_choice == "Others" else 0  # encode others education category
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0  # encode single marriage status
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0  # encode others marriage status

if predict_btn:  # run prediction only when button is clicked
    try:  # safely handle prediction logic
        input_data = {  # collect all input values into a dictionary
            "LIMIT_BAL": LIMIT_BAL,  # pass credit limit
            "AGE": AGE,  # pass age
            "PAY_0": PAY_0,  # pass recent payment status
            "AVG_BILL_AMT": AVG_BILL_AMT,  # pass average bill amount
            "AVG_PAY_AMT": AVG_PAY_AMT,  # pass average payment amount
            "TOTAL_DELAY_MONTHS": TOTAL_DELAY_MONTHS,  # pass total delay months
            "MAX_DELAY": MAX_DELAY,  # pass maximum delay
            "AVG_UTILIZATION": AVG_UTILIZATION,  # pass average utilization
            "PAYMENT_RATIO": PAYMENT_RATIO,  # pass payment ratio
            "SEX_2": SEX_2,  # pass encoded sex
            "EDUCATION_2": EDUCATION_2,  # pass encoded education university
            "EDUCATION_3": EDUCATION_3,  # pass encoded education high school
            "EDUCATION_4": EDUCATION_4,  # pass encoded education others
            "MARRIAGE_2": MARRIAGE_2,  # pass encoded marriage single
            "MARRIAGE_3": MARRIAGE_3  # pass encoded marriage others
        }

        input_df = prepare_input(input_data, feature_names)  # convert dictionary into aligned model input dataframe
        prediction = model.predict(input_df)[0]  # get class prediction from model
        probability = model.predict_proba(input_df)[0][1]  # get default probability for class 1

        st.markdown('<div class="section-card">', unsafe_allow_html=True)  # open results card
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)  # add result section title

        if probability < 0.30:  # check low risk range
            st.markdown(  # display low risk message
                f"""
                <div class="result-good">
                    Customer is not likely to default<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            interpretation = "This customer appears to have relatively low default risk."  # low risk explanation
        elif probability < 0.60:  # check moderate risk range
            st.markdown(  # display moderate risk message
                f"""
                <div class="result-medium">
                    Customer appears to have moderate default risk<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            interpretation = "This customer appears to have moderate default risk."  # moderate risk explanation
        else:  # handle high risk range
            st.markdown(  # display high risk message
                f"""
                <div class="result-high">
                    Customer is likely to default<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            interpretation = "This customer appears to have high default risk."  # high risk explanation

        st.subheader("Risk Interpretation")  # show interpretation heading
        st.write(interpretation)  # display interpretation text

        st.subheader("Input Summary")  # show summary heading
        st.dataframe(input_df, use_container_width=True)  # display aligned input dataframe

        if hasattr(model, "feature_importances_"):  # check whether the model supports feature importance
            importance_df = pd.DataFrame(  # create dataframe for feature importance
                {
                    "Feature": feature_names,  # use feature names
                    "Importance": model.feature_importances_  # use model importance scores
                }
            ).sort_values(by="Importance", ascending=False).head(10)  # sort and keep top 10

            st.subheader("Top 10 Important Features")  # show chart heading

            fig, ax = plt.subplots(figsize=(10, 5))  # create chart figure and axis
            ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])  # draw horizontal bar chart
            ax.set_title("Top 10 Important Features")  # set chart title
            ax.set_xlabel("Importance Score")  # set x axis label
            ax.set_ylabel("Feature")  # set y axis label
            fig.patch.set_facecolor("#111827")  # set figure background color
            ax.set_facecolor("#111827")  # set axis background color
            ax.tick_params(axis="x", colors="#e5e7eb")  # set x axis tick color
            ax.tick_params(axis="y", colors="#e5e7eb")  # set y axis tick color
            ax.xaxis.label.set_color("#e5e7eb")  # set x label color
            ax.yaxis.label.set_color("#e5e7eb")  # set y label color
            ax.title.set_color("#f9fafb")  # set title color
            for spine in ax.spines.values():  # loop through chart borders
                spine.set_color("#374151")  # set border color
            st.pyplot(fig)  # render chart in streamlit

        st.markdown('</div>', unsafe_allow_html=True)  # close results card

    except Exception as e:  # catch any prediction errors
        logger.error(f"Prediction error: {e}")  # write prediction error to log
        st.error(f"Prediction failed: {e}")  # show prediction error to user
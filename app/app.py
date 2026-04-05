import os  # used to handle file paths safely across systems
import streamlit as st  # main library for building the web app
import pandas as pd  # used for handling tabular data
import matplotlib.pyplot as plt  # used for plotting feature importance
from utils import load_model, prepare_input  # custom functions for model loading and input prep
from errorLog import setup_logger  # custom logger setup

logger = setup_logger()  # initialize logger for error tracking

# configure the page layout and metadata
st.set_page_config(
    page_title="Credit Card Default Prediction",  # title shown in browser tab
    page_icon="💳",  # icon shown in tab
    layout="wide",  # use full screen width
    initial_sidebar_state="collapsed"  # hide sidebar by default
)

# inject custom CSS styling to improve UI design
st.markdown("""
<style>
.main {
    background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

.hero-card {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);
    padding: 2rem;
    border-radius: 24px;
    color: white;
    box-shadow: 0 18px 45px rgba(37, 99, 235, 0.22);
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, #e0ecff 0%, #f0f7ff 100%);
    border-radius: 20px;
    padding: 1.2rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}

.section-card {
    background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
    border-radius: 22px;
    padding: 1.4rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    color: white;
    font-weight: 700;
    border-radius: 14px;
    padding: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# display header section with title and description
st.markdown("""
<div class="hero-card">
    <h2>Credit Card Default Prediction System</h2>
    <p>This application predicts whether a customer is likely to default using a trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

# create three columns for top metrics
m1, m2, m3 = st.columns(3)

# show model name
with m1:
    st.markdown("""
    <div class="metric-card">
        <p>Model</p>
        <h3>Random Forest</h3>
    </div>
    """, unsafe_allow_html=True)

# show accuracy
with m2:
    st.markdown("""
    <div class="metric-card">
        <p>Accuracy</p>
        <h3>0.82</h3>
    </div>
    """, unsafe_allow_html=True)

# show prediction target
with m3:
    st.markdown("""
    <div class="metric-card">
        <p>Target</p>
        <h3>Default Risk</h3>
    </div>
    """, unsafe_allow_html=True)

# get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# move one level up to project root
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# build correct paths to model files
model_path = os.path.join(ROOT_DIR, "model", "random_forest_model.pkl")
features_path = os.path.join(ROOT_DIR, "model", "model_features.pkl")

# safely load model and features
try:
    model, feature_names = load_model(model_path, features_path)
except Exception as e:
    logger.error(f"Error loading model: {e}")  # log error
    st.error("Model loading failed")  # show error to user
    st.stop()  # stop execution

# create layout with two columns
left, right = st.columns([1.2, 1])

# left column for numeric inputs
with left:
    st.subheader("Customer Financial Profile")

    # create sub columns for better layout
    c1, c2 = st.columns(2)

    # first column inputs
    with c1:
        LIMIT_BAL = st.number_input("Credit Limit", value=50000.0)
        AGE = st.number_input("Age", value=30)
        PAY_0 = st.number_input("Recent Payment Status", value=0)
        AVG_BILL_AMT = st.number_input("Average Bill Amount", value=10000.0)
        AVG_PAY_AMT = st.number_input("Average Payment Amount", value=5000.0)

    # second column inputs
    with c2:
        TOTAL_DELAY_MONTHS = st.number_input("Total Delay Months", value=0)
        MAX_DELAY = st.number_input("Maximum Delay", value=0)
        AVG_UTILIZATION = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
        PAYMENT_RATIO = st.slider("Payment Ratio", 0.0, 2.0, 0.5)

    st.markdown('</div>', unsafe_allow_html=True)

# right column for categorical inputs and button
with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Categorical Information")

    # dropdown for sex
    sex_choice = st.selectbox("Sex", ["Male", "Female"])

    # dropdown for education
    education_choice = st.selectbox(
        "Education",
        ["Graduate School", "University", "High School", "Others"]
    )

    # dropdown for marriage
    marriage_choice = st.selectbox(
        "Marriage",
        ["Married", "Single", "Others"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # prediction button
    predict_btn = st.button("Predict Default Risk")

# encode categorical values into numeric format
SEX_2 = 1 if sex_choice == "Female" else 0
EDUCATION_2 = 1 if education_choice == "University" else 0
EDUCATION_3 = 1 if education_choice == "High School" else 0
EDUCATION_4 = 1 if education_choice == "Others" else 0
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0

# run prediction when button is clicked
if predict_btn:
    try:
        # collect all inputs into dictionary
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

        # convert input into model-compatible format
        input_df = prepare_input(input_data, feature_names)

        # get prediction result
        prediction = model.predict(input_df)[0]

        # get probability of default
        probability = model.predict_proba(input_df)[0][1]

        # display results
        st.subheader("Prediction Result")
        st.success(f"Default probability: {probability:.2%}")

        # show input summary table
        st.subheader("Input Summary")
        st.dataframe(input_df)

        # show feature importance if available
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)

            st.subheader("Top 10 Important Features")

            fig, ax = plt.subplots()
            ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
            st.pyplot(fig)

    except Exception as e:
        logger.error(f"Prediction error: {e}")  # log error
        st.error("Prediction failed")  # show error message
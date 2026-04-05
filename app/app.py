import os  # Handles file and directory paths
import streamlit as st  # Builds the web app interface
import pandas as pd  # Works with tabular data
import matplotlib.pyplot as plt  # Creates visualizations
from utils import load_model, prepare_input  # Custom helper functions
from errorLog import setup_logger  # Logging utility

logger = setup_logger()  # Initialize application logger


# Configure Streamlit page settings
st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Apply custom CSS styling for appearance
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #071224 0%, #0e2749 55%, #163b66 100%);
        color: #e5e7eb;
    }
    header[data-testid="stHeader"] { display: none; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)


# Display application title and description
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Credit Card Default Prediction System</div>
        <p class="hero-subtitle">
            Predict whether a customer is likely to default using a trained Random Forest model.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Determine project directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))  # Current file location
root_dir = os.path.abspath(os.path.join(base_dir, ".."))  # Project root directory

# Define model and feature file paths
model_path = os.path.join(root_dir, "model", "random_forest_model.pkl")
features_path = os.path.join(root_dir, "model", "model_features.pkl")


# Load trained model and feature names safely
try:
    model, feature_names = load_model(model_path, features_path)
except Exception as e:
    logger.error(f"Error loading model or feature names: {e}")
    st.error(f"Model loading failed: {e}")
    st.stop()


# Create layout with two columns
top_left, top_right = st.columns([1.8, 1.15], gap="large")


# Input section for user data
with top_left:
    st.markdown('<div class="card-title">Customer Information</div>', unsafe_allow_html=True)

    # Create two columns for numeric inputs
    left_num, right_num = st.columns(2)

    # Left column inputs
    with left_num:
        LIMIT_BAL = st.number_input("Credit Limit", min_value=0.0, value=50000.0, step=1000.0)
        AGE = st.number_input("Age", min_value=18, value=30, step=1)
        PAY_0 = st.number_input("Recent Payment Status PAY_0", min_value=-1, max_value=6, value=0)
        AVG_BILL_AMT = st.number_input("Average Bill Amount", min_value=0.0, value=10000.0)
        AVG_PAY_AMT = st.number_input("Average Payment Amount", min_value=0.0, value=5000.0)

    # Right column inputs
    with right_num:
        TOTAL_DELAY_MONTHS = st.number_input("Total Delay Months", min_value=0, max_value=6, value=0)
        MAX_DELAY = st.number_input("Maximum Delay", min_value=-1, max_value=6, value=0)
        AVG_UTILIZATION = st.slider("Average Credit Utilization", 0.0, 1.0, 0.30)
        PAYMENT_RATIO = st.slider("Payment Ratio", 0.0, 2.0, 0.50)

    # Categorical selections
    sex_choice = st.selectbox("Sex", ["Male", "Female"])
    education_choice = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
    marriage_choice = st.selectbox("Marriage", ["Married", "Single", "Others"])

    predict_btn = st.button("Predict Default Risk")  # Button to trigger prediction


# Display model information on the right side
with top_right:
    st.markdown("### Model: Random Forest")
    st.markdown("### Accuracy: 0.82")
    st.markdown("### Target: Default Risk")


# Encode categorical variables into numeric format required by the model
SEX_2 = 1 if sex_choice == "Female" else 0
EDUCATION_2 = 1 if education_choice == "University" else 0
EDUCATION_3 = 1 if education_choice == "High School" else 0
EDUCATION_4 = 1 if education_choice == "Others" else 0
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0


# Perform prediction when user clicks the button
if predict_btn:
    try:
        # Collect all user inputs into a dictionary
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

        # Convert input data into a DataFrame aligned with model features
        input_df = prepare_input(input_data, feature_names)

        # Generate prediction and probability score
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("## Prediction Result")

        # Display result based on probability thresholds
        if probability < 0.30:
            st.success(f"Customer is unlikely to default. Probability: {probability:.2%}")
            interpretation = "Low default risk"
        elif probability < 0.60:
            st.warning(f"Customer has moderate default risk. Probability: {probability:.2%}")
            interpretation = "Moderate default risk"
        else:
            st.error(f"Customer is likely to default. Probability: {probability:.2%}")
            interpretation = "High default risk"

        # Show interpretation text
        st.subheader("Risk Interpretation")
        st.write(interpretation)

        # Display formatted input data
        st.subheader("Input Summary")
        styled_df = input_df.style.set_properties(
            **{
                'background-color': '#0b1730',
                'color': '#e5e7eb'
            }
        )
        st.dataframe(styled_df, use_container_width=True)

        # Plot feature importance if available in model
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)

            st.subheader("Top 10 Important Features")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(importance_df["Feature"], importance_df["Importance"])
            ax.invert_yaxis()

            ax.set_title("Top 10 Important Features")
            ax.set_xlabel("Importance Score")
            ax.set_ylabel("Feature")

            st.pyplot(fig)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"Prediction failed: {e}")
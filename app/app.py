# import os to build safe file paths
import os

# import streamlit to create the web app
import streamlit as st

# import pandas to handle tabular data
import pandas as pd

# import matplotlib for the feature importance chart
import matplotlib.pyplot as plt

# import helper functions from utils.py
from utils import load_model, prepare_input

# import logger setup from errorLog.py
from errorLog import setup_logger


# create logger object
logger = setup_logger()


# configure page settings
st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")


# show app title
st.title("Credit Card Default Prediction System")

# show short project description
st.write(
    "This application predicts whether a customer is likely to default on credit card payment "
    "using the trained Random Forest model."
)

# show model information
st.subheader("Model Information")
st.write("Selected model: Random Forest")
st.write("Model accuracy: 0.82")


# get the folder where this app.py file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# go one level up to the project root folder
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# create full path for the saved model file
model_path = os.path.join(project_root, "model", "random_forest_model.pkl")

# create full path for the saved feature names file
features_path = os.path.join(project_root, "model", "model_features.pkl")


# load model and feature names safely
try:
    model, feature_names = load_model(model_path, features_path)

except Exception as e:
    logger.error(f"Error loading model or feature names: {e}")
    st.error("Error loading model files")
    st.write("Model path:", model_path)
    st.write("Features path:", features_path)
    st.write("Model exists:", os.path.exists(model_path))
    st.write("Features exist:", os.path.exists(features_path))
    st.stop()


# create input section title
st.subheader("Enter Customer Information")


# create two columns for numeric input fields
col1, col2 = st.columns(2)


# first column inputs
with col1:
    LIMIT_BAL = st.number_input("Credit Limit", min_value=0.0, value=50000.0)
    AGE = st.number_input("Age", min_value=18, value=30)
    PAY_0 = st.number_input("Recent Payment Status PAY_0 (-1 to 6)", min_value=-1, max_value=6, value=0)
    AVG_BILL_AMT = st.number_input("Average Bill Amount", min_value=0.0, value=10000.0)
    AVG_PAY_AMT = st.number_input("Average Payment Amount", min_value=0.0, value=5000.0)

# second column inputs
with col2:
    TOTAL_DELAY_MONTHS = st.number_input("Total Delay Months", min_value=0, max_value=6, value=0)
    MAX_DELAY = st.number_input("Maximum Delay", min_value=-1, max_value=6, value=0)
    AVG_UTILIZATION = st.number_input(
        "Average Credit Utilization (0 to 1, where 0.3 means 30%)",
        min_value=0.0,
        max_value=1.0,
        value=0.3
    )
    PAYMENT_RATIO = st.number_input(
        "Payment Ratio (0 to 2)",
        min_value=0.0,
        max_value=2.0,
        value=0.5
    )


# create section for categorical variables
st.subheader("Categorical Information")


# create three columns for categorical inputs
col3, col4, col5 = st.columns(3)


# sex input
with col3:
    sex_choice = st.selectbox("Sex", ["Male", "Female"])

# education input
with col4:
    education_choice = st.selectbox(
        "Education",
        ["Graduate School", "University", "High School", "Others"]
    )

# marriage input
with col5:
    marriage_choice = st.selectbox(
        "Marriage",
        ["Married", "Single", "Others"]
    )


# convert sex into encoded value
SEX_2 = 1 if sex_choice == "Female" else 0

# convert education into encoded values
EDUCATION_2 = 1 if education_choice == "University" else 0
EDUCATION_3 = 1 if education_choice == "High School" else 0
EDUCATION_4 = 1 if education_choice == "Others" else 0

# convert marriage into encoded values
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0


# create prediction button
if st.button("Predict Default Risk"):
    try:
        # collect all inputs into a dictionary
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

        # prepare input in the same format as training features
        input_df = prepare_input(input_data, feature_names)

        # generate class prediction
        prediction = model.predict(input_df)[0]

        # generate probability for default class
        probability = model.predict_proba(input_df)[0][1]

        # show result section
        st.subheader("Prediction Result")

        # display prediction output
        if prediction == 1:
            st.error("Customer is likely to default")
        else:
            st.success("Customer is not likely to default")

        # display default probability
        st.write(f"Default probability: {probability:.2%}")

        # show risk interpretation
        st.subheader("Risk Interpretation")

        if probability < 0.30:
            st.write("This customer appears to have low default risk.")
        elif probability < 0.60:
            st.write("This customer appears to have moderate default risk.")
        else:
            st.write("This customer appears to have high default risk.")

        # show user input summary
        st.subheader("Input Summary")
        st.dataframe(input_df)

        # create dataframe for feature importance
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        # sort and keep top 10 important features
        importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

        # display feature importance section
        st.subheader("Top 10 Important Features")

        # create chart
        fig, ax = plt.subplots()
        ax.bar(importance_df["Feature"], importance_df["Importance"])
        ax.set_title("Top 10 Important Features")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # show chart in streamlit
        st.pyplot(fig)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error("An error occurred during prediction")

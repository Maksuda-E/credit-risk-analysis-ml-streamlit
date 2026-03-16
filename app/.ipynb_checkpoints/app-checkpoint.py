# import streamlit to build the web application
import streamlit as st

# import pandas to organize input data
import pandas as pd

# import matplotlib to create the feature importance chart
import matplotlib.pyplot as plt

# import custom functions for loading model and preparing input
from utils import load_model, prepare_input

# import custom logger setup
from errorLog import setup_logger


# create logger object for saving errors into log file
logger = setup_logger()


# set page title in browser tab
st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")


# show application title
st.title("Credit Card Default Prediction System")

# show short description of the project
st.write(
    "This application predicts whether a customer is likely to default on credit card payment "
    "using the trained Random Forest model."
)

# show short model information
st.subheader("Model Information")
st.write("Selected model: Random Forest")
st.write("Model accuracy: 0.82")


# load model and feature names
try:
    # load saved model and training feature names
    model, feature_names = load_model(
        "../model/random_forest_model.pkl",
        "../model/model_features.pkl"
    )

except Exception as e:
    # save error into log file
    logger.error(f"Error loading model or feature names: {e}")

    # show error in app
    st.error("Error loading model files")

    # stop application if model cannot load
    st.stop()


# create input section title
st.subheader("Enter Customer Information")


# create two columns to make the form cleaner
col1, col2 = st.columns(2)


# first column inputs
with col1:
    # collect customer credit limit
    LIMIT_BAL = st.number_input("Credit Limit", min_value=0.0, value=50000.0)

    # collect customer age
    AGE = st.number_input("Age", min_value=18, value=30)

    # collect most recent payment status
    PAY_0 = st.number_input("Recent Payment Status PAY_0", value=0)

    # collect average bill amount
    AVG_BILL_AMT = st.number_input("Average Bill Amount", min_value=0.0, value=10000.0)

    # collect average payment amount
    AVG_PAY_AMT = st.number_input("Average Payment Amount", min_value=0.0, value=5000.0)


# second column inputs
with col2:
    # collect number of delayed months
    TOTAL_DELAY_MONTHS = st.number_input("Total Delay Months", min_value=0, value=0)

    # collect maximum delay value
    MAX_DELAY = st.number_input("Maximum Delay", value=0)

    # collect credit utilization ratio
    AVG_UTILIZATION = st.number_input(
        "Average Credit Utilization (0 to 1, where 0.3 means 30%)",
        min_value=0.0,
        max_value=1.0,
        value=0.3
    )

    # collect payment ratio
    PAYMENT_RATIO = st.number_input(
        "Payment Ratio",
        min_value=0.0,
        value=0.5
    )


# create section for encoded categorical variables
st.subheader("Categorical Information")


# create three columns for cleaner layout
col3, col4, col5 = st.columns(3)


with col3:
    # sex input after one hot encoding
    SEX_2 = st.selectbox("Sex", ["Male", "Female"])

with col4:
    # education input using user friendly labels
    education_choice = st.selectbox(
        "Education",
        ["Graduate School", "University", "High School", "Others"]
    )

with col5:
    # marriage input using user friendly labels
    marriage_choice = st.selectbox(
        "Marriage",
        ["Married", "Single", "Others"]
    )


# convert sex input into encoded value
SEX_2 = 1 if SEX_2 == "Female" else 0

# convert education input into encoded values
EDUCATION_2 = 1 if education_choice == "University" else 0
EDUCATION_3 = 1 if education_choice == "High School" else 0
EDUCATION_4 = 1 if education_choice == "Others" else 0

# convert marriage input into encoded values
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0


# create prediction button
if st.button("Predict Default Risk"):

    try:
        # store user inputs in dictionary form
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

        # prepare input in the same feature order as training data
        input_df = prepare_input(input_data, feature_names)

        # generate predicted class
        prediction = model.predict(input_df)[0]

        # generate probability of default class
        probability = model.predict_proba(input_df)[0][1]

        # show prediction result section
        st.subheader("Prediction Result")

        # display prediction result
        if prediction == 1:
            st.error("Customer is likely to default")
        else:
            st.success("Customer is not likely to default")

        # display probability value
        st.write(f"Default probability: {probability:.2%}")

        # show simple risk interpretation
        st.subheader("Risk Interpretation")

        if probability < 0.30:
            st.write("This customer appears to have low default risk.")
        elif probability < 0.60:
            st.write("This customer appears to have moderate default risk.")
        else:
            st.write("This customer appears to have high default risk.")

        # show input summary section
        st.subheader("Input Summary")
        st.dataframe(input_df)

        # create feature importance dataframe from trained model
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        # sort feature importance values from highest to lowest
        importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

        # show feature importance section
        st.subheader("Top 10 Important Features")

        # create figure for bar chart
        fig, ax = plt.subplots()

        # plot top 10 feature importances
        ax.bar(importance_df["Feature"], importance_df["Importance"])

        # set chart title
        ax.set_title("Top 10 Important Features")

        # rotate x axis labels for better readability
        plt.xticks(rotation=45)

        # adjust layout so labels are not cut off
        plt.tight_layout()

        # display chart in streamlit
        st.pyplot(fig)

    except Exception as e:
        # log prediction related errors
        logger.error(f"Prediction error: {e}")

        # show error in app
        st.error("An error occurred during prediction")
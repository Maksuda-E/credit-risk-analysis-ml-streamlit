# import os for working with file and folder paths
import os

# import json for storing structured input and output records
import json

# import joblib to optionally load a saved scaler
import joblib

# import streamlit for the web application
import streamlit as st

# import pandas for dataframe handling and csv saving
import pandas as pd

# import matplotlib for feature importance plotting
import matplotlib.pyplot as plt

# import datetime for timestamped saved records
from datetime import datetime

# import helper functions from utils
from utils import load_model, prepare_input

# import project logger setup
from errorLog import setup_logger


# create the logger instance
logger = setup_logger()

# configure page settings for streamlit
st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# build base directory for the current app folder
base_dir = os.path.dirname(os.path.abspath(__file__))

# build project root directory
root_dir = os.path.abspath(os.path.join(base_dir, ".."))

# build model file path
model_path = os.path.join(root_dir, "model", "random_forest_model.pkl")

# build features file path
features_path = os.path.join(root_dir, "model", "model_features.pkl")

# build optional scaler file path
scaler_path = os.path.join(root_dir, "model", "scaler.pkl")

# build logs directory path
logs_dir = os.path.join(root_dir, "logs")

# create logs directory if it does not exist
os.makedirs(logs_dir, exist_ok=True)

# build prediction csv file path
prediction_csv_path = os.path.join(logs_dir, "prediction_records.csv")


# define a helper function to load scaler safely if it exists
def load_scaler_if_available(path):
    # check if scaler file exists
    if os.path.exists(path):
        try:
            # load scaler from disk
            scaler_obj = joblib.load(path)

            # write log that scaler was loaded
            logger.info(f"Scaler loaded successfully from {path}")

            # return loaded scaler
            return scaler_obj

        except Exception as e:
            # log scaler loading failure
            logger.error(f"Failed to load scaler from {path}: {e}")

            # show warning in app without stopping the application
            st.warning("Scaler file exists but could not be loaded. Prediction will continue without scaling.")

            # return none when scaler loading fails
            return None

    # write log if scaler file does not exist
    logger.warning(f"Scaler file not found at {path}. Prediction will continue without scaling.")

    # return none if scaler file is missing
    return None


# define a helper function to apply scaling only when scaler exists
def apply_scaler_if_available(input_df, scaler_obj):
    # check whether scaler is available
    if scaler_obj is not None:
        try:
            # transform the input dataframe using the fitted scaler
            scaled_array = scaler_obj.transform(input_df)

            # convert scaled array back to dataframe with original column names
            scaled_df = pd.DataFrame(scaled_array, columns=input_df.columns)

            # return scaled dataframe
            return scaled_df

        except Exception as e:
            # log scaling failure
            logger.error(f"Scaling failed: {e}")

            # show a warning in the app
            st.warning("Scaling failed during prediction. Using unscaled input instead.")

            # return original dataframe if scaling fails
            return input_df

    # return original dataframe if scaler is not available
    return input_df


# define a helper function to save prediction records to csv
def save_prediction_record(record_dict, csv_path):
    # convert dictionary to one row dataframe
    record_df = pd.DataFrame([record_dict])

    # check whether the csv file already exists
    if os.path.exists(csv_path):
        # append without header if file exists
        record_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        # write with header if file does not exist
        record_df.to_csv(csv_path, index=False)


# define a helper function to compute risk label from probability
def get_risk_label(probability_value):
    # return low risk if probability is below 0.30
    if probability_value < 0.30:
        return "Low"

    # return moderate risk if probability is below 0.60
    if probability_value < 0.60:
        return "Moderate"

    # return high risk otherwise
    return "High"


# add custom css for the application ui
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #06111f 0%, #0c1f37 50%, #132b49 100%);
        color: #e5e7eb;
    }

    header[data-testid="stHeader"] {
        display: none;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #e5e7eb;
    }

    .hero-card {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 45%, #38bdf8 100%);
        border-radius: 22px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.30);
        margin-bottom: 1.3rem;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.4rem;
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
        font-size: 1.8rem;
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
        padding: 1.15rem 1rem;
        text-align: center;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
        margin-bottom: 1rem;
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
        background-color: #08162f !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
    }

    input {
        color: #e5e7eb !important;
    }

    .summary-table-card {
        background: linear-gradient(180deg, rgba(8, 22, 47, 0.96) 0%, rgba(12, 30, 58, 0.96) 100%);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
        margin-top: 0.5rem;
        overflow-x: auto;
    }

    .summary-table-header {
        font-size: 1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.85rem;
        letter-spacing: 0.2px;
    }

    .summary-table {
        width: 100%;
        border-collapse: collapse;
        overflow: hidden;
        border-radius: 14px;
    }

    .summary-table thead tr {
        background: linear-gradient(135deg, #1d4ed8 0%, #38bdf8 100%);
    }

    .summary-table th {
        color: white;
        text-align: left;
        padding: 0.85rem 1rem;
        font-size: 0.92rem;
        font-weight: 700;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    .summary-table td {
        padding: 0.8rem 1rem;
        font-size: 0.92rem;
        color: #e5e7eb;
        border-bottom: 1px solid rgba(148, 163, 184, 0.10);
    }

    .summary-table tbody tr:nth-child(odd) {
        background: rgba(15, 23, 42, 0.72);
    }

    .summary-table tbody tr:nth-child(even) {
        background: rgba(30, 41, 59, 0.52);
    }

    .summary-table tbody tr:hover {
        background: rgba(37, 99, 235, 0.18);
        transition: 0.2s ease;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    ul[role="listbox"],
    ul[role="listbox"] *,
    li[role="option"],
    li[role="option"] * {
        color: black !important;
    }

    ul[role="listbox"],
    li[role="option"] {
        background: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# render the top hero card
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

# show model performance heading
st.markdown("### Model Performance")

# create four columns for performance cards
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4, gap="medium")

# render accuracy card
with perf_col1:
    st.markdown(
        '<div class="perf-card"><div class="perf-label">Accuracy</div><div class="perf-value">0.82</div></div>',
        unsafe_allow_html=True
    )

# render precision card
with perf_col2:
    st.markdown(
        '<div class="perf-card"><div class="perf-label">Precision</div><div class="perf-value">0.80</div></div>',
        unsafe_allow_html=True
    )

# render recall card
with perf_col3:
    st.markdown(
        '<div class="perf-card"><div class="perf-label">Recall</div><div class="perf-value">0.78</div></div>',
        unsafe_allow_html=True
    )

# render f1 card
with perf_col4:
    st.markdown(
        '<div class="perf-card"><div class="perf-label">F1 Score</div><div class="perf-value">0.79</div></div>',
        unsafe_allow_html=True
    )

# add a small vertical gap after the performance cards
st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


# cache model loading for performance
@st.cache_resource
def get_model_and_assets(model_path_arg, features_path_arg, scaler_path_arg):
    model, feature_names, scaler = load_model(model_path_arg, features_path_arg, scaler_path_arg)
    return model, feature_names, scaler

model, feature_names, scaler = get_model_and_assets(model_path, features_path, scaler_path)

# try to load model, features, and scaler safely
try:
    # load model assets
    model, feature_names, scaler = get_model_and_assets(model_path, features_path, scaler_path)

    # log success after model assets are loaded
    logger.info("Model and feature assets loaded successfully")

except Exception as e:
    # log model loading error
    logger.error(f"Error loading model assets: {e}")

    # show error to the user
    st.error(f"Model loading failed: {e}")

    # stop the app because prediction cannot continue without the model
    st.stop()

# create main layout columns for input section and right summary section
main_left, main_right = st.columns([1.8, 1.0], gap="large")

# create left section for inputs
with main_left:
    # open a panel container visually
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # show input section title
    st.markdown('<div class="card-title">Customer Information</div>', unsafe_allow_html=True)

    # create two columns for numeric input widgets
    left_num_col, right_num_col = st.columns(2, gap="medium")

    # add left column numeric inputs
    with left_num_col:
        # collect credit limit input
        LIMIT_BAL = st.number_input("Credit Limit", min_value=10000.0, value=50000.0, step=1000.0)

        # collect age input
        AGE = st.number_input("Age", min_value=18, value=30, step=1)

        # collect recent payment status input
        PAY_0 = st.number_input("Recent Payment Status PAY_0", min_value=-2, max_value=8, value=0, step=1)

        # collect average bill amount input
        AVG_BILL_AMT = st.number_input("Average Bill Amount", min_value=0.0, value=10000.0, step=500.0)

        # collect average payment amount input
        AVG_PAY_AMT = st.number_input("Average Payment Amount", min_value=0.0, value=5000.0, step=500.0)

    # add right column numeric inputs
    with right_num_col:
        # collect total delay months input
        TOTAL_DELAY_MONTHS = st.number_input("Total Delay Months", min_value=0, max_value=6, value=0, step=1)

        # collect maximum delay input
        MAX_DELAY = st.number_input("Maximum Delay", min_value=-2, max_value=8, value=0, step=1)

        # collect average utilization input
        AVG_UTILIZATION = st.slider("Average Credit Utilization", min_value=0.0, max_value=1.0, value=0.30, step=0.01)

        # collect payment ratio input with a safer range
        PAYMENT_RATIO = st.slider("Payment Ratio", min_value=0.0, max_value=1.2, value=0.50, step=0.01)

    # show categorical section title
    st.markdown('<div class="section-heading">Categorical Information</div>', unsafe_allow_html=True)

    # create three columns for select boxes
    sex_col, education_col, marriage_col = st.columns(3, gap="medium")

    # create sex select box
    with sex_col:
        sex_choice = st.selectbox("Sex", ["Male", "Female"])

    # create education select box
    with education_col:
        education_choice = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])

    # create marriage select box
    with marriage_col:
        marriage_choice = st.selectbox("Marriage", ["Married", "Single", "Others"])

    # add predict button
    predict_btn = st.button("Predict Default Risk")

    # close the visual panel container
    st.markdown('</div>', unsafe_allow_html=True)

# encode sex for model input
SEX_2 = 1 if sex_choice == "Female" else 0

# encode education university for model input
EDUCATION_2 = 1 if education_choice == "University" else 0

# encode education high school for model input
EDUCATION_3 = 1 if education_choice == "High School" else 0

# encode education others for model input
EDUCATION_4 = 1 if education_choice == "Others" else 0

# encode marriage single for model input
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0

# encode marriage others for model input
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0

# create right side section for model information and prediction result
with main_right:
    # open panel styling
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # render model info card
    st.markdown(
        """
        <div class="side-card">
            <div class="side-label">Model</div>
            <div class="side-value">Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # render target info card
    st.markdown(
        """
        <div class="side-card">
            <div class="side-label">Target</div>
            <div class="side-value">Default Risk</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # create placeholder for result box
    result_placeholder = st.empty()

    # create placeholder for support note
    note_placeholder = st.empty()

    # create placeholder for interpretation text
    interpretation_placeholder = st.empty()

    # create placeholder for quick debug info
    debug_placeholder = st.empty()

    # close panel styling
    st.markdown('</div>', unsafe_allow_html=True)

# run prediction only after clicking the button
if predict_btn:
    try:
        # create raw input dictionary based on the ui fields
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

        # log the raw input before preparation
        logger.info(f"Raw input received: {input_data}")

        # prepare input using utils.py to match training feature order
        prepared_input_df = prepare_input(input_data, feature_names)

        # log prepared input dataframe before scaling
        logger.info(f"Prepared input before scaling: {prepared_input_df.to_dict(orient='records')}")

        # apply scaler when available
        model_input_df = apply_scaler_if_available(prepared_input_df, scaler)

        # log final model input dataframe after scaling or no scaling
        logger.info(f"Final model input used for prediction: {model_input_df.to_dict(orient='records')}")

        # generate predicted class
        prediction = model.predict(model_input_df)[0]

        # generate class probability for default class
        probability = model.predict_proba(model_input_df)[0][1]

        # compute risk label from probability
        risk_label = get_risk_label(probability)

        # log prediction results
        logger.info(f"Prediction generated: prediction={prediction}, probability={probability:.6f}, risk_label={risk_label}")

        # render low risk result box when low
        if risk_label == "Low":
            result_placeholder.markdown(
                f"""
                <div class="result-box result-good">
                    Customer appears to have low default risk<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

            # set interpretation text for low risk
            interpretation_text = "This customer appears to have relatively low default risk."

        # render moderate risk result box when moderate
        elif risk_label == "Moderate":
            result_placeholder.markdown(
                f"""
                <div class="result-box result-medium">
                    Customer appears to have moderate default risk<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

            # set interpretation text for moderate risk
            interpretation_text = "The customer shows some financial risk signals and should be monitored carefully."

        # render high risk result box when high
        else:
            result_placeholder.markdown(
                f"""
                <div class="result-box result-high">
                    Customer appears to have high default risk<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

            # set interpretation text for high risk
            interpretation_text = "This customer appears to have high default risk and may require immediate attention."

        # render note under result
        note_placeholder.info("Probability reflects likelihood, not certainty. Use this result as decision support.")

        # render interpretation text
        interpretation_placeholder.markdown(f"**Risk Interpretation:** {interpretation_text}")

        # render small debug section so you can verify model behavior
        debug_placeholder.markdown(
            f"""
            **Debug Details**  
            Raw probability: `{probability:.6f}`  
            Predicted class: `{int(prediction)}`  
            Risk label: `{risk_label}`
            """
        )

        # build record for persistent logging and csv saving
        prediction_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_label": risk_label,
            "raw_input_json": json.dumps(input_data),
            "prepared_input_json": prepared_input_df.to_json(orient="records"),
            "model_input_json": model_input_df.to_json(orient="records")
        }

        # save prediction record into csv
        save_prediction_record(prediction_record, prediction_csv_path)

        # log that prediction record was saved successfully
        logger.info(f"Prediction record saved to {prediction_csv_path}")

        # add vertical space before detailed outputs
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # show key factors heading
        st.markdown("### Key Factors Influencing Risk")

        # show explanation text
        st.write("Important factors affecting this prediction include payment delays, credit utilization level, and payment-to-bill ratio.")

        # show input summary heading
        st.markdown("### Input Summary")

        # transpose the prepared input for vertical summary display
        summary_df = prepared_input_df.T.reset_index()

        # rename summary columns
        summary_df.columns = ["Feature", "Value"]

        # build html rows for the summary table
        table_rows = "".join(
            f"""
            <tr>
                <td>{row['Feature']}</td>
                <td>{row['Value']}</td>
            </tr>
            """
            for _, row in summary_df.iterrows()
        )

        # render the styled summary table
        st.markdown(
            f"""
            <div class="summary-table-card">
                <div class="summary-table-header">Customer Input Overview</div>
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        # check whether model supports feature importance
        if hasattr(model, "feature_importances_"):
            # create dataframe for feature importances
            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": model.feature_importances_
                }
            ).sort_values(by="Importance", ascending=False).head(10)

            # show heading for importance chart
            st.markdown("### Top 10 Important Features")

            # create figure and axes for the importance plot
            fig, ax = plt.subplots(figsize=(10, 5))

            # plot horizontal bar chart
            ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])

            # set plot title
            ax.set_title("Top 10 Important Features")

            # set x axis label
            ax.set_xlabel("Importance Score")

            # set y axis label
            ax.set_ylabel("Feature")

            # style figure background
            fig.patch.set_facecolor("#0f172a")

            # style axes background
            ax.set_facecolor("#0f172a")

            # style x axis ticks
            ax.tick_params(axis="x", colors="#e5e7eb")

            # style y axis ticks
            ax.tick_params(axis="y", colors="#e5e7eb")

            # style x axis label color
            ax.xaxis.label.set_color("#e5e7eb")

            # style y axis label color
            ax.yaxis.label.set_color("#e5e7eb")

            # style title color
            ax.title.set_color("#f8fafc")

            # style chart borders
            for spine in ax.spines.values():
                spine.set_color("#334155")

            # display chart in streamlit
            st.pyplot(fig)

            # add note under the chart
            st.write("These features represent the global importance learned by the trained model.")

        # create an expandable debug section for full raw records
        with st.expander("Show Debug Input Records"):
            # show raw input dictionary
            st.write("Raw Input Dictionary")
            st.json(input_data)

            # show prepared dataframe before scaling
            st.write("Prepared Input Before Scaling")
            st.dataframe(prepared_input_df, use_container_width=True)

            # show final model input dataframe used for prediction
            st.write("Final Model Input Used For Prediction")
            st.dataframe(model_input_df, use_container_width=True)

    except Exception as e:
        # log prediction error details
        logger.error(f"Prediction error: {e}")

        # show prediction failure in the app
        st.error(f"Prediction failed: {e}")
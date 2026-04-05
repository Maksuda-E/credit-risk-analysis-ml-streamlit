import os  # import os to work with project file paths
import streamlit as st  # import streamlit to build the web application
import pandas as pd  # import pandas for data handling
import matplotlib.pyplot as plt  # import matplotlib for charts
from utils import load_model, prepare_input, save_prediction_record  # import helper functions from utils
from errorLog import setup_logger  # import the logger setup function

logger = setup_logger()  # create the application logger

st.set_page_config(  # configure the main page settings
    page_title="Credit Card Default Prediction",  # set the browser tab title
    layout="wide",  # use wide layout for a dashboard-style page
    initial_sidebar_state="collapsed"  # keep the sidebar collapsed by default
)

st.markdown(  # inject custom css styles for the app theme and layout
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
        margin-bottom: 1rem;
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

    .side-card {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #38bdf8 100%);
        border-radius: 22px;
        padding: 1rem 0.5rem;
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

    input {
        color: #e5e7eb !important;
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

st.markdown(  # render the main hero section
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

st.markdown("### Model Performance")  # show performance section title

p1, p2, p3, p4 = st.columns(4, gap="medium")  # create four columns for performance cards
with p1:  # place accuracy card in first column
    st.markdown('<div class="perf-card"><div class="perf-label">Accuracy</div><div class="perf-value">0.82</div></div>', unsafe_allow_html=True)
with p2:  # place precision card in second column
    st.markdown('<div class="perf-card"><div class="perf-label">Precision</div><div class="perf-value">0.80</div></div>', unsafe_allow_html=True)
with p3:  # place recall card in third column
    st.markdown('<div class="perf-card"><div class="perf-label">Recall</div><div class="perf-value">0.78</div></div>', unsafe_allow_html=True)
with p4:  # place f1 score card in fourth column
    st.markdown('<div class="perf-card"><div class="perf-label">F1 Score</div><div class="perf-value">0.79</div></div>', unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.abspath(__file__))  # get the absolute path of the current app folder
root_dir = os.path.dirname(base_dir)  # move one level up to the project root folder

model_path = os.path.join(root_dir, "model", "random_forest_model.pkl")  # build the trained model path
features_path = os.path.join(root_dir, "model", "model_features.pkl")  # build the feature names file path
scaler_path = os.path.join(root_dir, "model", "scaler.pkl")  # build the scaler file path

@st.cache_resource  # cache model loading so files are not reloaded every rerun
def get_model_artifacts():
    return load_model(model_path, features_path, scaler_path)  # load and return model, features, and scaler

try:  # try to load the required model artifacts
    model, feature_names, scaler = get_model_artifacts()  # unpack model artifacts
except Exception as e:  # catch any loading failure
    logger.error(f"Model loading failed: {e}")  # write the loading error to the log file
    st.error(f"Model loading failed: {e}")  # show the error in the app
    st.stop()  # stop the app because prediction cannot continue without the artifacts

main_left, main_right = st.columns([1.8, 1.0], gap="large")  # create the main two-column dashboard layout

with main_left:  # begin left panel for user inputs
    st.markdown('<div class="panel">', unsafe_allow_html=True)  # open styled panel
    st.markdown('<div class="card-title">Customer Information</div>', unsafe_allow_html=True)  # show panel title

    left_col, right_col = st.columns(2, gap="medium")  # create two columns inside the input section

    with left_col:  # begin left input column
        LIMIT_BAL = st.number_input("Credit Limit", min_value=10000.0, value=50000.0, step=1000.0)  # collect credit limit
        AGE = st.number_input("Age", min_value=18, value=30, step=1)  # collect customer age
        PAY_0 = st.number_input("Recent Payment Status PAY_0", min_value=-2, max_value=8, value=0, step=1)  # collect recent payment status
        AVG_BILL_AMT = st.number_input("Average Bill Amount", min_value=0.0, value=10000.0, step=500.0)  # collect average bill amount
        AVG_PAY_AMT = st.number_input("Average Payment Amount", min_value=0.0, value=5000.0, step=500.0)  # collect average payment amount

    with right_col:  # begin right input column
        TOTAL_DELAY_MONTHS = st.number_input("Total Delay Months", min_value=0, max_value=6, value=0, step=1)  # collect total months with delay
        MAX_DELAY = st.number_input("Maximum Delay", min_value=-2, max_value=8, value=0, step=1)  # collect the maximum delay value
        AVG_UTILIZATION = st.slider("Average Credit Utilization", min_value=0.0, max_value=1.5, value=0.30, step=0.01)  # collect average utilization ratio
        PAYMENT_RATIO = st.slider("Payment Ratio", min_value=0.0, max_value=1.2, value=0.50, step=0.01)  # collect payment ratio

    st.markdown('<div class="section-heading">Categorical Information</div>', unsafe_allow_html=True)  # show categorical section heading

    cat1, cat2, cat3 = st.columns(3, gap="medium")  # create three columns for categorical fields

    with cat1:  # begin sex selection column
        sex_choice = st.selectbox("Sex", ["Male", "Female"])  # collect sex category

    with cat2:  # begin education selection column
        education_choice = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])  # collect education category

    with cat3:  # begin marriage selection column
        marriage_choice = st.selectbox("Marriage", ["Married", "Single", "Others"])  # collect marriage category

    predict_btn = st.button("Predict Default Risk")  # create the main prediction button

    st.markdown('</div>', unsafe_allow_html=True)  # close styled input panel

with main_right:  # begin right dashboard column
    st.markdown(  # render model card
        """
        <div class="side-card">
            <div class="side-label">Model</div>
            <div class="side-value">Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(  # render target card
        """
        <div class="side-card">
            <div class="side-label">Target</div>
            <div class="side-value">Default Risk</div>
        </div>
        """,
        unsafe_allow_html=True
    )

SEX_2 = 1 if sex_choice == "Female" else 0  # encode female as 1 for the trained one-hot feature
EDUCATION_2 = 1 if education_choice == "University" else 0  # encode university category
EDUCATION_3 = 1 if education_choice == "High School" else 0  # encode high school category
EDUCATION_4 = 1 if education_choice == "Others" else 0  # encode others education category
MARRIAGE_2 = 1 if marriage_choice == "Single" else 0  # encode single marriage category
MARRIAGE_3 = 1 if marriage_choice == "Others" else 0  # encode others marriage category

if predict_btn:  # run prediction only when the user clicks the button
    try:  # start prediction block
        input_data = {  # collect all app inputs into one dictionary
            "LIMIT_BAL": LIMIT_BAL,  # include credit limit
            "AGE": AGE,  # include age
            "PAY_0": PAY_0,  # include recent payment status
            "AVG_BILL_AMT": AVG_BILL_AMT,  # include average bill amount
            "AVG_PAY_AMT": AVG_PAY_AMT,  # include average payment amount
            "TOTAL_DELAY_MONTHS": TOTAL_DELAY_MONTHS,  # include total delay months
            "MAX_DELAY": MAX_DELAY,  # include max delay
            "AVG_UTILIZATION": AVG_UTILIZATION,  # include utilization ratio
            "PAYMENT_RATIO": PAYMENT_RATIO,  # include payment ratio
            "SEX_2": SEX_2,  # include encoded sex feature
            "EDUCATION_2": EDUCATION_2,  # include encoded education feature
            "EDUCATION_3": EDUCATION_3,  # include encoded education feature
            "EDUCATION_4": EDUCATION_4,  # include encoded education feature
            "MARRIAGE_2": MARRIAGE_2,  # include encoded marriage feature
            "MARRIAGE_3": MARRIAGE_3  # include encoded marriage feature
        }

        ordered_input_df, scaled_input_df = prepare_input(input_data, feature_names, scaler)  # prepare both ordered and scaled dataframes
        prediction = model.predict(scaled_input_df)[0]  # generate the predicted class using scaled input
        probability = model.predict_proba(scaled_input_df)[0][1]  # generate the probability for the default class

        if probability < 0.30:  # check whether the predicted probability belongs to low risk
            risk_label = "Low"  # assign low risk label
            st.markdown(  # render low risk result box
                f"""
                <div class="result-box result-good">
                    Customer is not likely to default<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            interpretation = "This customer appears to have relatively low default risk."  # define low risk interpretation
        elif probability < 0.60:  # check whether the probability belongs to moderate risk
            risk_label = "Moderate"  # assign moderate risk label
            st.markdown(  # render moderate risk result box
                f"""
                <div class="result-box result-medium">
                    Customer appears to have moderate default risk<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            interpretation = "The customer shows some financial risk signals and should be monitored carefully."  # define moderate interpretation
        else:  # handle high risk predictions
            risk_label = "High"  # assign high risk label
            st.markdown(  # render high risk result box
                f"""
                <div class="result-box result-high">
                    Customer is likely to default<br>
                    Default probability: {probability:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            interpretation = "This customer appears to have high default risk and may require immediate attention."  # define high interpretation

        logger.info(f"Prediction success | probability={probability:.4f} | class={int(prediction)} | risk={risk_label} | input={input_data}")  # write full prediction details to the log
        save_prediction_record(input_data, ordered_input_df, scaled_input_df, prediction, probability, risk_label)  # save prediction to csv history

        st.info("Probability reflects likelihood, not certainty. Use this result as decision support.")  # show user guidance note
        st.write(f"Risk Interpretation: {interpretation}")  # show the risk interpretation text

        st.markdown("### Input Summary")  # show input summary title

        summary_df = ordered_input_df.T.reset_index()  # transpose the ordered input so each row shows one feature
        summary_df.columns = ["Feature", "Value"]  # rename the summary table columns

        table_rows = "".join(  # build html rows for the summary table
            f"""
            <tr>
                <td>{row['Feature']}</td>
                <td>{row['Value']}</td>
            </tr>
            """
            for _, row in summary_df.iterrows()  # loop through each summary row
        )

        st.markdown(  # render the summary table using custom html
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

        if hasattr(model, "feature_importances_"):  # check whether the model exposes feature importances
            importance_df = pd.DataFrame({  # create a dataframe for feature importance values
                "Feature": feature_names,  # use model feature names
                "Importance": model.feature_importances_  # use model importance scores
            }).sort_values(by="Importance", ascending=False).head(10)  # sort and keep the top 10 features

            st.markdown("### Top 10 Important Features")  # show feature importance section title

            fig, ax = plt.subplots(figsize=(10, 5))  # create the matplotlib figure and axis
            ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])  # draw a horizontal bar chart
            ax.set_title("Top 10 Important Features")  # set the chart title
            ax.set_xlabel("Importance Score")  # set the x-axis label
            ax.set_ylabel("Feature")  # set the y-axis label
            fig.patch.set_facecolor("#0f172a")  # set the figure background color
            ax.set_facecolor("#0f172a")  # set the axes background color
            ax.tick_params(axis="x", colors="#e5e7eb")  # style x-axis tick labels
            ax.tick_params(axis="y", colors="#e5e7eb")  # style y-axis tick labels
            ax.xaxis.label.set_color("#e5e7eb")  # style the x-axis label color
            ax.yaxis.label.set_color("#e5e7eb")  # style the y-axis label color
            ax.title.set_color("#f8fafc")  # style the chart title color

            for spine in ax.spines.values():  # loop through the chart borders
                spine.set_color("#334155")  # set each border color

            st.pyplot(fig)  # display the chart in the app

    except Exception as e:  # catch any runtime prediction error
        logger.error(f"Prediction failed: {e}")  # write the error details to the log
        st.error(f"Prediction failed: {e}")  # show the error message in the app
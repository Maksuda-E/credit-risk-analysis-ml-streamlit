# import os so the code can create folders and file paths
import os

# import json so dictionary data can be safely stored in csv logs
import json

# import joblib so saved model files can be loaded
import joblib

# import pandas so input and saved records can be handled as dataframes
import pandas as pd

# import datetime so every saved prediction gets a timestamp
from datetime import datetime


# define a function to load the model, feature names, and optional scaler
def load_model(model_path, features_path, scaler_path=None):
    # load the trained model from disk
    model = joblib.load(model_path)

    # load the feature list that was used during training
    features = joblib.load(features_path)

    # set scaler to None by default
    scaler = None

    # check whether a scaler path was provided and whether the file exists
    if scaler_path is not None and os.path.exists(scaler_path):
        # load the scaler if it is available
        scaler = joblib.load(scaler_path)

    # return all loaded artifacts
    return model, features, scaler


# define a function to prepare user input to match the model feature structure
def prepare_input(input_data, feature_names):
    # convert the user input dictionary into a single-row dataframe
    input_df = pd.DataFrame([input_data])

    # loop through each feature expected by the model
    for col in feature_names:
        # if the column is missing from user input, create it and fill with zero
        if col not in input_df.columns:
            input_df[col] = 0

    # reorder the dataframe columns to exactly match the training feature order
    input_df = input_df[feature_names]

    # return the aligned dataframe
    return input_df


# define a function to optionally scale input before prediction
def transform_input(input_df, scaler=None):
    # if no scaler is available, return the original dataframe unchanged
    if scaler is None:
        return input_df.copy()

    # use the scaler to transform the dataframe values
    scaled_array = scaler.transform(input_df)

    # convert the scaled values back into a dataframe with the same columns
    scaled_df = pd.DataFrame(scaled_array, columns=input_df.columns)

    # return the scaled dataframe
    return scaled_df


# define a function to convert UI selections into encoded model features
def build_encoded_features(sex_choice, education_choice, marriage_choice):
    # encode sex using the same dummy structure used in the notebook
    sex_2 = 1 if sex_choice == "Female" else 0

    # encode education for university
    education_2 = 1 if education_choice == "University" else 0

    # encode education for high school
    education_3 = 1 if education_choice == "High School" else 0

    # encode education for others
    education_4 = 1 if education_choice == "Others" else 0

    # encode marriage for single
    marriage_2 = 1 if marriage_choice == "Single" else 0

    # encode marriage for others
    marriage_3 = 1 if marriage_choice == "Others" else 0

    # return all encoded values as a dictionary
    return {
        "SEX_2": sex_2,
        "EDUCATION_2": education_2,
        "EDUCATION_3": education_3,
        "EDUCATION_4": education_4,
        "MARRIAGE_2": marriage_2,
        "MARRIAGE_3": marriage_3,
    }


# define a function to create the final app input dictionary
def build_input_data(
    limit_bal,
    age,
    pay_0,
    avg_bill_amt,
    avg_pay_amt,
    total_delay_months,
    max_delay,
    avg_utilization,
    payment_ratio,
    sex_choice,
    education_choice,
    marriage_choice
):
    # get the encoded categorical values
    encoded = build_encoded_features(sex_choice, education_choice, marriage_choice)

    # build the input dictionary using the engineered features expected by the app
    input_data = {
        "LIMIT_BAL": limit_bal,
        "AGE": age,
        "PAY_0": pay_0,
        "AVG_BILL_AMT": avg_bill_amt,
        "AVG_PAY_AMT": avg_pay_amt,
        "TOTAL_DELAY_MONTHS": total_delay_months,
        "MAX_DELAY": max_delay,
        "AVG_UTILIZATION": avg_utilization,
        "PAYMENT_RATIO": payment_ratio,
        "SEX_2": encoded["SEX_2"],
        "EDUCATION_2": encoded["EDUCATION_2"],
        "EDUCATION_3": encoded["EDUCATION_3"],
        "EDUCATION_4": encoded["EDUCATION_4"],
        "MARRIAGE_2": encoded["MARRIAGE_2"],
        "MARRIAGE_3": encoded["MARRIAGE_3"],
    }

    # return the final input dictionary
    return input_data


# define a function that converts probability into a human-friendly risk label
def get_risk_label(probability):
    # return low risk for probabilities below 0.30
    if probability < 0.30:
        return "Low Risk"

    # return moderate risk for probabilities below 0.60
    if probability < 0.60:
        return "Moderate Risk"

    # otherwise return high risk
    return "High Risk"


# define a function that returns explanation text for each risk level
def get_risk_interpretation(risk_label):
    # return interpretation for low risk
    if risk_label == "Low Risk":
        return "This customer appears to have relatively low default risk."

    # return interpretation for moderate risk
    if risk_label == "Moderate Risk":
        return "The customer shows some financial risk signals and should be monitored carefully."

    # return interpretation for high risk
    return "This customer appears to have high default risk and may require immediate attention."


# define a function to save every prediction into a csv file
def save_prediction_record(raw_input, prepared_input_df, probability, prediction, risk_label, logs_dir="logs"):
    # create the logs folder if it does not exist
    os.makedirs(logs_dir, exist_ok=True)

    # build the csv file path
    csv_path = os.path.join(logs_dir, "prediction_records.csv")

    # create a timestamp for the record
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # create a dictionary representing one prediction record
    record = {
        "timestamp": timestamp,
        "prediction_class": int(prediction),
        "default_probability": float(probability),
        "risk_label": risk_label,
        "raw_input_json": json.dumps(raw_input),
        "prepared_input_json": prepared_input_df.to_json(orient="records"),
    }

    # convert the record into a one-row dataframe
    record_df = pd.DataFrame([record])

    # append to the existing csv if it already exists
    if os.path.exists(csv_path):
        record_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        record_df.to_csv(csv_path, index=False)

    # return the saved record for optional use in the app
    return record
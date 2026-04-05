import os  # import os to work with file paths
import json  # import json to save raw input as structured text
import joblib  # import joblib to load model artifacts
import pandas as pd  # import pandas to create and manage dataframes
from datetime import datetime  # import datetime to timestamp prediction records

def load_model(model_path, features_path, scaler_path):
    model = joblib.load(model_path)  # load the trained machine learning model
    features = joblib.load(features_path)  # load the exact feature names used during training
    scaler = joblib.load(scaler_path)  # load the scaler used during model training
    return model, features, scaler  # return all required artifacts together

def prepare_input(input_data, feature_names, scaler):
    input_df = pd.DataFrame([input_data])  # convert the input dictionary into a one-row dataframe

    for col in feature_names:  # loop through every feature the model expects
        if col not in input_df.columns:  # check whether the feature is missing from the input dataframe
            input_df[col] = 0  # add missing features with default value zero

    input_df = input_df[feature_names]  # reorder dataframe columns to exactly match the model training order

    scaled_array = scaler.transform(input_df)  # scale the dataframe using the same scaler used in training
    scaled_df = pd.DataFrame(scaled_array, columns=feature_names)  # convert the scaled array back into a dataframe

    return input_df, scaled_df  # return both raw ordered input and scaled model-ready input

def save_prediction_record(raw_input, ordered_input_df, scaled_input_df, prediction, probability, risk_label):
    root_dir = os.path.dirname(os.path.dirname(__file__))  # move from app folder to project root
    logs_dir = os.path.join(root_dir, "logs")  # build the logs folder path
    os.makedirs(logs_dir, exist_ok=True)  # create the logs folder if it does not exist

    csv_path = os.path.join(logs_dir, "prediction_records.csv")  # define the csv file path for saved predictions
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # create a human-readable timestamp

    record = {  # create one structured record for the prediction
        "timestamp": timestamp,  # save the prediction time
        "prediction": int(prediction),  # save the predicted class as integer
        "probability": float(probability),  # save the predicted probability
        "risk_label": risk_label,  # save the risk label text
        "raw_input": json.dumps(raw_input),  # save the raw user input as json text
        "ordered_input": ordered_input_df.to_json(orient="records"),  # save ordered unscaled input as json text
        "scaled_input": scaled_input_df.to_json(orient="records")  # save scaled input as json text
    }

    record_df = pd.DataFrame([record])  # convert the record dictionary into a one-row dataframe

    if os.path.exists(csv_path):  # check whether the csv file already exists
        record_df.to_csv(csv_path, mode="a", header=False, index=False)  # append the new record without rewriting headers
    else:  # handle the case where the file does not exist yet
        record_df.to_csv(csv_path, index=False)  # create the csv file and include headers

    return record  # return the saved record for optional use in the app
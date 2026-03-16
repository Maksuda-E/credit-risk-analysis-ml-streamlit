# import joblib to load the saved model
import joblib

# import pandas to handle input data
import pandas as pd

# function to load the trained model and feature names
def load_model(model_path, features_path):

    # load the trained model from the saved file
    model = joblib.load(model_path)

    # load the feature column names used during training
    features = joblib.load(features_path)

    # return both the model and the feature list
    return model, features


# function to prepare user input so it matches the model's feature structure
def prepare_input(input_data, feature_names):

    # convert user input dictionary into a dataframe
    input_df = pd.DataFrame([input_data])

    # loop through all model features
    for col in feature_names:

        # if a feature is missing from user input, add it with value 0
        if col not in input_df.columns:
            input_df[col] = 0

    # reorder columns so they match the order used during training
    input_df = input_df[feature_names]

    # return the prepared dataframe
    return input_df
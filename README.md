# Credit Risk Analysis and Default Prediction

## Overview
This project analyzes credit card customer data to predict the likelihood of payment default using machine learning. The goal is to help identify customers who may have difficulty repaying their credit card balance, allowing financial institutions to manage credit risk more effectively. It follows a complete data science workflow including exploratory data analysis, data preprocessing, feature engineering, model training, model evaluation, and deployment using a Streamlit web application.

## Objectives
The main objectives of this project are:
1. Analyze customer financial behavior and payment history
2. Build a predictive model to estimate credit default risk
3. Compare multiple machine learning models
4. Deploy the best model in an interactive web application

## Dataset
The dataset contains information about credit card customers including:
- Credit limit
- Age
- Payment status history
- Bill amounts
- Payment amounts
- Education and marital status

## The target variable indicates whether the customer defaulted on the next payment.

## Project Workflow
The project is divided into several stages:

1. Exploratory Data Analysis (EDA)

Initial analysis was performed to understand the structure and distribution of the data. Key visualizations were used to explore patterns in credit usage, payment behavior, and default rates.

2. Data Preprocessing

Data preprocessing steps included:

- Removing unnecessary columns

- Checking and handling missing values

- Encoding categorical variables

- Preparing the dataset for machine learning models

3. Feature Engineering

- Additional features were created to improve model performance, such as:

- Average bill amount

- Average payment amount

- Total delay months

- Maximum delay

- Credit utilization ratio

- Payment ratio

- These features help capture customer financial behavior more effectively.

4. Model Training

Multiple classification models were trained and compared:

1. Logistic Regression

2. Decision Tree

3. Random Forest

## Model Evaluation

Models were evaluated using several performance metrics:

- Accuracy

- Precision

- Recall

- F1-score

- Confusion Matrix

Among the tested models, Random Forest achieved the best overall performance with approximately 82% accuracy.

## Model Deployment

The best model was deployed using Streamlit, allowing users to input customer information and receive a prediction of default risk.

The application also includes:

Prediction probability

Feature importance visualization

Error handling and logging

## Streamlit Application

The Streamlit app allows users to:

1. Enter customer financial information

2. Predict default risk

3. View prediction probability

4. See important features influencing the model

This makes the model easier to interact with and demonstrates a practical use case for credit risk prediction.


## Run the Application

### Install dependencies:

pip install -r requirements.txt

### Run the app:

python -m streamlit run app/app.py


## Technologies Used

Python

Pandas

Scikit-learn

Matplotlib

Joblib

Streamlit

## Project Structure
```text
credit-risk-analysis-ml-streamlit/
│
├── app/                # Streamlit application files
├── dataset/            # Raw and cleaned datasets
├── model/              # Trained model and scaler files
├── notebook/           # Jupyter notebooks (EDA, preprocessing, log for preprocessing, model)
├── logs/               # Log files for error tracking and recording 
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

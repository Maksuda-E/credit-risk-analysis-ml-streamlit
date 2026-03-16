Credit Risk Analysis and Default Prediction
Overview

This project analyzes credit card customer data to predict the likelihood of payment default using machine learning. The goal is to help identify customers who may have difficulty repaying their credit card balance, allowing financial institutions to manage credit risk more effectively.

The project follows a complete data science workflow including exploratory data analysis, data preprocessing, feature engineering, model training, model evaluation, and deployment using a Streamlit web application.

Objectives

The main objectives of this project are:

Analyze customer financial behavior and payment history

Build a predictive model to estimate credit default risk

Compare multiple machine learning models

Deploy the best model in an interactive web application

Dataset

The dataset contains information about credit card customers including:

Credit limit

Age

Payment status history

Bill amounts

Payment amounts

Education and marital status

The target variable indicates whether the customer defaulted on the next payment.

Project Workflow

The project is divided into several stages:

1. Exploratory Data Analysis (EDA)

Initial analysis was performed to understand the structure and distribution of the data. Key visualizations were used to explore patterns in credit usage, payment behavior, and default rates.

2. Data Preprocessing

Data preprocessing steps included:

Removing unnecessary columns

Checking and handling missing values

Encoding categorical variables

Preparing the dataset for machine learning models

3. Feature Engineering

Additional features were created to improve model performance, such as:

Average bill amount

Average payment amount

Total delay months

Maximum delay

Credit utilization ratio

Payment ratio

These features help capture customer financial behavior more effectively.

4. Model Training

Multiple classification models were trained and compared:

Logistic Regression

Decision Tree

Random Forest

5. Model Evaluation

Models were evaluated using several performance metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Among the tested models, Random Forest achieved the best overall performance with approximately 82% accuracy.

6. Model Deployment

The best model was deployed using Streamlit, allowing users to input customer information and receive a prediction of default risk.

The application also includes:

Prediction probability

Feature importance visualization

Error handling and logging

Streamlit Application

The Streamlit app allows users to:

Enter customer financial information

Predict default risk

View prediction probability

See important features influencing the model

This makes the model easier to interact with and demonstrates a practical use case for credit risk prediction.

Technologies Used

Python

Pandas

Scikit-learn

Matplotlib

Joblib

Streamlit

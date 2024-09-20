## Stroke Prediction App 🧠

This is a Streamlit web application that predicts the likelihood of a stroke based on various health parameters such as gender, age, hypertension, heart disease, BMI, and more. The app uses a pre-trained machine learning model saved in a .pkl file to make predictions.
Table of Contents

    * [Overview](#overview)
    * [Features](#features)
    * [Potential Issues and Solutions] (#potential-issues-and-solutions)
    
## Overview
This project implements a stroke prediction system where users can input personal health data to determine their risk of having a stroke. The system uses Pre-trained models like `SVM`, `RandomForest` and `K-NearestNeighbours` saved in a .pkl file for inference and deployed it on Streamlit.

## Features

* User Input: Allows users to input health data such as age, gender, BMI, and more.
* Stroke Prediction: Uses pretrained machine learning models to predict the risk of stroke.
* Categorical Feature Encoding: Uses 'LabelEncoder' and 'OneHotEncoder' for categorical data.
* Model Handling: Loads a '.pkl' model for efficient inference.

## Potential Issues and Solutions
* Mismatch Between Input and Model Columns
    Issue: The input data had different feature names (e.g., gender, work_type) than the one-hot encoded features             expected by the model (e.g., gender_1, work_type_1).
    Solution: The input data was manually transformed using LabelEncoder and OneHotEncoder to match the one-hot encoded       feature names used during training.
* Incorrect Input Shape for Prediction
    Issue: The input shape for prediction didn't match the format expected by the model (due to one-hot encoding).            Solution: The input features were encoded using OneHotEncoder inside a ColumnTransformer, ensuring that the input         format was compatible with the trained model.

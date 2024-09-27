## Stroke Prediction App 🧠

This is a **Streamlit** web application that predicts the likelihood of a stroke based on various health parameters, such as gender, age, hypertension, heart disease, BMI, and more. The app utilizes a pre-trained machine learning model, stored in a `.pkl` file, to make real-time predictions.

![Stroke Prediction App](https://github.com/user-attachments/assets/f95d1d50-2ecc-4f96-b130-1d2ec41c25aa)

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Potential Issues and Solutions](#potential-issues-and-solutions)

---

## Overview

This project implements a **stroke prediction system** where users input personal health data to assess their risk of having a stroke. The system employs pre-trained models like `SVM`, `RandomForest`, and `K-Nearest Neighbors`, stored in a `.pkl` file for efficient inference, and deploys the prediction interface using **Streamlit**.

---

## Features

* **User Input**: Allows users to input personal health data (e.g., age, gender, BMI, etc.).
* **Stroke Prediction**: Utilizes pre-trained machine learning models to assess stroke risk.
* **Categorical Feature Encoding**: Incorporates `LabelEncoder` and `OneHotEncoder` to handle categorical data seamlessly.
* **Model Handling**: Loads a `.pkl` file to provide real-time, efficient model inference.

![Model Prediction](https://github.com/user-attachments/assets/07560eb6-3743-4a71-aef2-bd1ec3d33940)

---

## Potential Issues and Solutions

### 1. Mismatch Between Input and Model Columns
**Issue**: The input data used different feature names (e.g., `gender`, `work_type`) than the one-hot encoded features expected by the model (e.g., `gender_1`, `work_type_1`).

**Solution**: Manually transformed the input data using `LabelEncoder` and `OneHotEncoder` to match the one-hot encoded feature names used during training.

### 2. Incorrect Input Shape for Prediction
**Issue**: The input shape didn't match the format expected by the model due to one-hot encoding.

**Solution**: Used a `ColumnTransformer` with `OneHotEncoder` to ensure the input format was compatible with the model's expected feature set.

---


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import urllib.request
from sklearn.preprocessing import LabelEncoder

st.title('🧠 BrainStroke Prediction - Machine Learning Models 🤖')

st.info(
    'This app predicts the occurrence of Brain Strokes based on your inputs using different Machine Learning models.')

# Load the data
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv(
        'https://raw.githubusercontent.com/kvramanan93/Brain_Stroke_Predicion/master/healthcare-dataset-stroke-data.csv',
        sep=',')
    st.write(df)

    st.write('**Features (X)**')
    X = df.drop({'stroke', 'id'}, axis=1)
    st.write(X)

    st.write('**Target (y)**')
    y = df.stroke
    st.write(y)

# Load the pickled models (KNN, RandomForest, SVC)
import urllib.request
import pickle
import os

# Load the pickled models (KNN, RandomForest, SVC)
model_urls = {
    'KNN': 'https://raw.githubusercontent.com/kvramanan93/Brain_Stroke_Predicion/master/model_knn.pkl',
    'RandomForest': 'https://raw.githubusercontent.com/kvramanan93/Brain_Stroke_Predicion/master/randomForest.pkl',
    'SVC': 'https://raw.githubusercontent.com/kvramanan93/Brain_Stroke_Predicion/master/model_svc.pkl'
}

models = {}

for model_name, model_url in model_urls.items():
    try:
        # Download the model file
        file_path = f'{model_name}_model.pkl'
        urllib.request.urlretrieve(model_url, file_path)

        # Check if the downloaded file is valid
        with open(file_path, 'rb') as f:
            content = f.read(100)  # Read the first 100 bytes
            if content.startswith(b'\x80\x04'):  # A valid pickle file starts with these bytes
                print(f"{model_name} file seems valid. Loading model...")
                # Reset the file pointer and load the model
                f.seek(0)
                models[model_name] = pickle.load(f)
            else:
                print(f"{model_name} does not appear to be a valid model file. Skipping.")
                st.error(f"{model_name} could not be loaded. The file is not valid.")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        st.error(f"Error loading {model_name}: {e}")

# Model performance metrics (you'll have to manually populate these values from your evaluation results)
model_info = {
    'KNN': {
        'Accuracy': 88,
        'Precision': 0.22,
        'Recall': 0.37,
        'F1-score': 0.27
    },
    'RandomForest': {
        'Accuracy': 91,
        'Precision': 0.15,
        'Recall': 0.11,
        'F1-score': 0.12
    },
    'SVC': {
        'Accuracy': 73,
        'Precision': 0.75,
        'Recall': 0.15,
        'F1-score': 0.25
    }
}


# Function to collect user input features
def user_input_features():
    st.header('Enter Your Details for Prediction')

    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.slider('Age', 0, 100, 30)
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
    Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    # Prepare the dictionary with transformed input
    data = {
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,

        # Manually encoding gender
        'gender_1': 1 if gender == 'Male' else 0,  # Male
        'gender_2': 1 if gender == 'Female' else 0,  # Female

        # Manually encoding ever_married
        'ever_married_1': 1 if ever_married == 'Yes' else 0,

        # Manually encoding work_type
        'work_type_1': 1 if work_type == 'Private' else 0,
        'work_type_2': 1 if work_type == 'Self-employed' else 0,
        'work_type_3': 1 if work_type == 'Govt_job' else 0,
        'work_type_4': 1 if work_type == 'Children' else 0,

        # Manually encoding Residence_type
        'Residence_type_1': 1 if Residence_type == 'Urban' else 0,

        # Manually encoding smoking_status
        'smoking_status_1': 1 if smoking_status == 'formerly smoked' else 0,
        'smoking_status_2': 1 if smoking_status == 'never smoked' else 0,
        'smoking_status_3': 1 if smoking_status == 'smokes' else 0
    }

    features = pd.DataFrame(data, index=[0])
    features.head()
    return features


# Collect user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

input_encoded = input_df[:1]

# Model selection
st.subheader('Select Model for Prediction')
model_choice = st.selectbox('Choose a model', ['KNN', 'RandomForest', 'SVC'])

# Show model details
st.write(f"**{model_choice} Model Performance:**")
st.write(f"- **Accuracy**: {model_info[model_choice]['Accuracy']}%")
st.write(f"- **Precision:** {model_info[model_choice]['Precision']}")
st.write(f"- **Recall:** {model_info[model_choice]['Recall']}")
st.write(f"- **F1-score:** {model_info[model_choice]['F1-score']}")

if st.button('Predict'):
    # Ensure the selected model is valid
    model = models.get(model_choice)
    if model:
        print(type(model))  # Check the type of the model

        # Make predictions
        prediction = model.predict(input_encoded)
        prediction_proba = model.predict_proba(input_encoded)

        st.subheader('Prediction Result')
        stroke_result = 'Yes' if prediction[0] == 1 else 'No'
        st.write(f"**Will you have a stroke?** {stroke_result}")

        st.subheader('Prediction Probability')

        # Custom threshold (0.6)
        threshold = 0.6

        # Display probabilities
        st.write(f"**Probability of Not Having a Stroke:** {prediction_proba[0][0]:.2f}")
        st.write(f"**Probability of Having a Stroke:** {prediction_proba[0][1]:.2f}")

    else:
        st.error("No valid model selected for prediction.")

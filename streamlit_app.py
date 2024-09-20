import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import urllib.request
from sklearn.preprocessing import LabelEncoder

st.title('🧠 BrainStroke Prediction - Machine Learning Model 🤖')

st.info('This app predicts the occurrence of Brain Strokes based on your inputs')

# Load the data
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://github.com/kvramanan93/Brain_Stroke_Predicion/blob/40f2ecbeb6169fb88ccb4d550a4e7fbda3133d51/healthcare-dataset-stroke-data.csv')
    st.write(df)

    st.write('**Features (X)**')
    X = df.drop({'stroke', 'id'}, axis=1)
    st.write(X)

    st.write('**Target (y)**')
    y = df.stroke
    st.write(y)
             
# Data Visualization
with st.expander('Data Visualization'):
    df['hypertension'] = df['hypertension'].map({1: 'Yes', 0: 'No'})
    df['heart_disease'] = df['heart_disease'].astype('category')

    st.write('**Hypertension vs Heart Disease Rates**')

    fig = px.bar(
        df, x='hypertension', y='stroke', color='heart_disease', barmode='group', 
        title='Stroke vs Hypertension & Heart Disease',
        category_orders={'hypertension': ['Yes', 'No']},
        color_discrete_map={0: 'blue', 1: 'red'}
    )

    fig.update_yaxes(showticklabels=False, title='')
    st.plotly_chart(fig)

# Load the pickled model
model_url = 'https://raw.githubusercontent.com/arpitgour16/Brain_Stroke_prediction_analysis/main/stroke_model.pkl'
urllib.request.urlretrieve(model_url, 'stroke_model.pkl')

with open('stroke_model.pkl', 'rb') as f:
    model = pickle.load(f)

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
    
    data = {
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Collect user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Preprocessing
# Encode categorical variables using LabelEncoder
# Ensure the encoders are consistent with those used during model training
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Combine user input with original data for consistent encoding
df_combined = pd.concat([input_df, X], axis=0)

# Encode categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    df_combined[col] = le.fit_transform(df_combined[col])

# Select only the first row (user input)
input_encoded = df_combined[:1]

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_encoded)
    prediction_proba = model.predict_proba(input_encoded)

    st.subheader('Prediction Result')
    stroke_result = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f"**Will you have a stroke?** {stroke_result}")

    st.subheader('Prediction Probability')
    st.write(f"**Probability of Not Having a Stroke:** {prediction_proba[0][0]:.2f}")
    st.write(f"**Probability of Having a Stroke:** {prediction_proba[0][1]:.2f}")

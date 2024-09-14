import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title('🧠 BrainStroke Prediction - Machine learning Model 🤖')

st.info('This app predicts the occurance of Brain Strokes based on your inputs')

with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/arpitgour16/Brain_Stroke_prediction_analysis/main/healthcare-dataset-stroke-data.csv')
    df

    st.write('**X**')
    X = df.drop({'stroke','id'},axis=1)
    X
    
    st.write('**y**')
    y = df.stroke
    y
             
with st.expander('Data Visualization'):
    df['hypertension'] = df['hypertension'].map({1: 'Yes', 0: 'No'})

    # Ensure 'heart_disease' is treated as categorical with specific colors
    df['heart_disease'] = df['heart_disease'].astype('category')

    st.write('**Hypertension vs Heart Disease Rates**')

    # Create bar chart with custom colors
    fig = px.bar(df, x='hypertension', y='stroke', color='heart_disease', barmode='group', 
                title='Stroke vs Hypertension & Heart Disease',
                category_orders={'hypertension': ['Yes', 'No']},
                color_discrete_map={0: 'blue', 1: 'red'})  # Custom colors for heart disease categories

    # Hide y-axis tick labels and remove y-axis label
    fig.update_yaxes(showticklabels=False, title='')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def predict(stroke, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'stroke': [stroke],
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })
    

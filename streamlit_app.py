import streamlit as st
import pandas as pd

st.title('🧠 BrainStroke Prediction Machine learning Model 🤖')

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
             
with st.expander('Data isualization'):
    st.scatter_chart(data=df, x=)
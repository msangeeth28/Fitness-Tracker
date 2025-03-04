import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

import warnings
warnings.filterwarnings('ignore')

st.write("## Personal Fitness Tracker")
st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as Age, Gender, BMI, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load pre-trained model and expected columns
with open("own_model.pkl", "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    expected_columns = model_data["columns"]

# Ensure feature alignment
df = df.reindex(columns=expected_columns, fill_value=0)

# Convert to NumPy array for prediction
prediction = model.predict(df.to_numpy())

st.write("---")
st.header("Prediction: ")
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} *kilocalories*")
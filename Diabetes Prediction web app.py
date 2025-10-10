# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:01 2021

@author: BTech ETT
"""

import numpy as np
import pickle
import streamlit as st
import mysql.connector

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

# Save prediction to MySQL
def save_prediction_to_db(Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age, prediction):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="MpanoKuzwa@2",
            database="diabetes_predictions"
        )
        cursor = conn.cursor()
        query = """
            INSERT INTO diabetes_predictions (
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, bmi, DiabetesPedigreeFunction, Age, prediction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = (Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age, prediction)
        cursor.execute(query, data)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

# Streamlit interface
def main():
    st.title('Diabetes Prediction Web App')

    # User input
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]

            diagnosis = diabetes_prediction(input_data)

            save_prediction_to_db(float(Pregnancies), float(Glucose), float(BloodPressure),
                                  float(SkinThickness), float(Insulin), float(BMI),
                                  float(DiabetesPedigreeFunction), float(Age), diagnosis)

            st.success(diagnosis)

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

if __name__ == '__main__':
    main()

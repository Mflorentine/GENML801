# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:01 2021

@author: BTech ETT
"""

import numpy as np
import pickle
import streamlit as st
import mysql.connector

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# Prediction function

def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
   
    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
        
# Save prediction to MySQL
def save_prediction_to_db(Pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age, prediction):
    try:
        conn = mysql.connector.connect(
            host="localhost",  # Replace with your host
            user="root",  # Replace with your MySQL username
            password="MpanoKuzwa@2",  # Replace with your MySQL password
            database="diabetes_predictions"
        )
        cursor = conn.cursor()
        query = """
            INSERT INTO diabetes_predictions (
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age, prediction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = (pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age, prediction)
        cursor.execute(query, data)
        conn.commit()
        cursor.close()
        conn.close()
except Exception as e:
        st.error(f"Database error: {e}")

# Streamlit interface 
def main():
     
    # giving a title
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
        
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        #diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        try:
            # Convert inputs to float
            input_data = [float(Pregnancies), float(glucose), float(blood_pressure),
                          float(skin_thickness), float(insulin), float(bmi),
                          float(diabetes_pedigree), float(age)]

            diagnosis = diabetes_prediction(input_data)

            # Save to database
            save_prediction_to_db(float(Pregnancies), float(glucose), float(blood_pressure),
                                  float(skin_thickness), float(insulin), float(bmi),
                                  float(diabetes_pedigree), int(age), diagnosis)
        
             st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

      
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
  
    

  



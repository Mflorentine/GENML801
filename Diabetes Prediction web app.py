# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:01 2021

@author: BTech ETT
"""

# Import necessary libraries
import numpy as np  # For numerical operations and array handling
import pickle       # For loading the saved machine learning model
import streamlit as st  # For building the web app interface
import mysql.connector  # For connecting to MySQL database

# Load the trained model from file
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Define a function to make predictions using the model
def diabetes_prediction(input_data):
    # Convert input list to NumPy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshape the array to match model's expected input shape (1 row, n columns)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)
    
    # Return a human-readable result based on prediction
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

# Define a function to save prediction and inputs to MySQL database
def save_prediction_to_db(Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age, prediction):
    try:
        # Connect to the MySQL database
        conn = mysql.connector.connect(
            host="localhost",        # Hostname of the MySQL server
            user="root",             # MySQL username
            password="KuzwaAryan@1", # MySQL password
            database="diabetes_predictions"  # Database name
        )
        
        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()
        
        # SQL query to insert a new row into the table
        query = """
            INSERT INTO diabetes_predictions (
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, bmi, DiabetesPedigreeFunction, Age, prediction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Tuple of values to insert into the table
        data = (Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age, prediction)
        
        # Execute the query with the provided data
        cursor.execute(query, data)
        
        # Commit the transaction to save changes
        conn.commit()
        
        # Close the cursor and connection to free resources
        cursor.close()
        conn.close()
    
    # Handle any errors that occur during database operations
    except Exception as e:
        st.error(f"Database error: {e}")

# Define the main function that runs the Streamlit app
def main():
    # Set the title of the web app
    st.title('Diabetes Prediction Web App')

    # Collect user input from text fields
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Initialize diagnosis result
    diagnosis = ''

    # When the user clicks the prediction button
    if st.button('Diabetes Test Result'):
        try:
            # Convert all inputs to float for model compatibility
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]

            # Get prediction result from model
            diagnosis = diabetes_prediction(input_data)

            # Save the input and prediction result to the database
            save_prediction_to_db(float(Pregnancies), float(Glucose), float(BloodPressure),
                                  float(SkinThickness), float(Insulin), float(BMI),
                                  float(DiabetesPedigreeFunction), float(Age), diagnosis)

            # Display the result to the user
            st.success(diagnosis)

        # Handle invalid input (e.g., non-numeric values)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

# Run the app
if __name__ == '__main__':
    main()


# Importing necessary libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# Load the model and vectorizer
model = joblib.load('naive_bayes_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Creating the Streamlit app
st.title('FAKE NEWS DETECTION')
image = Image.open('fake news.jpg')
st.image(image, use_column_width=True)

# Taking user input
user_input = st.text_input("Enter the news text to verify:")

if user_input:
    user_input_transformed = vectorizer.transform([user_input])

    # Add a button for users to trigger the prediction
    if st.button("Verify"):
        # Check if the user has entered any text
        if user_input:
            # Make prediction using the loaded model
            prediction = model.predict(user_input_transformed)
            print(prediction)
            # Display the prediction result
            st.sidebar.write("## Prediction:")
            if prediction == 1:
                st.success("Real News Detected!")
            else:
                st.error("Fake News Detected!")
        else:
            st.warning("Please enter some news text to verify.")
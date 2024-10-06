import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the saved model
model = load_model('models/malaria_model.h5')

# Define the classes
class_names = ['Parasitized', 'Uninfected']

# Streamlit App Interface
st.title("Malaria Detection App")
st.write("Upload an image of a cell to predict if it's parasitized or uninfected.")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = image.resize((100, 100))  # Resize as per your model's input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]

    # Display prediction
    st.write(f"The image is predicted as: **{predicted_label}**")

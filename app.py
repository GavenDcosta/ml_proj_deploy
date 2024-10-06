import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your model
model = tf.keras.models.load_model('models/malaria_model.h5')

# Define the image dimensions
img_height, img_width = 100, 100

# Define classes
class_names = ['Parasitized', 'Uninfected']

# Streamlit app title
st.title("Malaria Detection App")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image to match the input shape of the model
    img = image.resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit the model input

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]

    # Display prediction
    st.write(f"Prediction: **{predicted_label}**")

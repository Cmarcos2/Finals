import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Set the page config
st.set_page_config(
    page_title="Weather Classification System",
    page_icon="⛅",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the model with caching to speed up subsequent runs
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Finals_Exam.hdf5")
    return model

model = load_model()

# Custom CSS to style the app and adjust the message position
st.markdown("""
    <style>
    body {
        background-color: #e0f7fa;  /* Light cyan background */
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .header {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .names {
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .date {
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .uploader {
        text-align: center;
        font-size: 1em;
        font-weight: bold;
        margin-bottom: 20px; /* Add margin-bottom */
    }
    .file-uploader {
        text-align: center;
        margin-bottom: 10px;
    }
    .placeholder {
        font-size: 0.9em;
        color: red;
        margin-top: -10px; /* Adjust margin-top */
        margin-bottom: 20px; /* Add margin-bottom */
    }
    .output {
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        color: green;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="header main">Final Examination: Weather Classification System</div>', unsafe_allow_html=True)

# Names and Date
st.markdown('<div class="names main">Christian Marcos | Ji Han Gang | May 19, 2024</div>', unsafe_allow_html=True)

# File uploader with a placeholder for the message
st.markdown('<div class="uploader main">A Weather Photo Can Predict if it is Shine, Cloudy, Sunrise, or Rain</div>', unsafe_allow_html=True)
upload_placeholder = st.empty()  # Placeholder for the message

# Wrap the file uploader inside a div and assign a custom class
file_uploader_div = st.empty()
file_uploader_div.markdown('<div class="file-uploader">', unsafe_allow_html=True)
file = st.file_uploader("", type=["jpg", "png"], key="file_uploader", label_visibility="collapsed")
file_uploader_div.markdown('</div>', unsafe_allow_html=True)

def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size)
    img_array = np.asarray(image)
    img_array = img_array[np.newaxis, ...]
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    return prediction

if file is None:
    upload_placeholder.markdown('<div class="placeholder">Please upload an image file</div>', unsafe_allow_html=True)
else:
    upload_placeholder.empty()  # Clear the placeholder
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    st.success(f'Prediction: {predicted_class_label}')

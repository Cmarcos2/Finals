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

# Custom CSS to style the app
st.markdown("""
    <style>
     .main {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .header {
        text-align: center;
        font-size: 1.5em;
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
        margin-bottom: 20px;
    }
    .output {
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        color: blue;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="header main">Final Examination: Weather Classification System</div>', unsafe_allow_html=True)

# Names and Date
st.markdown('<div class="names main">Christian Marcos | Ji Han Gang | May 19, 2024</div>', unsafe_allow_html=True)

# File uploader
st.markdown('<div class="uploader main">Explore the weather with a simple photo upload! Our system predicts whether the sky is shine, cloudy, displaying a beautiful sunrise, or experiencing rain. Try it out now!</div>', unsafe_allow_html=True)
# Message
st.text("Please upload an image file")
file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size)
    img_array = np.asarray(image)
    img_array = img_array[np.newaxis, ...]
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    return prediction

if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    st.success(f'Prediction: {predicted_class_label}')

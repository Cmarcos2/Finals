import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("Finals_Exam.hdf5")  
    return model

model = load_model()

st.write("""
# Weather Classification System
""")

# Your names
name1 = 'Christian Marcos'
name2 = 'Ji Han Gang'
# Display the names in bold
st.markdown(f"**{name1}**")
st.markdown(f"**{name2}**")
# Display the date
st.write('**May 19, 2024**')
file = st.file_uploader("**Choose a weather photo from your computer**", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size)
    img_array = np.asarray(image)
    img_array = img_array[np.newaxis, ...]  
    img_array = img_array / 255.0 

    prediction = model.predict(img_array)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    st.success(f"OUTPUT: {predicted_class_label}")



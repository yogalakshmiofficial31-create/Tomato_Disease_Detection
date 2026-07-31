import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to identify the disease.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_tomato_model.keras")

model = load_model()

with open("class_names.json", "r") as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {class_names[predicted_index]}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.progress(float(confidence / 100))

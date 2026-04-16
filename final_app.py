import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fsl_abc_tiny.h5")
    return model

model = load_model()

# -----------------------
# Preprocess
# -----------------------
def preprocess_image(image, target_size=(224,224)):
    image = image.resize(target_size)
    image = np.array(image)/255.0

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)
    return image

# -----------------------
# UI
# -----------------------
st.title("FSL Image Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Predict"):
        processed = preprocess_image(image)
        prediction = model.predict(processed)

        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Prediction: {pred_class}")
        st.info(f"Confidence: {confidence:.4f}")

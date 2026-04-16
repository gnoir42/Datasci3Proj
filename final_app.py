import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fsl_abc_tiny.h5")
    return model

model = load_model()

def preprocess_image(image, target_size=(64,64)):
    image = image.resize(target_size)
    image = np.array(image)/255.0

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)
    return image

st.title("FSL Image Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Predict"):
        processed = preprocess_image(image)

        prediction = model.predict(processed)

        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        st.success(f"Prediction: {pred_class}")
        st.info(f"Confidence: {confidence:.4f}")

import streamlit as st
import numpy as np
from PIL import Image
import pickle

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -----------------------
# Preprocess
# -----------------------
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0

    # remove alpha channel if PNG
    if image.shape[-1] == 4:
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)
    return image

# -----------------------
# UI
# -----------------------
st.title("FSL Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed = preprocess_image(image)

        try:
            prediction = model.predict(processed)

            pred_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.4f}")

        except Exception as e:
            st.error(f"Error: {e}")

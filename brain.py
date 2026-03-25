import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("brain_tumor_model.h5")
    return model

model = load_cnn_model()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🧠 Brain Tumor Detection")
st.sidebar.info(
    "This application uses a Convolutional Neural Network (CNN) "
    "to detect brain tumors from MRI images."
)

st.sidebar.markdown("### Instructions:")
st.sidebar.write("1. Upload an MRI image")
st.sidebar.write("2. Click on 'Predict'")
st.sidebar.write("3. View the result")

# -------------------------------
# Main Title
# -------------------------------
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to check for tumor presence.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image):
    img = np.array(image)

    # Convert grayscale to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    return img

# -------------------------------
# Prediction Button
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing MRI image..."):
            processed_image = preprocess_image(image)

            prediction = model.predict(processed_image)
            confidence = float(np.max(prediction))
            class_index = np.argmax(prediction)

        st.subheader("📊 Prediction Result")

        if class_index == 1:
            st.error("⚠️ Tumor Detected")
        else:
            st.success("✅ No Tumor Detected")

        st.write(f"**Confidence Score:** {confidence:.2f}")

        # Show probabilities
        st.markdown("### 🔢 Prediction Probabilities")
        st.write(f"No Tumor: {prediction[0][0]:.4f}")
        st.write(f"Tumor: {prediction[0][1]:.4f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("⚠️ Disclaimer: This is an AI-based demo and not a medical diagnosis tool.")
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

st.title("🥔 Potato Leaf Disease Classifier")

# Path to the model
model_path = 'S:/SPIT/Experiments/Shreeya_Nemade/potato-disease-classification/saved_models/potato_disease_model.h5'

# Load the model
with st.spinner("🔄 Loading model..."):
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.stop()
    try:
        model = load_model(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# Class labels (must match training order)
classes = ['Early Blight', 'Late Blight', 'Healthy']

# File uploader
uploaded_file = st.file_uploader("📁 Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = load_img(uploaded_file, target_size=(128, 128))
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show result
    st.subheader(f"🔍 Prediction: **{pred_class}** ({confidence * 100:.2f}%)")

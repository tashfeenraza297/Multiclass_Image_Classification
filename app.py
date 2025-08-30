import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json
import gdown
import os

# Set page config
st.set_page_config(page_title="Dog Breed Classifier", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .tagline {
        font-size: 24px;
        font-weight: bold;
        color: #FF6347;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 10px;
    }
    .main-subtitle {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
        font-style: italic;
    }
    .divider {
        border-top: 3px solid #1E90FF;
        margin: 20px 0;
    }
    .camera-button {
        background: linear-gradient(45deg, #1E90FF, #4682B4);
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        display: block;
        margin: 15px auto;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .camera-button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_tf_model():
    model_path = "model.keras"
    # Download model from Google Drive if it doesn't exist
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(
                "https://drive.google.com/file/d/18TWbFawj2xcVtXvoYrWxiskJRr38oOpi/view?usp=sharing",
                model_path,
                quiet=False
            )
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Ensure the downloaded 'model.keras' is valid.")
        raise

model = load_tf_model()

# Load class indices
try:
    with open('class_indices.json', 'r') as f:
        class_indices_original = json.load(f)
        class_indices = {str(v): k for k, v in class_indices_original.items()}
except FileNotFoundError:
    st.error("class_indices.json not found. Create it during training.")
    class_indices = {}
except Exception as e:
    st.error(f"Error loading class_indices.json: {str(e)}")
    class_indices = {}

num_classes = len(class_indices)

# Sidebar
with st.sidebar:
    st.header("About the Model")
    st.write("An advanced deep learning model trained to classify 120 dog breeds, built on InceptionResNetV2.")
    st.write(f"Number of Breeds: {num_classes}")
    st.write("Upload a dog image or use your camera to predict the breed and see confidence scores.")
    st.info("For best results, use clear, centered images of dogs.")
    st.markdown("### Top Breeds Example")
    st.write("Try it out with popular breeds (e.g., Labrador, Poodle) to see how the model performs.")

# Main content
st.markdown('<div class="tagline">🐶 Instantly identify your dog\'s breed with AI-powered image recognition!</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">🐾 Dog Breed Predictor 🐾</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Upload or capture a dog image to discover its breed with confidence!</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Initialize session state for camera
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# Image input (file upload or camera)
col_upload, col_camera = st.columns([1, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload a dog image...", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")
with col_camera:
    st.markdown("**Capture with Camera**", unsafe_allow_html=True)
    if st.button("📸 Take a Photo", key="camera_button", help="Click to open camera and capture a dog image"):
        st.session_state.show_camera = True
    camera_file = None
    if st.session_state.show_camera:
        camera_file = st.camera_input("Capture a dog image...", help="Use your camera to take a photo", key="camera_input")
        if st.button("Close Camera", key="close_camera"):
            st.session_state.show_camera = False

# Process image (from either upload or camera)
input_image = None
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
elif camera_file is not None:
    input_image = Image.open(camera_file)

if input_image is not None:
    # Display uploaded/captured image and predictions
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Input Image")
        st.image(input_image, use_column_width=True)
    
    # Preprocess the image
    img = input_image.resize((331, 331))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    with st.spinner("Analyzing the image..."):
        try:
            prediction = model.predict(img_array)[0]
            pred_idx = np.argmax(prediction)
            pred_breed = class_indices.get(str(pred_idx), "Unknown")
            confidence = prediction[pred_idx] * 100
            
            # Top 5 predictions
            top_indices = np.argsort(prediction)[-5:][::-1]
            top_breeds = [class_indices.get(str(i), "Unknown") for i in top_indices]
            top_confidences = [prediction[i] * 100 for i in top_indices]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            prediction = None
    
    if prediction is not None:
        # Display results
        with col2:
            st.header("Prediction Results")
            st.success(f"**Predicted Breed:** {pred_breed}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar
            st.write("Prediction Confidence:")
            st.progress(confidence / 100)
            
            # Top 5 table
            st.subheader("Top 5 Possible Breeds")
            top_df = pd.DataFrame({
                "Breed": top_breeds,
                "Confidence (%)": [f"{c:.2f}" for c in top_confidences]
            })
            st.table(top_df)
            
            # Bar chart
            fig, ax = plt.subplots()
            ax.barh(top_breeds, top_confidences, color='skyblue')
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Top 5 Breed Confidences")
            st.pyplot(fig)

# Additional sections
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.header("How It Works")
st.write("1. Upload or capture an image of a dog.")
st.write("2. The image is processed using a transfer learning model (InceptionResNetV2) trained on dog breed data.")
st.write(f"3. It predicts the breed from {num_classes} possible classes with confidence scores.")
st.write("4. View the top predictions and a visual bar chart.")

st.header("Tips for Better Predictions")
st.write("- Use high-quality, centered images of dogs.")
st.write("- Avoid images with multiple dogs or cluttered backgrounds.")
st.write("- If the prediction seems off, try cropping the image to focus on the dog.")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
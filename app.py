"""
Image Classifier with MobileNetV2
Author: [Your Name]
Description: A Streamlit web app that classifies images using deep learning
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

def load_model():
    """Load pre-trained MobileNetV2 model with ImageNet weights"""
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    """
    Prepare image for model prediction
    Args:
        image: PIL Image object
    Returns:
        Processed numpy array
    """
    img = image.resize((224, 224))  # MobileNet's expected input size
    img_array = np.array(img)
    
    # Handle potential grayscale images
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
        
    img_array = preprocess_input(img_array)  # Model-specific preprocessing
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def classify_image(model, image):
    """
    Make prediction and format results
    Args:
        model: Loaded MobileNetV2 model
        image: PIL Image object
    Returns:
        List of (class, description, probability) tuples
    """
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return decode_predictions(predictions, top=3)[0]

def main():
    """Main application interface"""
    st.set_page_config(
        page_title="AI Image Classifier",
        page_icon="🖼️",
        layout="centered"
    )
    
    st.title("Image Classification Demo")
    st.write("Upload an image to identify its contents using deep learning")
    
    # Load model with caching
    @st.cache_resource
    def get_model():
        return load_model()
    
    model = get_model()
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Classify Image"):
                with st.spinner("Analyzing..."):
                    predictions = classify_image(model, image)
                    
                    st.subheader("Prediction Results")
                    for _, label, prob in predictions:
                        st.write(f"- **{label.replace('_', ' ').title()}**: {prob:.1%} confidence")
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()

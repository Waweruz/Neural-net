import streamlit as st
import os

# Set TensorFlow environment variables BEFORE importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import json
from PIL import Image
import cv2
import pytesseract
import re

# Page configuration
st.set_page_config(
    page_title="Drink Label Classifier",
    page_icon="🍺",
    layout="wide"
)

# Title and description
st.title("🍺 OCR-Based Drink Classifier")
st.markdown("""
This app uses OCR (Optical Character Recognition) and deep learning to classify drink labels.
Upload an image of a drink, and the model will identify it!
""")

# Load model and configuration
@st.cache_resource
def load_model_and_config():
    """Load the trained model, tokenizer, and configuration"""
    try:
        # Load model with compile=False to avoid optimizer issues
        model = keras.models.load_model('best_ocr_drink_classifier.h5', compile=False)
        
        # Recompile with current TensorFlow version
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load config
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        return model, tokenizer, config
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please ensure model files are in the same directory as this app.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Error details: {str(e)}")
        return None, None, None

# OCR preprocessing
def preprocess_image_for_ocr(image):
    """Preprocess image to improve OCR accuracy"""
    try:
        # Convert PIL Image to OpenCV format
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Denoise
        gray = cv2.medianBlur(gray, 3)
        
        return gray
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def extract_text_from_image(image):
    """Extract text from drink label using OCR"""
    try:
        processed_img = preprocess_image_for_ocr(image)
        
        if processed_img is None:
            return ""
        
        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        
        # Clean the text
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def predict_drink(image, model, tokenizer, config):
    """Predict drink label from image"""
    try:
        # Extract text using OCR
        extracted_text = extract_text_from_image(image)
        
        if not extracted_text or len(extracted_text) < 3:
            return None, None, extracted_text, None, None
        
        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([extracted_text])
        padded = keras.preprocessing.sequence.pad_sequences(
            sequence, 
            maxlen=config['max_length'], 
            padding='post', 
            truncating='post'
        )
        
        # Predict
        predictions = model.predict(padded, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        # Get top 5 predictions
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_5_labels = [config['labels'][idx] for idx in top_5_idx]
        top_5_probs = [predictions[0][idx] for idx in top_5_idx]
        
        predicted_label = config['labels'][predicted_idx]
        
        return predicted_label, confidence, extracted_text, top_5_labels, top_5_probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, "", None, None

# Main app
def main():
    # Load model
    model, tokenizer, config = load_model_and_config()
    
    if model is None:
        st.warning("⚠️ Model not loaded. Please ensure all model files are present.")
        st.info("""
        Required files:
        - `best_ocr_drink_classifier.h5`
        - `tokenizer.pkl`
        - `model_config.json`
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This classifier recognizes **30 different drink brands** including:
        - Beers (Tusker, Guinness, Heineken, etc.)
        - Wines (4th Street Wine, etc.)
        - Spirits (Captain Morgan, Johnnie Walker, etc.)
        - Soft drinks (Coca Cola, Pepsi, Fanta, etc.)
        
        **How it works:**
        1. Upload an image of a drink
        2. OCR extracts text from the label
        3. Deep learning model classifies the drink
        """)
        
        st.header("📊 Model Info")
        st.write(f"**Classes:** {config['num_classes']}")
        st.write(f"**Vocabulary Size:** {config['vocab_size']}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image of a drink...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Add a predict button
            if st.button("🚀 Classify Drink", type="primary"):
                with st.spinner("Analyzing image..."):
                    result = predict_drink(image, model, tokenizer, config)
                    
                    if result[0] is None:
                        st.error("❌ Could not extract readable text from image")
                        st.info("Tips: Ensure the image is clear and the text is visible")
                    else:
                        predicted_label, confidence, extracted_text, top_5_labels, top_5_probs = result
                        
                        # Display results
                        st.success(f"**Predicted Drink:** {predicted_label}")
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                        
                        # Show extracted text
                        with st.expander("📝 Extracted Text (OCR)"):
                            st.code(extracted_text if extracted_text else "No text extracted")
                        
                        # Show top 5 predictions
                        with st.expander("📊 Top 5 Predictions"):
                            for i, (label, prob) in enumerate(zip(top_5_labels, top_5_probs), 1):
                                st.write(f"{i}. **{label}** - {prob*100:.2f}%")
                                st.progress(float(prob))
    
    # Example section
    st.markdown("---")
    st.subheader("💡 Tips for Best Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📸 Image Quality**
        - Use clear, well-lit photos
        - Avoid blurry images
        - Capture the full label
        """)
    
    with col2:
        st.markdown("""
        **🎯 Label Visibility**
        - Ensure text is readable
        - Avoid glare or reflections
        - Front-facing angle works best
        """)
    
    with col3:
        st.markdown("""
        **🔧 Troubleshooting**
        - Try different angles
        - Adjust lighting
        - Use higher resolution images
        """)

if __name__ == "__main__":
    main()

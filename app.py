import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import json
from PIL import Image
import cv2
import pytesseract
import re
import io
import os

# Check if TensorFlow loaded successfully
if tf is None:
    st.error("TensorFlow failed to load in this environment.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Drink Label Classifier",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Drink labels
DRINK_LABELS = [
    'Chrome', 'Fanta', '4th Street Wine', 'Captain Morgan', 'Johnnie Walker',
    'Tusker', 'Guarana', 'Smirnoff Ice', 'Predator', 'Coca Cola',
    'Spirit', 'Kibao', 'Guinness', 'Afia', 'Novida',
    'White Cap', 'Senator Keg', 'Gilbeys Gin', 'Heineken', 'Pilsner',
    'Hunters', 'Allsops', 'Balozi', 'Kingfisher', 'Jameson',
    'Richot', 'Bond7', 'Del Monte Juice', 'Minute Maid', 'Pepsi'
]

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Check if model files exist locally
        model_paths = ['best_ocr_drink_classifier.h5', 'ocr_drink_classifier_final.h5']
        model = None
        
        # Try loading from local files first
        for path in model_paths:
            if os.path.exists(path):
                model = keras.models.load_model(path)
                st.success(f"✅ Model loaded from {path}")
                break
        
        # If no local model, try downloading from cloud (optional)
        if model is None:
            st.warning("⚠️ Model files not found locally.")
            
            # Option to download from Google Drive or other cloud storage
            # Uncomment and add your model URLs if using cloud storage:
            
            # MODEL_URL = "YOUR_GOOGLE_DRIVE_DIRECT_LINK"
            # TOKENIZER_URL = "YOUR_TOKENIZER_DIRECT_LINK"
            # CONFIG_URL = "YOUR_CONFIG_DIRECT_LINK"
            
            # import requests
            # with st.spinner("Downloading model from cloud..."):
            #     # Download model
            #     r = requests.get(MODEL_URL)
            #     with open('model.h5', 'wb') as f:
            #         f.write(r.content)
            #     model = keras.models.load_model('model.h5')
            
            st.error("❌ Model file not found. Please add model files to the repository or configure cloud storage.")
            return None, None, None
        
        # Load tokenizer
        if not os.path.exists('tokenizer.pkl'):
            st.error("❌ tokenizer.pkl not found")
            return None, None, None
            
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load config
        if not os.path.exists('model_config.json'):
            st.error("❌ model_config.json not found")
            return None, None, None
            
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        return model, tokenizer, config
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please check the logs for more details.")
        return None, None, None

def preprocess_image_for_ocr(image):
    """Preprocess image for better OCR results"""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply thresholding
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Denoise
        gray = cv2.medianBlur(gray, 3)
        
        return gray
    
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        processed_img = preprocess_image_for_ocr(image)
        
        if processed_img is None:
            return ""
        
        # Extract text
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        
        # Clean text
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        return text
    
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def predict_drink(image, model, tokenizer, config):
    """Predict drink label from image"""
    try:
        # Extract text using OCR
        with st.spinner("🔍 Extracting text from image..."):
            extracted_text = extract_text_from_image(image)
        
        if not extracted_text or len(extracted_text) < 2:
            st.warning("⚠️ Could not extract meaningful text from the image.")
            return None, None, None
        
        # Display extracted text
        st.info(f"📝 Extracted Text: `{extracted_text}`")
        
        # Tokenize and pad
        max_length = config['max_length']
        sequence = tokenizer.texts_to_sequences([extracted_text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=max_length, padding='post', truncating='post'
        )
        
        # Predict
        with st.spinner("🤖 Classifying drink..."):
            prediction = model.predict(padded_sequence, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(prediction[0])[::-1][:5]
        top_labels = [DRINK_LABELS[i] for i in top_indices]
        top_confidences = [prediction[0][i] * 100 for i in top_indices]
        
        predicted_label = top_labels[0]
        confidence = top_confidences[0]
        
        return predicted_label, confidence, list(zip(top_labels, top_confidences))
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    # Header
    st.markdown('<div class="main-header">🍺 Drink Label Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered OCR-based drink recognition system</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This app uses OCR (Optical Character Recognition) and deep learning to identify drink labels from images.
        
        **How it works:**
        1. Upload an image of a drink label
        2. OCR extracts text from the image
        3. AI model classifies the drink
        
        **Supported Drinks:**
        30 different beverages including beers, wines, sodas, and spirits.
        """)
        
        st.header("⚙️ Settings")
        show_top_predictions = st.checkbox("Show top 5 predictions", value=True)
        show_extracted_text = st.checkbox("Show extracted text details", value=False)
        
        st.header("📊 Statistics")
        st.metric("Total Drink Classes", len(DRINK_LABELS))
        st.metric("Model Type", "LSTM/CNN/GRU")
    
    # Load model
    model, tokenizer, config = load_model_and_tokenizer()
    
    if model is None:
        st.error("⚠️ Please ensure the model files are in the repository:")
        st.code("""
        - best_ocr_drink_classifier.h5 (or ocr_drink_classifier_final.h5)
        - tokenizer.pkl
        - model_config.json
        """)
        st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a drink label image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a drink label"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔮 Classify Drink", type="primary", use_container_width=True):
                predicted_label, confidence, top_predictions = predict_drink(
                    image, model, tokenizer, config
                )
                
                if predicted_label is not None:
                    # Store in session state
                    st.session_state.predicted_label = predicted_label
                    st.session_state.confidence = confidence
                    st.session_state.top_predictions = top_predictions
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if 'predicted_label' in st.session_state:
            predicted_label = st.session_state.predicted_label
            confidence = st.session_state.confidence
            top_predictions = st.session_state.top_predictions
            
            # Main prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("Predicted Drink:")
            st.markdown(f"# {predicted_label}")
            
            # Confidence with color coding
            if confidence > 80:
                conf_class = "confidence-high"
            elif confidence > 50:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
            
            st.markdown(f'<p class="{conf_class}">Confidence: {confidence:.2f}%</p>', 
                       unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Top predictions
            if show_top_predictions and top_predictions:
                st.subheader("Top 5 Predictions:")
                
                for i, (label, conf) in enumerate(top_predictions, 1):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"{i}. **{label}**")
                    with col_b:
                        st.write(f"{conf:.2f}%")
                    st.progress(conf / 100)
            
            # Download results
            st.download_button(
                label="📥 Download Results",
                data=json.dumps({
                    'predicted_drink': predicted_label,
                    'confidence': f"{confidence:.2f}%",
                    'top_5_predictions': [
                        {'rank': i+1, 'drink': label, 'confidence': f"{conf:.2f}%"}
                        for i, (label, conf) in enumerate(top_predictions)
                    ]
                }, indent=2),
                file_name="prediction_results.json",
                mime="application/json"
            )
        else:
            st.info("👆 Upload an image and click 'Classify Drink' to see results")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
        <p>OCR-Based Drink Label Classification System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
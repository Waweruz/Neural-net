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
import os
import time

# Page configuration
st.set_page_config(
    page_title="DrinkID - AI Drink Identifier",
    page_icon="🍹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful interface
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in-out;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-in-out;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        animation: scaleIn 0.5s ease-in-out;
        margin: 2rem 0;
    }
    
    .prediction-drink {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .prediction-confidence {
        font-size: 1.5rem;
        color: rgba(255,255,255,0.95);
        font-weight: 500;
    }
    
    .confidence-bar {
        background: rgba(255,255,255,0.3);
        border-radius: 50px;
        height: 20px;
        margin: 1.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%);
        height: 100%;
        border-radius: 50px;
        transition: width 1s ease-in-out;
        box-shadow: 0 0 20px rgba(74, 222, 128, 0.5);
    }
    
    .top-predictions {
        background: #f8fafc;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .prediction-item {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .prediction-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .medal {
        font-size: 2rem;
        margin-right: 1rem;
    }
    
    .drink-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
        flex-grow: 1;
    }
    
    .confidence-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .success-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        animation: pulse 2s infinite;
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        display: inline-block;
        font-weight: 600;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 5px solid #0ea5e9;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        font-size: 1.3rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
    }
    
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        model = keras.models.load_model('best_ocr_drink_classifier.h5')
        
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        return model, tokenizer, config
    except Exception as e:
        return None, None, None

# Image processing functions
def preprocess_image_for_analysis(image):
    """Process image for AI analysis"""
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray

def analyze_drink_image(image):
    """Analyze drink image using AI"""
    try:
        processed_img = preprocess_image_for_analysis(image)
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    except:
        return ""

def identify_drink(model, tokenizer, config, image):
    """Identify drink from image"""
    text = analyze_drink_image(image)
    
    if not text or len(text) < 2:
        return None
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, 
        maxlen=config['max_length'], 
        padding='post'
    )
    
    predictions = model.predict(padded, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        (config['labels'][i], predictions[0][i]) 
        for i in top_3_indices
    ]
    
    return config['labels'][predicted_class], confidence, top_3_predictions

# Main App
def main():
    # Header
    st.markdown('<h1 class="hero-title">🍹 DrinkID</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-Powered Drink Identification System</p>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, config = load_model_and_tokenizer()
    
    if model is None:
        st.error("⚠️ AI Model not loaded. Please train the model first by running `python train.py`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 About DrinkID")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; color: white;">
            <p style="font-size: 1.1rem; line-height: 1.8;">
                <strong>DrinkID</strong> uses advanced AI technology to instantly identify drinks from photos.
                Simply upload an image and let our intelligent system do the rest!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("## ⚙️ Settings")
        show_alternatives = st.checkbox("Show alternative matches", value=True)
        confidence_threshold = st.slider("Confidence threshold (%)", 0, 100, 60)
        
        st.markdown("---")
        
        st.markdown("## 📊 Supported Drinks")
        st.info(f"**{len(config['labels'])} drink brands** recognized")
        
        with st.expander("View all drinks"):
            for i, drink in enumerate(config['labels'], 1):
                st.write(f"{i}. {drink}")
    
    # Features Section
    st.markdown("### ✨ Why Choose DrinkID?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Instant Recognition</h3>
            <p>Get results in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3>High Accuracy</h3>
            <p>Advanced AI technology</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌟</div>
            <h3>30+ Brands</h3>
            <p>Wide drink coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main upload section
    st.markdown("### 📸 Upload Drink Image")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of your drink"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🖼️ Your Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Uploaded drink image")
            
            if st.button("🔍 Identify Drink", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing your drink..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    result = identify_drink(model, tokenizer, config, image)
                    
                    if result and result[0] is not None:
                        st.session_state.prediction = result[0]
                        st.session_state.confidence = result[1]
                        st.session_state.top_3 = result[2]
                        st.session_state.identified = True
                    else:
                        st.error("❌ Could not identify the drink. Please try a clearer image.")
        
        with col2:
            st.markdown("#### 🎯 Identification Results")
            
            if 'identified' in st.session_state and st.session_state.identified:
                predicted_drink = st.session_state.prediction
                confidence = st.session_state.confidence
                top_3 = st.session_state.top_3
                
                # Main prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-drink">🍾 {predicted_drink}</div>
                    <div class="prediction-confidence">Confidence: {confidence*100:.1f}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence badge
                if confidence * 100 >= confidence_threshold:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="success-badge">✅ High Confidence Match</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="warning-badge">⚠️ Low Confidence - Try Another Image</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top 3 predictions
                if show_alternatives:
                    st.markdown("### 🏆 Alternative Matches")
                    st.markdown('<div class="top-predictions">', unsafe_allow_html=True)
                    
                    medals = ["🥇", "🥈", "🥉"]
                    for i, (drink, conf) in enumerate(top_3):
                        st.markdown(f"""
                        <div class="prediction-item">
                            <span class="medal">{medals[i]}</span>
                            <span class="drink-name">{drink}</span>
                            <span class="confidence-badge">{conf*100:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-card">
                    <h3>👆 Ready to identify!</h3>
                    <p>Click the "Identify Drink" button to start the AI analysis.</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👆 Upload a drink image above to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem;">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            <strong>DrinkID</strong> - Powered by Advanced AI Technology
        </p>
        <p style="font-size: 0.9rem;">
            Accurately identifying 30+ drink brands with machine learning
        </p>
        <p style="font-size: 0.8rem; margin-top: 1rem; color: #94a3b8;">
            Built with ❤️ using TensorFlow and Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import json
import os
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import cv2
import re

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and tokenizer
model = None
tokenizer = None
config = None

DRINK_LABELS = [
    'Chrome', 'Fanta', '4th Street Wine', 'Captain Morgan', 'Johnnie Walker',
    'Tusker', 'Guarana', 'Smirnoff Ice', 'Predator', 'Coca Cola',
    'Spirit', 'Kibao', 'Guinness', 'Afia', 'Novida',
    'White Cap', 'Senator Keg', 'Gilbeys Gin', 'Heineken', 'Pilsner',
    'Hunters', 'Allsops', 'Balozi', 'Kingfisher', 'Jameson',
    'Richot', 'Bond7', 'Del Monte Juice', 'Minute Maid', 'Pepsi'
]

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    global model, tokenizer, config
    
    try:
        # Load model
        model_path = 'best_ocr_drink_classifier.h5'
        if not os.path.exists(model_path):
            model_path = 'ocr_drink_classifier_final.h5'
        
        model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        print("✓ Tokenizer loaded")
        
        # Load config
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        print("✓ Config loaded")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_ocr(image_path):
    """Preprocess image to improve OCR accuracy"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Denoise
    gray = cv2.medianBlur(gray, 3)
    
    return gray

def extract_text_from_image(image_path):
    """Extract text from drink label using OCR"""
    try:
        processed_img = preprocess_image_for_ocr(image_path)
        if processed_img is None:
            return ""
        
        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        
        # Clean the text
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def predict_drink(text):
    """Predict drink label from extracted text"""
    if not text or len(text) < 2:
        return None, None, []
    
    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, 
        maxlen=config['max_length'], 
        padding='post', 
        truncating='post'
    )
    
    # Predict
    predictions = model.predict(padded, verbose=0)[0]
    
    # Get top 5 predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_predictions = [
        {
            'label': DRINK_LABELS[idx],
            'confidence': float(predictions[idx] * 100)
        }
        for idx in top_indices
    ]
    
    # Get best prediction
    best_idx = np.argmax(predictions)
    best_label = DRINK_LABELS[best_idx]
    best_confidence = float(predictions[best_idx] * 100)
    
    return best_label, best_confidence, top_predictions

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict drink from uploaded image"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text using OCR
        extracted_text = extract_text_from_image(filepath)
        
        if not extracted_text:
            os.remove(filepath)
            return jsonify({
                'error': 'No text could be extracted from the image',
                'suggestion': 'Please ensure the image contains readable text and is not blurry'
            }), 400
        
        # Predict drink
        label, confidence, top_predictions = predict_drink(extracted_text)
        
        # Clean up
        os.remove(filepath)
        
        if label is None:
            return jsonify({
                'error': 'Could not classify the drink',
                'extracted_text': extracted_text
            }), 400
        
        return jsonify({
            'success': True,
            'prediction': {
                'label': label,
                'confidence': round(confidence, 2)
            },
            'extracted_text': extracted_text,
            'top_predictions': top_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-text', methods=['POST'])
def predict_from_text():
    """Predict drink from text input"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip().lower()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        label, confidence, top_predictions = predict_drink(text)
        
        if label is None:
            return jsonify({'error': 'Could not classify the drink'}), 400
        
        return jsonify({
            'success': True,
            'prediction': {
                'label': label,
                'confidence': round(confidence, 2)
            },
            'top_predictions': top_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/labels', methods=['GET'])
def get_labels():
    """Get all available drink labels"""
    return jsonify({
        'labels': DRINK_LABELS,
        'count': len(DRINK_LABELS)
    })

if __name__ == '__main__':
    print("Loading model and tokenizer...")
    if load_model_and_tokenizer():
        print("✓ Ready to serve predictions!")
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("✗ Failed to load model. Please ensure model files exist.")
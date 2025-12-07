"""
Train the drink classifier model
Run this script before using the Streamlit app
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pytesseract
from PIL import Image
import cv2
import os
import re
from pathlib import Path
import pickle
import json

# Drink labels (30 classes)
DRINK_LABELS = [
    'Chrome', 'Fanta', '4th Street Wine', 'Captain Morgan', 'Johnnie Walker',
    'Tusker', 'Guarana', 'Smirnoff Ice', 'Predator', 'Coca Cola',
    'Spirit', 'Kibao', 'Guinness', 'Afia', 'Novida',
    'White Cap', 'Senator Keg', 'Gilbeys Gin', 'Heineken', 'Pilsner',
    'Hunters', 'Allsops', 'Balozi', 'Kingfisher', 'Jameson',
    'Richot', 'Bond7', 'Del Monte Juice', 'Minute Maid', 'Pepsi'
]

NUM_CLASSES = len(DRINK_LABELS)
label_to_idx = {label: idx for idx, label in enumerate(DRINK_LABELS)}

# Image processing functions
def preprocess_image_for_ocr(image_path):
    """Preprocess image for OCR"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    
    return gray

def extract_text_from_image(image_path):
    """Extract text using OCR"""
    try:
        processed_img = preprocess_image_for_ocr(image_path)
        if processed_img is None:
            return ""
        
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        return text
    except Exception as e:
        print(f"Error: {e}")
        return ""

def build_dataset_from_images(dataset_path):
    """Extract text from images"""
    print("\n" + "="*60)
    print("EXTRACTING TEXT FROM IMAGES")
    print("="*60)
    
    texts = []
    labels = []
    
    for label in DRINK_LABELS:
        label_folder = os.path.join(dataset_path, label)
        
        if not os.path.exists(label_folder):
            print(f"⚠ Folder not found: {label}")
            continue
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(Path(label_folder).glob(ext)))
        
        if len(image_files) == 0:
            print(f"⚠ No images in {label}")
            continue
        
        print(f"Processing '{label}': {len(image_files)} images...")
        
        for img_path in image_files:
            extracted_text = extract_text_from_image(str(img_path))
            if extracted_text and len(extracted_text) > 3:
                texts.append(extracted_text)
                labels.append(label_to_idx[label])
    
    print(f"\n✓ Total samples: {len(texts)}")
    return texts, labels

def generate_synthetic_data(samples_per_class=100):
    """Generate synthetic training data"""
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC DATA")
    print("="*60)
    
    texts = []
    labels = []
    
    context_words = ['', 'bottle', 'drink', 'beer', 'wine', 'juice', 'soda', 
                     'premium', 'cold', 'fresh', 'original', 'classic']
    
    for label in DRINK_LABELS:
        for _ in range(samples_per_class):
            label_lower = label.lower()
            
            if np.random.random() > 0.5:
                context = np.random.choice(context_words)
                text = f"{context} {label_lower}" if context else label_lower
            else:
                text = label_lower
            
            if np.random.random() > 0.7:
                text = text.replace(' ', '')
            
            texts.append(text)
            labels.append(label_to_idx[label])
    
    print(f"✓ Generated {len(texts)} samples")
    return texts, labels

def preprocess_texts(texts, labels):
    """Preprocess text data"""
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    max_length = 50
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    labels_cat = keras.utils.to_categorical(labels, NUM_CLASSES)
    
    return padded, labels_cat, tokenizer, max_length

def create_model(vocab_size, max_length):
    """Create LSTM model"""
    model = models.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_length),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.5),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def train_model(dataset_path=None, use_synthetic=True, epochs=50, batch_size=32):
    """Main training function"""
    print("\n" + "="*60)
    print("DRINK CLASSIFIER TRAINING")
    print("="*60)
    
    # Get data
    if use_synthetic or dataset_path is None:
        texts, labels = generate_synthetic_data(100)
    else:
        texts, labels = build_dataset_from_images(dataset_path)
    
    if len(texts) == 0:
        print("❌ No training data!")
        return
    
    # Preprocess
    X, y, tokenizer, max_length = preprocess_texts(texts, labels)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining: {len(X_train)}, Testing: {len(X_test)}")
    
    # Build model
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(vocab_size, max_length)
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint('best_ocr_drink_classifier.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Save
    model.save('ocr_drink_classifier_final.h5')
    
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open('model_config.json', 'w') as f:
        json.dump({
            'max_length': max_length,
            'vocab_size': vocab_size,
            'num_classes': NUM_CLASSES,
            'labels': DRINK_LABELS
        }, f, indent=2)
    
    print("\n✓ Model saved!")
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    
    return model, tokenizer

if __name__ == "__main__":
    # CONFIGURATION
    DATASET_PATH = None  # Set your dataset path here
    USE_SYNTHETIC = True  # Set False when you have real images
    EPOCHS = 50
    BATCH_SIZE = 32
    
    print("="*60)
    print("DRINKID TRAINING SCRIPT")
    print("="*60)
    
    train_model(
        dataset_path=DATASET_PATH,
        use_synthetic=USE_SYNTHETIC,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("  - best_ocr_drink_classifier.h5")
    print("  - tokenizer.pkl")
    print("  - model_config.json")
    print("\nNow run: streamlit run app.py")
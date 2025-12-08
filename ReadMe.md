# 🍺 Drink Label Classifier

AI-powered drink classification system using OCR and deep learning to identify 30 different drink brands from images.

## 🎯 Features

- **Image Upload**: Drag & drop or click to upload drink label images
- **OCR Processing**: Automatically extracts text from images using Tesseract
- **Text Classification**: Predicts drink brand using trained neural network
- **Top 5 Predictions**: Shows confidence scores for top 5 matches
- **REST API**: Full API for integration with other applications
- **Web Interface**: Beautiful, responsive UI for easy testing

## 🏷️ Supported Drinks (30 Classes)

Chrome, Fanta, 4th Street Wine, Captain Morgan, Johnnie Walker, Tusker, Guarana, Smirnoff Ice, Predator, Coca Cola, Spirit, Kibao, Guinness, Afia, Novida, White Cap, Senator Keg, Gilbeys Gin, Heineken, Pilsner, Hunters, Allsops, Balozi, Kingfisher, Jameson, Richot, Bond7, Del Monte Juice, Minute Maid, Pepsi

## 🚀 Quick Start

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/drink-classifier.git
cd drink-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR**
- **Ubuntu/Linux**: `sudo apt-get install tesseract-ocr`
- **Mac**: `brew install tesseract`
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Run the application**
```bash
python app.py
```

5. **Open browser**
Navigate to `http://localhost:5000`

## 📡 API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Get All Labels
```bash
GET /labels
```

### 3. Predict from Image
```bash
POST /predict
Content-Type: multipart/form-data
Parameters: file (jpg, jpeg, png)
```

### 4. Predict from Text
```bash
POST /predict-text
Content-Type: application/json
Body: {"text": "coca cola"}
```

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **ML Framework**: TensorFlow/Keras
- **OCR**: Tesseract + OpenCV
- **Model**: LSTM/CNN/GRU for text classification
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render

## 📊 Model Architecture

The classifier uses a neural network trained on OCR-extracted text:
- **Input**: Text sequences from drink labels
- **Embedding Layer**: 128-dimensional word embeddings
- **Recurrent Layers**: Bidirectional LSTM/GRU
- **Output**: 30-class softmax classifier

## 📝 Project Structure

```
drink-classifier/
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config
├── best_ocr_drink_classifier.h5    # Trained model
├── tokenizer.pkl                   # Text tokenizer
├── model_config.json               # Model configuration
├── templates/
│   └── index.html                  # Web interface
└── uploads/                        # Temporary uploads
```

## 🌐 Deployment on Render

1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Create new Blueprint
4. Connect your repository
5. Render will automatically deploy using `render.yaml`

## 🧪 Testing

```bash
# Health check
curl https://your-app.onrender.com/health

# Text prediction
curl -X POST https://your-app.onrender.com/predict-text \
  -H "Content-Type: application/json" \
  -d '{"text": "coca cola"}'

# Image upload
curl -X POST https://your-app.onrender.com/predict \
  -F "file=@image.jpg"
```

## 📄 License

MIT License

## 👥 Contributors

Your Name/Team

---

**Made with ❤️ for drink classification**
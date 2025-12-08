# 🍺 Drink Label Classifier

An AI-powered web application that uses OCR (Optical Character Recognition) and deep learning to automatically identify and classify drink labels from images.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **OCR Text Extraction**: Automatically extracts text from drink label images using Tesseract OCR
- **AI Classification**: Uses deep learning (LSTM/CNN/GRU) to classify drinks into 30 categories
- **Real-time Predictions**: Instant classification with confidence scores
- **Top-5 Predictions**: Shows the most likely drink matches
- **User-friendly Interface**: Clean and intuitive Streamlit web interface
- **Export Results**: Download prediction results as JSON

## 🍹 Supported Drinks (30 Classes)

The system can identify the following drinks:

- **Beers**: Tusker, Guinness, White Cap, Senator Keg, Heineken, Pilsner, Hunters, Allsops, Balozi, Kingfisher
- **Spirits**: Captain Morgan, Johnnie Walker, Smirnoff Ice, Gilbeys Gin, Jameson
- **Wines**: 4th Street Wine
- **Sodas**: Coca Cola, Fanta, Pepsi, Predator, Guarana, Novida
- **Energy Drinks**: Chrome, Spirit, Kibao, Afia
- **Juices**: Del Monte Juice, Minute Maid
- **Others**: Richot, Bond7

## 🚀 Live Demo

[Deploy on Streamlit Cloud](https://streamlit.io/)

## 📋 Prerequisites

Before running this project, you need:

1. **Python 3.9 or higher**
2. **Tesseract OCR** installed on your system:
   - **Windows**: Download from [GitHub Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Mac**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

3. **Trained Model Files** (required):
   - `best_ocr_drink_classifier.h5` or `ocr_drink_classifier_final.h5`
   - `tokenizer.pkl`
   - `model_config.json`

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/drink-label-classifier.git
cd drink-label-classifier
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Model Files

Place your trained model files in the project root directory:
- `best_ocr_drink_classifier.h5`
- `tokenizer.pkl`
- `model_config.json`

**Note**: These files are generated from the training script. If you don't have them, you need to train the model first using the provided training code.

## 💻 Usage

### Running Locally

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the App

1. **Upload Image**: Click "Browse files" and select a drink label image
2. **Classify**: Click the "🔮 Classify Drink" button
3. **View Results**: See the predicted drink name and confidence score
4. **Export**: Download results as JSON if needed

## 📁 Project Structure

```
drink-label-classifier/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── .streamlit/
│   └── config.toml                # Streamlit configuration
│
├── best_ocr_drink_classifier.h5   # Trained model (not in repo)
├── tokenizer.pkl                   # Tokenizer (not in repo)
├── model_config.json              # Model config (not in repo)
│
└── training/
    └── train_model.py             # Training script (optional)
```

## 🎯 Training Your Own Model

To train the model from scratch:

1. Organize your dataset:
```
dataset/
├── Chrome/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Fanta/
│   ├── image1.jpg
│   └── ...
└── ...
```

2. Run the training script:
```python
python train_model.py
```

3. The trained model files will be generated:
   - `best_ocr_drink_classifier.h5`
   - `tokenizer.pkl`
   - `model_config.json`

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

**Important**: You'll need to:
- Add `packages.txt` with `tesseract-ocr` for Linux deployment
- Ensure model files are in the repository or use cloud storage

### Deploy to Heroku

```bash
heroku create your-app-name
git push heroku main
```

Add a `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

## ⚙️ Configuration

Edit `.streamlit/config.toml` to customize:

- Theme colors
- Server settings
- Upload limits
- Browser settings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Known Issues

- OCR accuracy depends on image quality and lighting
- Low-resolution images may not work well
- Text needs to be clearly visible in the image

## 🔮 Future Enhancements

- [ ] Add batch processing for multiple images
- [ ] Improve OCR preprocessing for better accuracy
- [ ] Add image augmentation for training
- [ ] Support for more drink categories
- [ ] Mobile app version
- [ ] API endpoint for integration

## 👥 Authors

- **Your Name** - [GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit for the amazing web framework
- Tesseract OCR for text extraction capabilities
- All contributors and users of this project

## 📧 Contact

For questions or support, please open an issue or contact:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Made with ❤️ and Python**
# 🧠 Brain Tumor Detection System

A professional-grade AI-powered web application for early-stage brain tumor detection using deep learning and medical imaging analysis.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Results](#results)
- [Disclaimer](#disclaimer)

## ✨ Features

### 🤖 AI Model
- **EfficientNetB4** architecture with transfer learning
- **4-class classification**: Glioma, Meningioma, No Tumor, Pituitary
- Optimized for 224×224 RGB medical images
- High accuracy with confidence scores

### 🎨 Web Interface
- **Streamlit-based** professional UI
- Single image prediction
- Batch image analysis
- Real-time confidence visualization
- Detailed diagnostic reports

### 📊 Analysis Features
- Per-class confidence scores
- ROC curves and confusion matrices
- Training history visualization
- Detailed performance metrics

### 🔒 Professional Standards
- Medical-grade data preprocessing
- Regularization and dropout
- Early stopping and learning rate optimization
- Comprehensive validation strategy

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│      Input Brain MRI Image              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Image Preprocessing & Normalization   │
│   (224×224, RGB, 0-1 range)            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   EfficientNetB4 (Transfer Learning)    │
│   - Pre-trained on ImageNet             │
│   - Fine-tuned on brain tumor data      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Dense Layers + Dropout Regularization │
│   - 512 → 256 → 128 neurons            │
│   - Batch Normalization                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Softmax Output (4 classes)            │
│   - Confidence scores for each class    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Result: Class + Confidence Score      │
└─────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- 8GB+ RAM recommended

### Setup Steps

1. **Clone repository**
```bash
https://github.com/mianhussnain/brain-tumor.git
```

2. **Create virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify data structure**
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 🎯 Usage

### 1. Training the Model

```bash
python train.py

OR

python main.py train
```

This will:
- Load and preprocess all training data
- Build EfficientNetB4 model with transfer learning
- Train with data augmentation
- Fine-tune with unfrozen layers
- Evaluate on test set
- Save model and training history
- Generate performance plots

**Expected Output:**
- Trained model: `models/brain_tumor_model_final.keras`
- Best model checkpoint: `models/brain_tumor_model_best.keras`
- Training plots: `models/training_history.png`
- Confusion matrix: `models/confusion_matrix.png`

### 2. Running the Web Application

```bash
python main.py app
```

Then open your browser to: `http://localhost:8501`

**Features:**
- **Single Prediction**: Upload one image for diagnosis
- **Batch Analysis**: Analyze multiple images at once
- **Visualization**: Interactive charts and confidence scores
- **Detailed Reports**: Per-class prediction breakdown

### 3. Making Predictions Programmatically

```python
from predict import BrainTumorPredictor
from PIL import Image

# Initialize predictor
predictor = BrainTumorPredictor(model_path="models/brain_tumor_model_final.keras")

# Make prediction
result = predictor.predict("path/to/image.jpg")

# Print results
print(f"Predicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"High Confidence: {result['is_confident']}")

# View all predictions
for cls, conf in result['sorted_predictions']:
    print(f"{cls}: {conf:.2%}")
```

## 📁 Project Structure

```
brain-tumor/
├── data/                          # Dataset
│   ├── Training/                  # Training images (4 classes)
│   └── Testing/                   # Testing images (4 classes)
├── models/                        # Saved models & outputs
│   ├── brain_tumor_model_final.keras
│   ├── brain_tumor_model_best.keras
│   ├── training_history.png
│   └── confusion_matrix.png
├── logs/                          # TensorBoard logs
├── config.py                      # Configuration settings
├── data_loader.py                 # Data preprocessing & augmentation
├── model.py                       # Model architecture & training
├── train.py                       # Training script
├── predict.py                     # Prediction/inference module
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## 🧠 Model Details

### Architecture: EfficientNetB4

**Why EfficientNetB4?**
- Superior accuracy on medical imaging datasets
- Optimal balance between performance and efficiency
- Robust to variations in image quality
- Pre-trained on 1.2M ImageNet images
- Scales efficiently to large datasets

### Training Strategy

1. **Initial Training**
   - Frozen base model layers
   - Train top layers only
   - Learns tumor-specific features

2. **Fine-tuning**
   - Unfreeze last 30 base model layers
   - Lower learning rate (0.1x)
   - Adapts pre-trained features to medical domain

### Data Augmentation

Applied to training data:
- Rotation: ±20°
- Width/Height shift: ±20%
- Zoom: ±20%
- Horizontal/Vertical flip
- Brightness: 0.8-1.2x
- Fill mode: Nearest neighbor

### Loss Function & Metrics

- **Loss**: Categorical Crossentropy
- **Metrics**: 
  - Accuracy
  - Precision
  - Recall
  - AUC-ROC

### Regularization

- L2 regularization (λ=1e-4)
- Dropout (0.2-0.4)
- Batch Normalization
- Early stopping (patience=15)
- Learning rate reduction (patience=5, factor=0.5)

## 📊 Results

### Expected Performance

After training on your dataset:

```
Test Set Results:
  Loss:      ~0.43
  Accuracy:  ~87.31%
  Precision: ~87.85%
  Recall:    ~87.31%
  AUC:       ~0.98

Per-Class Performance (Accuracy/Recall):
  Glioma:     ~67.25%
  Meningioma: ~84.25%
  No Tumor:   ~99.50%
  Pituitary:  ~98.25%
```

### Confidence Levels

- **High (≥90%)**: Very reliable prediction
- **Moderate (60-75%)**: Use with caution
- **Low (<60%)**: Requires professional review

## ⚠️ Disclaimer

### Important Legal Notice

**THIS SYSTEM IS FOR RESEARCH AND ASSISTANCE PURPOSES ONLY**

1. **Not a Diagnostic Tool**: This AI system is NOT designed to replace professional medical diagnosis
2. **Professional Review Required**: All predictions must be verified by qualified radiologists and physicians
3. **Accuracy Limitations**: The model may have false positives or false negatives
4. **Clinical Use Restrictions**: Do not use in clinical decision-making without professional validation
5. **Liability**: Users assume all responsibility for any clinical decisions based on this system's output
6. **Regulatory Compliance**: Ensure compliance with healthcare regulations (HIPAA, GDPR, etc.) when handling medical data

### Best Practices

- Always use in consultation with medical professionals
- Consider clinical context and patient history
- Use as a second opinion, not a primary diagnostic tool
- Keep human oversight at all decision points
- Regularly validate predictions against clinical outcomes
- Monitor for model drift and performance degradation

## 🔧 Troubleshooting

### Model Not Loading
```bash
# Check if model exists
ls models/brain_tumor_model_final.keras

# If missing, run training first
python train.py
```

### Out of Memory Error
- Reduce BATCH_SIZE in config.py
- Use GPU: Install CUDA-enabled TensorFlow
- Process smaller image resolutions

### Slow Predictions
- Enable GPU acceleration
- Reduce image resolution
- Batch process images

### Poor Accuracy
- Check data quality and labeling
- Increase training epochs
- Add more diverse training data
- Adjust learning rate

### Consistent "noTumor" Predictions (68-75% Confidence)
- **Cause**: Input normalization mismatch. EfficientNetB4 expects inputs in the `[0, 255]` range. If prediction preprocessing divides inputs by `255.0` to force them into a `[0, 1]` range, the model will fail to recognize features and default to its bias distribution.
- **Solution**: This has been resolved in v1.0.1+ by removing the `/ 255.0` normalization step in `predict.py` and correctly handling integer image arrays without artificial scaling.

## 📚 Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Image processing
- **Scikit-learn**: ML utilities
- **Streamlit**: Web application
- **Plotly**: Interactive visualization
- **NumPy/Pandas**: Data manipulation

## 📝 License

This project is provided for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Please:
1. Test thoroughly before submitting
2. Follow PEP 8 style guidelines
3. Add documentation
4. Include usage examples

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review model logs
3. Verify data format and quality
4. Contact development team

---

**Last Updated**: 2026  
**Status**: Production Ready  
**Version**: 1.0.3

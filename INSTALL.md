# Installation & Setup Guide

Complete step-by-step guide to set up the Brain Tumor Detection System.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Verify Installation](#verify-installation)
4. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB (16 GB recommended)
- **Disk Space**: 20 GB (for model and data)

### Optional - GPU Acceleration
- **CUDA**: 11.0 or higher (NVIDIA GPUs only)
- **cuDNN**: 8.0 or higher
- **TensorFlow GPU**: Significantly faster training

### Recommended Hardware
- **GPU**: NVIDIA GeForce RTX 2070 or better
- **RAM**: 16 GB
- **SSD**: 256 GB

## Installation Steps

### Step 1: Clone/Download Repository
```bash
cd d:\pythonprojects\brain-tumor
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Why?** Virtual environments isolate project dependencies from system Python.

### Step 3: Upgrade pip
```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

**If installation fails:**
- Update pip: `pip install --upgrade pip`
- Try with specific version: `pip install tensorflow==2.13.0`
- Check internet connection
- On Windows, ensure Visual C++ redistributables are installed

### Step 5: Verify Data Structure

Your `data/` folder should look like:
```
data/
├── Training/
│   ├── glioma/          (200-300 images)
│   ├── meningioma/      (200-300 images)
│   ├── notumor/         (200-300 images)
│   └── pituitary/       (200-300 images)
└── Testing/
    ├── glioma/          (50-100 images)
    ├── meningioma/      (50-100 images)
    ├── notumor/         (50-100 images)
    └── pituitary/       (50-100 images)
```

All images should be in `.jpg` format.

### Step 6: Environment Check
```bash
python quick_start.py --check
```

This will verify:
- ✓ Python version
- ✓ Data directory structure
- ✓ Required packages
- ✓ GPU availability (if applicable)

## Verify Installation

### Run Quick Check
```bash
python quick_start.py --check
```

### Check TensorFlow Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}'); print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\"))}');"
```

### Check All Packages
```bash
python -c "import keras, cv2, streamlit; print('All packages OK!')"
```

### View System Info
```bash
python quick_start.py --info
```

## Optional - GPU Setup (Recommended for Faster Training)

### For NVIDIA GPUs (Windows)

1. **Install CUDA Toolkit**
   - Download from: https://developer.nvidia.com/cuda-toolkit
   - Version: CUDA 11.8 or higher
   - Follow installation wizard

2. **Install cuDNN**
   - Download from: https://developer.nvidia.com/cudnn
   - Extract to: `C:\Program Files\NVIDIA\CUDA\v11.8`
   - Add to PATH environment variable

3. **Verify GPU Setup**
```bash
python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU'))) > 0"
```

### Install GPU-optimized TensorFlow
```bash
pip install tensorflow[and-cuda]
```

## Running the System

### 1. Train the Model (First Time)
```bash
python train.py
```
- Estimated time: 30 minutes to 2 hours (CPU), 5-15 minutes (GPU)
- Output: Trained model saved to `models/brain_tumor_model_final.h5`

### 2. Start Web Application
```bash
streamlit run app.py
```
- Opens at: `http://localhost:8501`
- Upload images to get predictions

### 3. Evaluate Model
```bash
python evaluate.py
```
- Tests on unseen data
- Generates performance reports

## Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow>=2.13.0
```

Or for GPU:
```bash
pip install tensorflow[and-cuda]>=2.13.0
```

### Issue: "No module named 'keras'"

**Solution:**
```bash
pip install keras>=2.13.0
```

### Issue: Out of Memory Error

**Solutions:**
1. Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # Default is 32
```

2. Use GPU:
```bash
pip install tensorflow[and-cuda]
```

3. Process images in smaller batches

### Issue: GPU Not Detected

**Check:**
1. Install NVIDIA drivers: https://www.nvidia.com/Download/driverDetails.aspx
2. Install CUDA Toolkit matching TensorFlow version
3. Verify: `python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"`

**Alternative:** Use CPU (slower but works everywhere)

### Issue: "Model not found" when running app

**Solution:**
```bash
python train.py  # Train the model first
```

### Issue: Slow Performance

**Optimization:**
1. Enable GPU (see GPU Setup section)
2. Reduce image size in `config.py`:
```python
IMG_WIDTH = 128
IMG_HEIGHT = 128
```

3. Use batch predictions instead of single predictions

### Issue: Data Loading Error

**Check:**
1. Image format: Must be `.jpg` (not `.jpeg` or `.png`)
2. Directory structure: Follow the exact structure shown above
3. File names: No special characters, spaces OK
4. Image validity: Try opening images in an image viewer

### Issue: ImportError on Streamlit

**Solution:**
```bash
pip install streamlit==1.28.0 --force-reinstall
```

## Next Steps

After successful installation:

1. **Run quick start guide:**
   ```bash
   python quick_start.py --guide
   ```

2. **Check your setup:**
   ```bash
   python quick_start.py --check
   ```

3. **Train the model:**
   ```bash
   python train.py
   ```

4. **Try the web app:**
   ```bash
   streamlit run app.py
   ```

## Getting Help

### Resources
- TensorFlow: https://www.tensorflow.org/
- Streamlit: https://docs.streamlit.io/
- OpenCV: https://docs.opencv.org/

### Common Issues Repository
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more detailed solutions.

### Performance Tips

**For Faster Training:**
- Use GPU: Install CUDA and TensorFlow GPU version
- Reduce image resolution (224×224 → 128×128)
- Increase batch size (if you have GPU memory)
- Use fewer epochs initially for testing

**For Better Accuracy:**
- Use full-resolution images (224×224)
- Increase epochs
- Add more training data
- Use data augmentation (enabled by default)
- Fine-tune pre-trained layers (done automatically)

## Verification Checklist

After installation, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All packages installed (`pip freeze | grep -i tensorflow`)
- [ ] Data directory structure correct
- [ ] Data images in .jpg format
- [ ] `python quick_start.py --check` passes
- [ ] TensorFlow can access GPU (if applicable)
- [ ] Can import all modules without errors

## Configuration

Optional: Customize `config.py`:

```python
# Image size
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Training
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.60
```

---

**Installation Complete!** 🎉

You're now ready to:
1. Train the model: `python train.py`
2. Run the web app: `streamlit run app.py`
3. Make predictions: `python predict.py`

For detailed usage, see [README.md](README.md)

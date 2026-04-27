"""
Quick start guide and utility functions
"""

import os
import sys
from pathlib import Path
import argparse


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   🧠 BRAIN TUMOR DETECTION SYSTEM v1.0                 ║
    ║                                                          ║
    ║   AI-Powered Medical Imaging Analysis                  ║
    ║   Built with EfficientNetB4 Transfer Learning         ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Check if environment is properly set up"""
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required (found {sys.version_info.major}.{sys.version_info.minor})")
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        issues.append("Data directory not found")
    else:
        print("✓ Data directory found")
        
        # Check subdirectories
        train_dir = data_dir / "Training"
        test_dir = data_dir / "Testing"
        
        if train_dir.exists():
            train_count = sum(1 for _ in train_dir.rglob("*.jpg"))
            print(f"  - Training images: {train_count}")
        
        if test_dir.exists():
            test_count = sum(1 for _ in test_dir.rglob("*.jpg"))
            print(f"  - Testing images: {test_count}")
    
    # Check models directory
    models_dir = Path("models")
    if models_dir.exists():
        model_file = models_dir / "brain_tumor_model_final.h5"
        if model_file.exists():
            print("✓ Trained model found")
        else:
            print("  ℹ Trained model not found (run 'python train.py' to train)")
    
    # Check required packages
    print("\n" + "Checking packages...")
    required_packages = ['tensorflow', 'keras', 'numpy', 'streamlit', 'opencv']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            issues.append(f"Package '{package}' not installed")
    
    if issues:
        print("\n" + "⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ Environment check passed!")
        return True


def print_usage_guide():
    """Print usage guide"""
    guide = """
    
╔══════════════════════════════════════════════════════════╗
║                    QUICK START GUIDE                     ║
╚══════════════════════════════════════════════════════════╝

1. TRAINING THE MODEL
   ────────────────────
   python train.py
   
   This will:
   - Load and preprocess all training/testing images
   - Build EfficientNetB4 model with transfer learning
   - Train with data augmentation
   - Fine-tune with unfrozen layers
   - Evaluate and save the model
   - Generate performance plots
   
   Expected time: 30 minutes to several hours (depends on GPU)

2. WEB APPLICATION
   ────────────────
   streamlit run app.py
   
   Then open: http://localhost:8501
   
   Features:
   - Single image prediction
   - Batch image analysis
   - Interactive visualizations
   - Detailed diagnostic reports

3. EVALUATION
   ──────────
   python evaluate.py
   
   This will:
   - Load the trained model
   - Test on unseen images
   - Generate detailed metrics
   - Create ROC curves and confusion matrices

4. PROGRAMMATIC PREDICTION
   ──────────────────────
   
   from predict import BrainTumorPredictor
   
   predictor = BrainTumorPredictor("models/brain_tumor_model_final.keras")
   result = predictor.predict("path/to/image.jpg")
   
   print(f"Class: {result['predicted_class']}")
   print(f"Confidence: {result['confidence']:.2%}")

════════════════════════════════════════════════════════════

DATASET STRUCTURE REQUIRED:

data/
├── Training/
│   ├── glioma/          (all glioma training images)
│   ├── meningioma/      (all meningioma training images)
│   ├── notumor/         (all notumor training images)
│   └── pituitary/       (all pituitary training images)
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/

════════════════════════════════════════════════════════════

CONFIGURATION

Edit config.py to customize:
- Image size (default: 224×224)
- Batch size (default: 32)
- Learning rate (default: 1e-4)
- Epochs (default: 100)
- Data augmentation parameters
- Confidence threshold (default: 0.60)

════════════════════════════════════════════════════════════

MODEL ARCHITECTURE

EfficientNetB4 (Pre-trained on ImageNet)
    ↓
Dense Layer (512 units) + Batch Norm + Dropout(0.4)
    ↓
Dense Layer (256 units) + Batch Norm + Dropout(0.3)
    ↓
Dense Layer (128 units) + Batch Norm + Dropout(0.2)
    ↓
Output Layer (Softmax, 4 classes)

════════════════════════════════════════════════════════════

IMPORTANT DISCLAIMER

⚠️  This system is for RESEARCH and ASSISTANCE purposes only
⚠️  NOT a replacement for professional medical diagnosis
⚠️  Always consult qualified medical professionals
⚠️  Results should be verified by radiologists

════════════════════════════════════════════════════════════

TROUBLESHOOTING

Issue: "Model not found"
→ Solution: Run 'python train.py' to train the model

Issue: "Out of memory"
→ Solution: Reduce BATCH_SIZE in config.py or use GPU

Issue: "Slow predictions"
→ Solution: Enable GPU or reduce image resolution

Issue: "Poor accuracy"
→ Solution: Check data quality, increase epochs, add more data

════════════════════════════════════════════════════════════
    """
    print(guide)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Brain Tumor Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py --check      Check environment
  python quick_start.py --guide      Show usage guide
  python quick_start.py --info       Show system information
        """
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if environment is properly set up'
    )
    
    parser.add_argument(
        '--guide',
        action='store_true',
        help='Show quick start guide'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show system information'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.check:
        check_environment()
    elif args.guide:
        print_usage_guide()
    elif args.info:
        print_system_info()
    else:
        # Default: show guide
        print_usage_guide()
        print("\nRun 'python quick_start.py --check' to verify your setup")


def print_system_info():
    """Print system information"""
    import platform
    import tensorflow as tf
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    print(f"\nOS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Machine: {platform.machine()}")
    
    print(f"\nTensorFlow: {tf.__version__}")
    
    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("GPUs: None (will use CPU)")
    
    # Project info
    print("\n" + "-"*60)
    print("PROJECT INFO")
    print("-"*60)
    
    data_dir = Path("data")
    if data_dir.exists():
        train_count = sum(1 for _ in data_dir.rglob("*.jpg") if "Training" in str(_))
        test_count = sum(1 for _ in data_dir.rglob("*.jpg") if "Testing" in str(_))
        print(f"Training images: {train_count}")
        print(f"Testing images: {test_count}")
    
    model_file = Path("models/brain_tumor_model_final.h5")
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024*1024)
        print(f"Model file: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
